package tools

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/media"
	"github.com/sipeed/picoclaw/pkg/providers"
)

type ToolEntry struct {
	Tool   Tool
	IsCore bool
	TTL    int
}

type ToolRegistry struct {
	tools      map[string]*ToolEntry
	mu         sync.RWMutex
	version    atomic.Uint64 // incremented on Register/RegisterHidden for cache invalidation
	mediaStore media.MediaStore
}

type mediaStoreAware interface {
	SetMediaStore(store media.MediaStore)
}

func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools: make(map[string]*ToolEntry),
	}
}

func (r *ToolRegistry) Register(tool Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	name := tool.Name()
	if _, exists := r.tools[name]; exists {
		logger.WarnCF("tools", "Tool registration overwrites existing tool",
			map[string]any{"name": name})
	}
	r.tools[name] = &ToolEntry{
		Tool:   tool,
		IsCore: true,
		TTL:    0, // Core tools do not use TTL
	}
	if aware, ok := tool.(mediaStoreAware); ok && r.mediaStore != nil {
		aware.SetMediaStore(r.mediaStore)
	}
	r.version.Add(1)
	logger.DebugCF("tools", "Registered core tool", map[string]any{"name": name})
}

// RegisterHidden saves hidden tools (visible only via TTL)
func (r *ToolRegistry) RegisterHidden(tool Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	name := tool.Name()
	if _, exists := r.tools[name]; exists {
		logger.WarnCF("tools", "Hidden tool registration overwrites existing tool",
			map[string]any{"name": name})
	}
	r.tools[name] = &ToolEntry{
		Tool:   tool,
		IsCore: false,
		TTL:    0,
	}
	if aware, ok := tool.(mediaStoreAware); ok && r.mediaStore != nil {
		aware.SetMediaStore(r.mediaStore)
	}
	r.version.Add(1)
	logger.DebugCF("tools", "Registered hidden tool", map[string]any{"name": name})
}

// SetMediaStore injects a MediaStore into all registered tools that can
// consume it, and remembers it for future registrations.
func (r *ToolRegistry) SetMediaStore(store media.MediaStore) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.mediaStore = store
	for _, entry := range r.tools {
		if aware, ok := entry.Tool.(mediaStoreAware); ok {
			aware.SetMediaStore(store)
		}
	}
}

// PromoteTools atomically sets the TTL for multiple non-core tools.
// This prevents a concurrent TickTTL from decrementing between promotions.
func (r *ToolRegistry) PromoteTools(names []string, ttl int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	promoted := 0
	for _, name := range names {
		if entry, exists := r.tools[name]; exists {
			if !entry.IsCore {
				entry.TTL = ttl
				promoted++
			}
		}
	}
	logger.DebugCF(
		"tools",
		"PromoteTools completed",
		map[string]any{"requested": len(names), "promoted": promoted, "ttl": ttl},
	)
}

// TickTTL decreases TTL only for non-core tools
func (r *ToolRegistry) TickTTL() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, entry := range r.tools {
		if !entry.IsCore && entry.TTL > 0 {
			entry.TTL--
		}
	}
}

// Version returns the current registry version (atomically).
func (r *ToolRegistry) Version() uint64 {
	return r.version.Load()
}

// HiddenToolSnapshot holds a consistent snapshot of hidden tools and the
// registry version at which it was taken. Used by BM25SearchTool cache.
type HiddenToolSnapshot struct {
	Docs    []HiddenToolDoc
	Version uint64
}

// HiddenToolDoc is a lightweight representation of a hidden tool for search indexing.
type HiddenToolDoc struct {
	Name        string
	Description string
}

// SnapshotHiddenTools returns all non-core tools and the current registry
// version under a single read-lock, guaranteeing consistency between the
// two values.
func (r *ToolRegistry) SnapshotHiddenTools() HiddenToolSnapshot {
	r.mu.RLock()
	defer r.mu.RUnlock()
	docs := make([]HiddenToolDoc, 0, len(r.tools))
	for name, entry := range r.tools {
		if !entry.IsCore {
			docs = append(docs, HiddenToolDoc{
				Name:        name,
				Description: entry.Tool.Description(),
			})
		}
	}
	return HiddenToolSnapshot{
		Docs:    docs,
		Version: r.version.Load(),
	}
}

func (r *ToolRegistry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entry, ok := r.tools[name]
	if !ok {
		return nil, false
	}
	// Hidden tools with expired TTL are not callable.
	if !entry.IsCore && entry.TTL <= 0 {
		return nil, false
	}
	return entry.Tool, true
}

func (r *ToolRegistry) Execute(ctx context.Context, name string, args map[string]any) *ToolResult {
	return r.ExecuteWithContext(ctx, name, args, "", "", nil)
}

// buildShellEnv builds a complete shell environment similar to `source ~/.zshrc`.
// It runs an interactive shell to capture the full environment with all PATH
// extensions and shell-specific variables.
func buildShellEnv() []string {
	// Start with current process environment
	env := os.Environ()

	// Parse current environment into a map for easy manipulation
	envMap := make(map[string]string)
	var pathValue string
	for _, e := range env {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) == 2 {
			envMap[parts[0]] = parts[1]
			if parts[0] == "PATH" {
				pathValue = parts[1]
			}
		}
	}

	// Ensure standard macOS/Unix paths are included in PATH
	standardPaths := []string{"/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"}
	if pathValue != "" {
		existingPaths := strings.Split(pathValue, ":")
		existingPathSet := make(map[string]bool)
		for _, p := range existingPaths {
			if p != "" {
				existingPathSet[p] = true
			}
		}
		// Prepend standard paths that are not already in PATH
		var newPathParts []string
		for _, sp := range standardPaths {
			if !existingPathSet[sp] {
				newPathParts = append(newPathParts, sp)
			}
		}
		// Add existing paths after standard paths
		newPathParts = append(newPathParts, strings.Split(pathValue, ":")...)
		pathValue = strings.Join(newPathParts, ":")
	} else {
		pathValue = strings.Join(standardPaths, ":")
	}
	envMap["PATH"] = pathValue

	// Try to source shell config files directly if they exist
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/zsh" // Default to zsh on macOS
	}

	// Build command to source config files and output environment
	// Try .zshrc first, then .bashrc, then .profile
	var sourceCmd string
	homeDir, _ := os.UserHomeDir()
	if homeDir != "" {
		var sourceFiles []string
		zshrc := filepath.Join(homeDir, ".zshrc")
		bashrc := filepath.Join(homeDir, ".bashrc")
		profile := filepath.Join(homeDir, ".profile")
		bashProfile := filepath.Join(homeDir, ".bash_profile")

		if _, err := os.Stat(zshrc); err == nil && strings.HasSuffix(shell, "zsh") {
			sourceFiles = append(sourceFiles, zshrc)
		}
		if _, err := os.Stat(bashrc); err == nil && (strings.HasSuffix(shell, "bash") || len(sourceFiles) == 0) {
			sourceFiles = append(sourceFiles, bashrc)
		}
		if _, err := os.Stat(profile); err == nil && len(sourceFiles) == 0 {
			sourceFiles = append(sourceFiles, profile)
		}
		if _, err := os.Stat(bashProfile); err == nil && len(sourceFiles) == 0 {
			sourceFiles = append(sourceFiles, bashProfile)
		}

		if len(sourceFiles) > 0 {
			for _, f := range sourceFiles {
				sourceCmd += fmt.Sprintf("[ -f %q ] && . %q; ", f, f)
			}
			sourceCmd += "env"
		}
	}

	// Use interactive shell to load profile files (.zshrc, .bash_profile, etc.)
	cmd := exec.Command(shell, "-ic", sourceCmd)
	cmd.Env = os.Environ() // Use original env, not modified one
	output, err := cmd.Output()
	if err == nil && len(output) > 0 {
		// Parse the output into environment variables
		lines := string(output)
		parsedEnv := make([]string, 0)
		for _, line := range strings.Split(lines, "\n") {
			if line != "" && strings.Contains(line, "=") {
				parsedEnv = append(parsedEnv, line)
			}
		}
		if len(parsedEnv) > 0 {
			env = parsedEnv
		}
	}

	// Final fallback: ensure PATH has standard paths even if shell command failed
	// Re-parse env to ensure PATH is correct
	finalEnvMap := make(map[string]string)
	var finalHasPath bool
	for _, e := range env {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) == 2 {
			finalEnvMap[parts[0]] = parts[1]
			if parts[0] == "PATH" {
				finalHasPath = true
			}
		}
	}

	if finalHasPath {
		currentPath := finalEnvMap["PATH"]
		existingPaths := strings.Split(currentPath, ":")
		existingPathSet := make(map[string]bool)
		for _, p := range existingPaths {
			if p != "" {
				existingPathSet[p] = true
			}
		}
		// Prepend standard paths that are not already in PATH
		var newPathParts []string
		for _, sp := range standardPaths {
			if !existingPathSet[sp] {
				newPathParts = append(newPathParts, sp)
			}
		}
		newPathParts = append(newPathParts, strings.Split(currentPath, ":")...)
		finalEnvMap["PATH"] = strings.Join(newPathParts, ":")
	} else {
		finalEnvMap["PATH"] = pathValue
	}

	// Convert map back to slice
	finalEnv := make([]string, 0, len(finalEnvMap))
	for k, v := range finalEnvMap {
		info := fmt.Sprintf("%s=%s", k, v)
		//logger.InfoC("env_red", info)
		finalEnv = append(finalEnv, info)
	}

	return finalEnv
}

// ExecuteWithContext executes a tool with channel/chatID context and optional async callback.
// If the tool implements AsyncExecutor and a non-nil callback is provided,
// ExecuteAsync is called instead of Execute — the callback is a parameter,
// never stored as mutable state on the tool.
func (r *ToolRegistry) ExecuteWithContext(
	ctx context.Context,
	name string,
	args map[string]any,
	channel, chatID string,
	asyncCallback AsyncCallback,
) *ToolResult {
	logger.InfoCF("tool", "tool|exec_start",
		map[string]any{
			"tool": name,
			"args": args,
		})

	tool, ok := r.Get(name)
	if !ok {
		logger.ErrorCF("tool", "tool|not found",
			map[string]any{
				"tool": name,
			})
		return ErrorResult(fmt.Sprintf("tool %q not found", name)).WithError(fmt.Errorf("tool not found"))
	}

	// Validate arguments against the tool's declared schema.
	if err := validateToolArgs(tool.Parameters(), args); err != nil {
		logger.WarnCF("tool", "Tool argument validation failed",
			map[string]any{"tool": name, "error": err.Error()})
		return ErrorResult(fmt.Sprintf("invalid arguments for tool %q: %s", name, err)).
			WithError(fmt.Errorf("argument validation failed: %w", err))
	}

	// Inject channel/chatID into ctx so tools read them via ToolChannel(ctx)/ToolChatID(ctx).
	// Always inject — tools validate what they require.
	ctx = WithToolContext(ctx, channel, chatID)

	// Inject shell environment into ctx so tools can access full environment
	// (similar to `source ~/.zshrc`). This ensures commands like npm, go, etc.
	// can be found in PATH and other shell-specific variables are available.
	ctx = WithToolEnv(ctx, buildShellEnv())

	// If tool implements AsyncExecutor and callback is provided, use ExecuteAsync.
	// The callback is a call parameter, not mutable state on the tool instance.
	var result *ToolResult
	start := time.Now()

	// Use recover to catch any panics during tool execution
	// This prevents tool crashes from killing the entire agent
	func() {
		defer func() {
			if re := recover(); re != nil {
				errMsg := fmt.Sprintf("Tool '%s' crashed with panic: %v", name, re)
				logger.ErrorCF("tool", "Tool execution panic recovered",
					map[string]any{
						"tool":  name,
						"panic": fmt.Sprintf("%v", re),
					})
				result = &ToolResult{
					ForLLM:  errMsg,
					ForUser: errMsg,
					IsError: true,
					Err:     fmt.Errorf("panic: %v", re),
				}
			}
		}()

		if asyncExec, ok := tool.(AsyncExecutor); ok && asyncCallback != nil {
			logger.DebugCF("tool", "Executing async tool via ExecuteAsync",
				map[string]any{
					"tool": name,
				})
			result = asyncExec.ExecuteAsync(ctx, args, asyncCallback)
		} else {
			result = tool.Execute(ctx, args)
		}
	}()

	// Handle nil result (should not happen, but defensive)
	if result == nil {
		result = &ToolResult{
			ForLLM:  fmt.Sprintf("Tool '%s' returned nil result unexpectedly", name),
			ForUser: fmt.Sprintf("Tool '%s' returned nil result unexpectedly", name),
			IsError: true,
			Err:     fmt.Errorf("nil result from tool"),
		}
	}

	result = normalizeToolResult(result, name, r.mediaStore, channel, chatID)

	duration := time.Since(start)

	// Log based on result type
	if result.IsError {
		logger.ErrorCF("tool", "tool|exec_failed",
			map[string]any{
				"tool":     name,
				"lapse": duration.Milliseconds(),
				"error":    result.ForLLM,
			})
	} else if result.Async {
		logger.InfoCF("tool", "tool|async_start",
			map[string]any{
				"tool":     name,
				"duration": duration.Milliseconds(),
			})
	} else {
		logger.InfoCF("tool", "tool|exec_done",
			map[string]any{
				"tool":          name,
				"lapse_ms":   duration.Milliseconds(),
				"ret_len": len(result.ContentForLLM()),
			})
	}

	return result
}

// sortedToolNames returns tool names in sorted order for deterministic iteration.
// This is critical for KV cache stability: non-deterministic map iteration would
// produce different system prompts and tool definitions on each call, invalidating
// the LLM's prefix cache even when no tools have changed.
func (r *ToolRegistry) sortedToolNames() []string {
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (r *ToolRegistry) GetDefinitions() []map[string]any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	sorted := r.sortedToolNames()
	definitions := make([]map[string]any, 0, len(sorted))
	for _, name := range sorted {
		entry := r.tools[name]

		if !entry.IsCore && entry.TTL <= 0 {
			continue
		}

		definitions = append(definitions, ToolToSchema(r.tools[name].Tool))
	}
	return definitions
}

// ToProviderDefs converts tool definitions to provider-compatible format.
// This is the format expected by LLM provider APIs.
func (r *ToolRegistry) ToProviderDefs() []providers.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	sorted := r.sortedToolNames()
	definitions := make([]providers.ToolDefinition, 0, len(sorted))
	for _, name := range sorted {
		entry := r.tools[name]

		if !entry.IsCore && entry.TTL <= 0 {
			continue
		}

		schema := ToolToSchema(entry.Tool)

		// Safely extract nested values with type checks
		fn, ok := schema["function"].(map[string]any)
		if !ok {
			continue
		}

		name, _ := fn["name"].(string)
		desc, _ := fn["description"].(string)
		params, _ := fn["parameters"].(map[string]any)

		definitions = append(definitions, providers.ToolDefinition{
			Type: "function",
			Function: providers.ToolFunctionDefinition{
				Name:        name,
				Description: desc,
				Parameters:  params,
			},
		})
	}
	return definitions
}

// List returns a list of all registered tool names.
func (r *ToolRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.sortedToolNames()
}

// Clone creates an independent copy of the registry containing the same tool
// entries (shallow copy of each ToolEntry). This is used to give subagents a
// snapshot of the parent agent's tools without sharing the same registry —
// tools registered on the parent after cloning (e.g. spawn, spawn_status)
// will NOT be visible to the clone, preventing recursive subagent spawning.
// The version counter is reset to 0 in the clone as it's a new independent registry.
func (r *ToolRegistry) Clone() *ToolRegistry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	clone := &ToolRegistry{
		tools:      make(map[string]*ToolEntry, len(r.tools)),
		mediaStore: r.mediaStore,
	}
	for name, entry := range r.tools {
		clone.tools[name] = &ToolEntry{
			Tool:   entry.Tool,
			IsCore: entry.IsCore,
			TTL:    entry.TTL,
		}
	}
	return clone
}

// Count returns the number of registered tools.
func (r *ToolRegistry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}

// GetSummaries returns human-readable summaries of all registered tools.
// Returns a slice of "name - description" strings.
func (r *ToolRegistry) GetSummaries() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	sorted := r.sortedToolNames()
	summaries := make([]string, 0, len(sorted))
	for _, name := range sorted {
		entry := r.tools[name]

		if !entry.IsCore && entry.TTL <= 0 {
			continue
		}

		summaries = append(summaries, fmt.Sprintf("- `%s` - %s", entry.Tool.Name(), entry.Tool.Description()))
	}
	return summaries
}

// GetAll returns all registered tools (both core and non-core with TTL > 0).
// Used by SubTurn to inherit parent's tool set.
func (r *ToolRegistry) GetAll() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	sorted := r.sortedToolNames()
	tools := make([]Tool, 0, len(sorted))
	for _, name := range sorted {
		entry := r.tools[name]

		// Include core tools and non-core tools with active TTL
		if entry.IsCore || entry.TTL > 0 {
			tools = append(tools, entry.Tool)
		}
	}
	return tools
}
