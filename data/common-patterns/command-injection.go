// Common Command Injection Pattern - For Cache Warming
// This pattern should be detected as a security vulnerability

package main

import (
	"fmt"
	"os/exec"
)

// VULNERABLE: Direct command execution with user input
func runCommandUnsafe(userInput string) (string, error) {
	// DANGEROUS: Shell injection possible
	cmd := exec.Command("sh", "-c", "echo "+userInput)
	output, err := cmd.Output()
	return string(output), err
}

// VULNERABLE: Using user input in command arguments
func processFileUnsafe(filename string) error {
	// DANGEROUS: Path traversal and command injection
	cmd := exec.Command("cat", filename)
	return cmd.Run()
}

// SAFE: Using exec.Command with separate arguments
func runCommandSafe(arg string) (string, error) {
	// SAFE: Arguments are properly escaped
	cmd := exec.Command("echo", arg)
	output, err := cmd.Output()
	return string(output), err
}

// SAFE: Validating and sanitizing input
func processFileSafe(filename string) error {
	// Validate filename doesn't contain path traversal
	if containsPathTraversal(filename) {
		return fmt.Errorf("invalid filename")
	}
	
	// Use absolute path within allowed directory
	safePath := "/data/uploads/" + filepath.Base(filename)
	cmd := exec.Command("cat", safePath)
	return cmd.Run()
}

func containsPathTraversal(path string) bool {
	return strings.Contains(path, "..") || strings.HasPrefix(path, "/")
}
