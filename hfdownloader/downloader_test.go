package hfdownloader

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestNewDownloader(t *testing.T) {
	d := NewDownloader()

	// Check default ignore patterns
	if len(d.ignorePatterns) != 2 {
		t.Errorf("Expected 2 default ignore patterns, got %d", len(d.ignorePatterns))
	}

	// Check client timeout
	if d.client.Timeout != 30*time.Second {
		t.Errorf("Expected 30s timeout, got %v", d.client.Timeout)
	}
}

func TestSetCustomPath(t *testing.T) {
	d := NewDownloader()
	testPath := "/test/path"
	d.SetCustomPath(testPath)

	if d.customPath != testPath {
		t.Errorf("Expected custom path %s, got %s", testPath, d.customPath)
	}
}

func TestSetIgnorePatterns(t *testing.T) {
	d := NewDownloader()
	patterns := []string{"\\.bin$", "\\.onnx$"}
	d.SetIgnorePatterns(patterns)

	if len(d.ignorePatterns) != len(patterns) {
		t.Errorf("Expected %d patterns, got %d", len(patterns), len(d.ignorePatterns))
	}
	for i, pattern := range patterns {
		if d.ignorePatterns[i] != pattern {
			t.Errorf("Expected pattern %s, got %s", pattern, d.ignorePatterns[i])
		}
	}
}

func TestDownload(t *testing.T) {
	// Create a temporary directory for testing
	tmpDir, err := os.MkdirTemp("", "hf-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	d := NewDownloader()
	d.SetCustomPath(tmpDir)

	// Test with a small model to avoid long downloads in tests
	modelRepo := "hf-internal-testing/tiny-random-gpt2"
	revision := "main"

	downloadPath, err := d.Download(modelRepo, revision)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Check if the download path exists
	if _, err := os.Stat(downloadPath); os.IsNotExist(err) {
		t.Errorf("Download path does not exist: %s", downloadPath)
	}

	// Check if basic directory structure was created
	dirs := []string{
		filepath.Join(downloadPath, "blobs"),
		filepath.Join(downloadPath, "refs"),
		filepath.Join(downloadPath, "snapshots"),
	}

	for _, dir := range dirs {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			t.Errorf("Expected directory does not exist: %s", dir)
		}
	}
}

func TestDownloadWithInvalidRepo(t *testing.T) {
	d := NewDownloader()
	_, err := d.Download("invalid/repo/name", "main")
	if err == nil {
		t.Error("Expected error for invalid repository, got nil")
	}
}

func TestDownloadWithToken(t *testing.T) {
	// Skip if no token is set
	token := os.Getenv(HF_TOKEN_ENV)
	if token == "" {
		t.Skip("Skipping test: No Hugging Face token set")
	}

	d := NewDownloader()
	tmpDir, err := os.MkdirTemp("", "hf-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	d.SetCustomPath(tmpDir)

	// Try downloading a private model (this assumes the token has access)
	_, err = d.Download("your-private-model/test", "main")
	if err != nil {
		t.Errorf("Failed to download with token: %v", err)
	}
}

func TestCustomHTTPClient(t *testing.T) {
	d := NewDownloader()
	customClient := &http.Client{
		Timeout: 60 * time.Second,
	}
	d.client = customClient

	if d.client.Timeout != 60*time.Second {
		t.Errorf("Expected 60s timeout, got %v", d.client.Timeout)
	}
}
