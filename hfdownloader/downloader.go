package hfdownloader

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

const (
	HF_API_URL          = "https://huggingface.co/api"
	HF_TOKEN_ENV        = "HF_TOKEN"
	DOWNLOAD_CHUNK_SIZE = 8192
)

// Downloader handles downloading models from Hugging Face
type Downloader struct {
	customPath     string
	ignorePatterns []string
	client         *http.Client
}

// NewDownloader creates a new instance of Downloader with default settings
func NewDownloader() *Downloader {
	return &Downloader{
		ignorePatterns: []string{`\.md$`, `\.txt$`},
		client: &http.Client{
			Timeout: 30 * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= 10 {
					return fmt.Errorf("too many redirects")
				}
				if len(via) > 0 {
					for key, val := range via[0].Header {
						if _, ok := req.Header[key]; !ok {
							req.Header[key] = val
						}
					}
				}
				return nil
			},
		},
	}
}

// SetCustomPath sets a custom download path instead of using the default cache directory
func (d *Downloader) SetCustomPath(path string) {
	d.customPath = path
}

// SetIgnorePatterns sets custom patterns for files to ignore during download
func (d *Downloader) SetIgnorePatterns(patterns []string) {
	d.ignorePatterns = patterns
}

// Download downloads a model from Hugging Face
func (d *Downloader) Download(modelRepo, revision string) (string, error) {
	if revision == "" {
		revision = "main"
	}

	// Compile ignore patterns
	ignoreRegexps := make([]*regexp.Regexp, len(d.ignorePatterns))
	for i, pattern := range d.ignorePatterns {
		ignoreRegexps[i] = regexp.MustCompile(pattern)
	}

	// Get model information
	modelInfo, commitHash, err := d.getModelInfo(modelRepo, revision)
	if err != nil {
		return "", fmt.Errorf("could not get model info: %w", err)
	}

	// Determine storage location
	var storageFolder string
	if d.customPath != "" {
		storageFolder = d.customPath
	} else {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("could not get home directory: %w", err)
		}
		repoPath := d.repoFolderName(modelRepo, "model")
		storageFolder = filepath.Join(homeDir, ".cache", "huggingface", "hub", repoPath)
	}

	// Create directory structure
	for _, dir := range []string{
		storageFolder,
		filepath.Join(storageFolder, "blobs"),
		filepath.Join(storageFolder, "refs"),
		filepath.Join(storageFolder, "snapshots"),
	} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", fmt.Errorf("could not create directory %s: %w", dir, err)
		}
	}

	// Cache the commit hash
	if revision != commitHash {
		refPath := filepath.Join(storageFolder, "refs", revision)
		os.MkdirAll(filepath.Dir(refPath), 0755)
		if err := os.WriteFile(refPath, []byte(commitHash), 0644); err != nil {
			fmt.Printf("Warning: Could not write revision reference: %v\n", err)
		}
	}

	// Create snapshot directory
	snapshotDir := filepath.Join(storageFolder, "snapshots", commitHash)
	if err := os.MkdirAll(snapshotDir, 0755); err != nil {
		return "", fmt.Errorf("could not create snapshot directory: %w", err)
	}

	// Download files in parallel
	var wg sync.WaitGroup
	errorChan := make(chan error, len(modelInfo.Siblings))

	for _, file := range modelInfo.Siblings {
		shouldDownload := true
		for _, pattern := range ignoreRegexps {
			if pattern.MatchString(file.RID) {
				shouldDownload = false
				break
			}
		}

		if shouldDownload {
			wg.Add(1)
			go func(filename string, blobId string, size int64) {
				defer wg.Done()
				if err := d.downloadFile(modelRepo, revision, filename, blobId, size, storageFolder, snapshotDir); err != nil {
					errorChan <- fmt.Errorf("error processing %s: %w", filename, err)
				}
			}(file.RID, file.Blob, file.Size)
		}
	}

	wg.Wait()
	close(errorChan)

	for err := range errorChan {
		return "", err
	}

	return storageFolder, nil
}

func (d *Downloader) downloadFile(modelRepo, revision, filename, blobId string, size int64, storageFolder, snapshotDir string) error {
	relativePath := strings.ReplaceAll(filename, "/", string(os.PathSeparator))
	blobPath := filepath.Join(storageFolder, "blobs", blobId)
	pointerPath := filepath.Join(snapshotDir, relativePath)

	if err := os.MkdirAll(filepath.Dir(pointerPath), 0755); err != nil {
		return fmt.Errorf("could not create directory: %w", err)
	}

	blobExists := false
	if _, err := os.Stat(blobPath); err == nil {
		blobExists = true
	}

	if _, err := os.Stat(pointerPath); err == nil && !blobExists {
		fmt.Printf("Warning: Pointer exists but blob missing for %s, redownloading\n", filename)
	} else if err == nil && blobExists {
		fmt.Printf("File already exists: %s\n", filename)
		return nil
	}

	if !blobExists {
		url := fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s", modelRepo, revision, filename)
		metadata, err := d.getFileMetadata(url)
		if err != nil {
			return fmt.Errorf("error getting metadata: %w", err)
		}

		fmt.Printf("Downloading: %s (%.2f MB)\n", filename, float64(metadata.Size)/1024/1024)

		tempPath := blobPath + ".incomplete"
		if err := d.downloadWithProgress(url, tempPath, metadata.Size); err != nil {
			return fmt.Errorf("error downloading: %w", err)
		}

		if err := os.Rename(tempPath, blobPath); err != nil {
			return fmt.Errorf("error renaming temp file: %w", err)
		}
	}

	if isSymlinkSupported() {
		relPath, err := filepath.Rel(filepath.Dir(pointerPath), blobPath)
		if err != nil {
			relPath = blobPath
		}

		if _, err := os.Stat(pointerPath); err == nil {
			os.Remove(pointerPath)
		}

		if err := os.Symlink(relPath, pointerPath); err != nil {
			fmt.Printf("Warning: Could not create symlink, copying file instead: %v\n", err)
			if err := copyFile(blobPath, pointerPath); err != nil {
				return fmt.Errorf("error copying: %w", err)
			}
		}
	} else {
		if err := copyFile(blobPath, pointerPath); err != nil {
			return fmt.Errorf("error copying: %w", err)
		}
	}

	fmt.Printf("Processed: %s\n", filename)
	return nil
}

// Helper functions...

func (d *Downloader) getModelInfo(modelRepo, revision string) (*ModelInfo, string, error) {
	url := fmt.Sprintf("%s/models/%s/tree/%s", HF_API_URL, modelRepo, revision)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, "", fmt.Errorf("error creating request: %w", err)
	}

	if token := os.Getenv(HF_TOKEN_ENV); token != "" {
		req.Header.Add("Authorization", "Bearer "+token)
	}
	req.Header.Add("User-Agent", "huggingface-go/0.1")

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("error making API request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("API returned status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	var files []struct {
		Type    string    `json:"type"`
		Path    string    `json:"path"`
		Oid     string    `json:"oid"`
		Size    int64     `json:"size"`
		LfsInfo *struct{} `json:"lfs"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		return nil, "", fmt.Errorf("error decoding API response: %w", err)
	}

	modelInfo := &ModelInfo{
		Siblings: make([]struct {
			RID  string    `json:"rfilename"`
			Size int64     `json:"size"`
			Blob string    `json:"blob_id"`
			LFS  *struct{} `json:"lfs,omitempty"`
		}, 0),
	}

	for _, file := range files {
		if file.Type == "file" {
			modelInfo.Siblings = append(modelInfo.Siblings, struct {
				RID  string    `json:"rfilename"`
				Size int64     `json:"size"`
				Blob string    `json:"blob_id"`
				LFS  *struct{} `json:"lfs,omitempty"`
			}{
				RID:  file.Path,
				Size: file.Size,
				Blob: file.Oid,
				LFS:  file.LfsInfo,
			})
		}
	}

	commitHash := resp.Header.Get("X-Repo-Commit")
	if commitHash == "" && len(files) > 0 {
		commitHash = files[0].Oid
	}
	if commitHash == "" {
		commitHash = revision
	}

	return modelInfo, commitHash, nil
}

func (d *Downloader) getFileMetadata(url string) (*HfFileMetadata, error) {
	req, err := http.NewRequest("HEAD", url, nil)
	if err != nil {
		return nil, fmt.Errorf("error creating HEAD request: %w", err)
	}

	if token := os.Getenv(HF_TOKEN_ENV); token != "" {
		req.Header.Add("Authorization", "Bearer "+token)
	}
	req.Header.Add("Accept-Encoding", "identity")

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making HEAD request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HEAD request returned status code: %d", resp.StatusCode)
	}

	metadata := &HfFileMetadata{
		CommitHash: resp.Header.Get("X-Repo-Commit"),
		Etag:       normalizeETag(resp.Header.Get("X-Linked-Etag")),
		Location:   resp.Request.URL.String(),
		Size:       parseInt64(resp.Header.Get("Content-Length")),
	}

	if metadata.Etag == "" {
		metadata.Etag = normalizeETag(resp.Header.Get("ETag"))
	}

	return metadata, nil
}

func (d *Downloader) downloadWithProgress(url, filepath string, expectedSize int64) error {
	if err := os.MkdirAll(path.Dir(filepath), 0755); err != nil {
		return fmt.Errorf("could not create directory: %w", err)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("could not create file: %w", err)
	}
	defer out.Close()

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("error creating request: %w", err)
	}

	if token := os.Getenv(HF_TOKEN_ENV); token != "" {
		req.Header.Add("Authorization", "Bearer "+token)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	var downloaded int64
	lastProgressUpdate := time.Now()
	progressInterval := 1 * time.Second

	buffer := make([]byte, DOWNLOAD_CHUNK_SIZE)
	for {
		n, err := resp.Body.Read(buffer)
		if err != nil && err != io.EOF {
			return fmt.Errorf("error reading response: %w", err)
		}

		if n > 0 {
			if _, err := out.Write(buffer[:n]); err != nil {
				return fmt.Errorf("error writing to file: %w", err)
			}

			downloaded += int64(n)
			if time.Since(lastProgressUpdate) > progressInterval {
				if expectedSize > 0 {
					percentage := float64(downloaded) / float64(expectedSize) * 100
					fmt.Printf("  %.1f%% (%d/%d bytes)\r", percentage, downloaded, expectedSize)
				} else {
					fmt.Printf("  %d bytes downloaded\r", downloaded)
				}
				lastProgressUpdate = time.Now()
			}
		}

		if err == io.EOF {
			break
		}
	}

	fmt.Println()

	if expectedSize > 0 && downloaded != expectedSize {
		return fmt.Errorf("download size mismatch: got %d bytes, expected %d bytes", downloaded, expectedSize)
	}

	return nil
}

func (d *Downloader) repoFolderName(repoID, repoType string) string {
	parts := []string{repoType + "s"}
	parts = append(parts, strings.Split(repoID, "/")...)
	return strings.Join(parts, "--")
}

// ModelInfo represents model repository information from API
type ModelInfo struct {
	Siblings []struct {
		RID  string    `json:"rfilename"`
		Size int64     `json:"size"`
		Blob string    `json:"blob_id"`
		LFS  *struct{} `json:"lfs,omitempty"`
	} `json:"siblings"`
}

// HfFileMetadata structure similar to Python's HfFileMetadata
type HfFileMetadata struct {
	CommitHash string
	Etag       string
	Location   string
	Size       int64
}

// Utility functions that don't need to be methods
func normalizeETag(etag string) string {
	if etag == "" {
		return ""
	}
	etag = strings.TrimPrefix(etag, "W/")
	return strings.Trim(etag, "\"")
}

func parseInt64(s string) int64 {
	var size int64
	if s != "" {
		fmt.Sscanf(s, "%d", &size)
	}
	return size
}

func isSymlinkSupported() bool {
	tempDir, err := os.MkdirTemp("", "symlink-test")
	if err != nil {
		return false
	}
	defer os.RemoveAll(tempDir)

	testFile := filepath.Join(tempDir, "test-file")
	if err := os.WriteFile(testFile, []byte("test"), 0644); err != nil {
		return false
	}

	testLink := filepath.Join(tempDir, "test-link")
	err = os.Symlink(testFile, testLink)
	return err == nil
}

func copyFile(src, dst string) error {
	source, err := os.Open(src)
	if err != nil {
		return err
	}
	defer source.Close()

	destination, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destination.Close()

	_, err = io.Copy(destination, source)
	return err
}
