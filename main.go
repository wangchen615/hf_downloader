package main

import (
	"fmt"
	"os"

	"github.com/wangchen615/hf_download/hfdownloader"
)

func main() {
	// Default model repository
	modelRepo := "openai-community/gpt2"
	revision := "main"
	var customPath string

	// Parse command line arguments
	switch len(os.Args) {
	case 4:
		customPath = os.Args[3]
		fallthrough
	case 3:
		revision = os.Args[2]
		fallthrough
	case 2:
		modelRepo = os.Args[1]
	}

	// Create downloader instance
	downloader := hfdownloader.NewDownloader()

	// Set custom path if provided
	if customPath != "" {
		downloader.SetCustomPath(customPath)
	}

	// Download the model
	fmt.Printf("Downloading model: %s (revision: %s)\n", modelRepo, revision)
	if customPath != "" {
		fmt.Printf("Download location: %s\n", customPath)
	}

	downloadPath, err := downloader.Download(modelRepo, revision)
	if err != nil {
		fmt.Printf("Error downloading model: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Downloaded to: %s\n", downloadPath)
}
