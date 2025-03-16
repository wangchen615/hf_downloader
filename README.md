# HF Downloader

A lightweight and efficient Go package for downloading models from Hugging Face's model hub. This package provides a simple way to download machine learning models and their associated files from Hugging Face, with support for caching, parallel downloads, and custom storage locations.

## Features

- 🚀 Parallel file downloads for improved performance
- 💾 Local caching with smart file management
- 🔄 Support for specific model revisions
- 📁 Custom download path configuration
- 🎯 Configurable file filtering
- 🔑 Hugging Face token support for private models

## Installation

```bash
go get github.com/wangchen615/hf_downloader
```

## Usage

### Basic Usage

```go
package main

import (
    "fmt"
    "github.com/wangchen615/hf_downloader/hfdownloader"
)

func main() {
    // Create a new downloader instance
    downloader := hfdownloader.NewDownloader()
    
    // Download a model (using default revision "main")
    downloadPath, err := downloader.Download("openai-community/gpt2", "main")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Model downloaded to: %s\n", downloadPath)
}
```

### Using Custom Download Path

```go
downloader := hfdownloader.NewDownloader()
downloader.SetCustomPath("/path/to/your/models")
downloadPath, err := downloader.Download("openai-community/gpt2", "main")
```

### Command Line Usage

The package includes a command-line interface for easy model downloading:

```bash
# Basic usage (downloads openai-community/gpt2)
go run main.go

# Download specific model
go run main.go bert-base-uncased

# Download specific model revision
go run main.go bert-base-uncased v1.0

# Download to custom path
go run main.go bert-base-uncased main /path/to/download
```

## Authentication

For private models, set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

## Default Behavior

- Models are cached in `~/.cache/huggingface/hub` by default
- Files are downloaded in parallel for better performance
- Text files (`.md`, `.txt`) are ignored by default
- Default revision is "main" if not specified

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Hugging Face for providing the model hub and API
- This project is inspired by the official Hugging Face libraries 