#!/bin/bash

# Exit on any error
set -e

echo "Starting E2E test for HF Downloader..."

# Create a test directory
TEST_DIR="./test_downloads"
echo "Creating test directory: $TEST_DIR"
mkdir -p $TEST_DIR

# Clean up function
cleanup() {
    echo "Cleaning up test directory..."
    rm -rf $TEST_DIR
}

# Set up trap to clean up on script exit
trap cleanup EXIT

# Run main.go to download a small public model
echo "Downloading test model: hf-internal-testing/tiny-random-gpt2..."
go run main.go hf-internal-testing/tiny-random-gpt2 main $TEST_DIR

# Verify the download was successful
echo "Verifying download..."

# Check if download directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "❌ Error: Download directory not created"
    exit 1
fi

# Check if the basic directory structure exists
REQUIRED_DIRS=("blobs" "refs" "snapshots")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$TEST_DIR/$dir" ]; then
        echo "❌ Error: Expected directory $dir not found"
        exit 1
    else
        echo "✅ Directory $dir exists"
    fi
done

# Check if any files were downloaded in blobs directory
if [ -z "$(ls -A $TEST_DIR/blobs)" ]; then
    echo "❌ Error: No files downloaded in blobs directory"
    exit 1
else
    echo "✅ Files downloaded successfully"
fi

# Print download size
echo -n "📦 Download size: "
du -sh $TEST_DIR

echo "✅ E2E test completed successfully!" 