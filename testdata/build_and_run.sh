#!/bin/bash

# Script to compile and run the Word2Vec sample creator
# This script ONLY builds and runs the sample creator, nothing else

# Compile the Go program
echo "=== Compiling create_sample_embeddings.go ==="
go build -o create_sample_embeddings create_sample_embeddings.go

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✅ Compilation successful!"

# Default values
MAX_WORDS=0
RANDOM_WORDS=0

# Function to display usage
function show_usage {
    echo "Usage: $0 <path_to_word2vec_binary_file> [output_file] [options]"
    echo "Options:"
    echo "  -m, --max-words N     Maximum number of words to extract (default: all)"
    echo "  -r, --random-words N  Number of random words to include (default: 0)"
    echo "  --add_words N         Alias for --random-words"
    echo ""
    echo "Example: $0 GoogleNews-vectors-negative300.bin word2vec_sample.bin -r 100"
    echo "         $0 GoogleNews-vectors-negative300.bin -m 50 -r 50"
    echo "         $0 GoogleNews-vectors-negative300.bin --add_words 800"
}

# Check if source file was provided as argument
if [ -z "$1" ]; then
    show_usage
    exit 1
fi

SOURCE_FILE="$1"
shift

# Check if the next argument is the output file (not starting with -)
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    OUTPUT_FILE="$1"
    shift
else
    OUTPUT_FILE="word2vec_sample.bin"
fi

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--max-words)
            MAX_WORDS="$2"
            shift 2
            ;;
        -r|--random-words|--add_words)
            RANDOM_WORDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Verify source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ Error: Source file not found at $SOURCE_FILE"
    exit 1
fi

# Check if we need to create a relative or absolute path
if [[ "$SOURCE_FILE" != /* ]]; then
    # Relative path - make it relative to the current directory
    SOURCE_FILE="$(pwd)/$SOURCE_FILE"
fi

echo "=== Creating Word2Vec sample ==="
echo "Source: $SOURCE_FILE"
echo "Output: $OUTPUT_FILE"
if [ $MAX_WORDS -gt 0 ]; then
    echo "Max words: $MAX_WORDS"
fi
if [ $RANDOM_WORDS -gt 0 ]; then
    echo "Random words: $RANDOM_WORDS"
fi
echo ""

# Build the command
CMD="./create_sample_embeddings -source \"$SOURCE_FILE\" -output \"$OUTPUT_FILE\""
if [ $MAX_WORDS -gt 0 ]; then
    CMD+=" -max_words $MAX_WORDS"
fi
if [ $RANDOM_WORDS -gt 0 ]; then
    CMD+=" -random_words $RANDOM_WORDS"
fi

# Run the program
echo "Running sample creator..."
eval $CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Sample embeddings created successfully!"
    echo "Output file: $OUTPUT_FILE"
    echo "======================================"
    
    # Print file size
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "File size: $FILE_SIZE"
    
    # Print some stats about the file
    echo "Vector stats:"
    head -n 1 "$OUTPUT_FILE"
    
    echo ""
    echo "You can now run the tests using:"
    echo "cd .."
    echo "./run_analogy_tests.sh"
    
    exit 0
else
    echo "❌ Error creating sample embeddings!"
    exit 1
fi 