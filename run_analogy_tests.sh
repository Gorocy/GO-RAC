#!/bin/bash

# Script to run vector analogy tests - ONLY runs tests, does not create samples

# Default values for test configuration
TOP_K=10
ADDITIONAL_WORDS=""
TIMEOUT="10m"
MAX_VECTORS=100000
SAMPLE_FILENAME="word2vec_sample.bin"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--embedding-file)
      SAMPLE_FILENAME="$2"
      shift 2
      ;;
    -m|--max-vectors)
      MAX_VECTORS="$2"
      shift 2
      ;;
    -k|--top-k)
      TOP_K="$2"
      shift 2
      ;;
    -a|--additional-words)
      ADDITIONAL_WORDS="$2"
      shift 2
      ;;
    -t|--timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -o, --embedding-file <file>     Embedding file to use (default: word2vec_sample.bin)"
      echo "  -m, --max-vectors <number>      Maximum vectors to load (default: 100000)"
      echo "  -k, --top-k <number>            Number of top results to consider (default: 10)"
      echo "  -a, --additional-words <file>   File with additional words/analogies to test"
      echo "  -t, --timeout <duration>        Test timeout (default: 10m)"
      echo "  -h, --help                      Show this help message"
      echo ""
      echo "Note: To create a sample from source embeddings, use:"
      echo "  cd testdata"
      echo "  ./build_and_run.sh /path/to/original.bin [output.bin]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Try '$0 --help' for more information"
      exit 1
      ;;
  esac
done

# Check if embeddings exist
if [ ! -f "testdata/$SAMPLE_FILENAME" ]; then
    echo "Error: Word2Vec embeddings not found at testdata/$SAMPLE_FILENAME"
    echo ""
    echo "To create a sample from existing Word2Vec embeddings:"
    echo "  cd testdata"
    echo "  ./build_and_run.sh /path/to/original/word2vec.bin"
    echo ""
    echo "Or download pre-trained embeddings from:"
    echo "  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing"
    echo "  https://nlp.stanford.edu/projects/glove/"
    exit 1
fi

# Display configuration
echo "Running vector analogy tests with the following configuration:"
echo "  - Embeddings file: testdata/$SAMPLE_FILENAME"
echo "  - Max vectors: $MAX_VECTORS"
echo "  - Top K results: $TOP_K"
if [ -n "$ADDITIONAL_WORDS" ]; then
    echo "  - Additional words file: $ADDITIONAL_WORDS"
    ADDITIONAL_WORDS_PARAM="-additional_words=$ADDITIONAL_WORDS"
else
    echo "  - No additional words file specified"
    ADDITIONAL_WORDS_PARAM=""
fi
echo "  - Timeout: $TIMEOUT"
echo ""

# Run the tests with verbose output
echo "Running vector analogy tests..."
go test -v -run TestVectorAnalogies -timeout "$TIMEOUT" \
  -max_vectors="$MAX_VECTORS" -top_k="$TOP_K" $ADDITIONAL_WORDS_PARAM \
  -embedding_file="testdata/$SAMPLE_FILENAME"

# Examples:
# ./run_analogy_tests.sh                                        # Run with defaults
# ./run_analogy_tests.sh -k 5 -a testdata/custom_analogies.txt  # Custom settings 