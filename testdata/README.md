# Word2Vec Embeddings for Vector Analogy Tests

This directory contains (or should contain) Word2Vec embeddings used for testing semantic vector analogies.

## Required Files

- `word2vec_sample.bin`: A binary file containing word embeddings in Word2Vec format

## How to Get Word2Vec Embeddings

There are two main options for obtaining Word2Vec embeddings:

### Option 1: Download Pre-trained Embeddings

You can download pre-trained Word2Vec embeddings from various sources:

1. **Google's original Word2Vec embeddings**:
   - [Google News vectors (3 billion running words)](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
   - File size: ~1.5GB
   - 300-dimensional vectors for 3 million words and phrases

2. **Smaller, more manageable datasets**:
   - [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (Similar format, easier to work with)
   - Available in various dimensions (50d, 100d, 200d, 300d)

### Option 2: Create a Smaller Sample

For testing purposes, you might want to create a smaller sample from a larger dataset:

```go
package main

import (
    "bufio"
    "encoding/binary"
    "fmt"
    "os"
    "strings"
)

func main() {
    // Open the original Word2Vec file
    srcFile, err := os.Open("path/to/original/word2vec.bin")
    if err != nil {
        panic(err)
    }
    defer srcFile.Close()

    // Create the sample file
    sampleFile, err := os.Create("testdata/word2vec_sample.bin")
    if err != nil {
        panic(err)
    }
    defer sampleFile.Close()

    // Read header (vocab size and vector dimension)
    reader := bufio.NewReader(srcFile)
    var vocabSize, vectorDim int
    fmt.Fscanf(reader, "%d %d", &vocabSize, &vectorDim)
    
    // Skip newline
    reader.ReadByte()

    // Words we want to include in our sample
    // Include all words from our test cases plus some extras
    wordsToInclude := map[string]bool{
        "king": true, "queen": true, "man": true, "woman": true,
        "father": true, "mother": true, "husband": true, "wife": true,
        "paris": true, "france": true, "berlin": true, "germany": true,
        "rome": true, "italy": true, "tokyo": true, "japan": true,
        "madrid": true, "spain": true, "london": true, "england": true,
        "walking": true, "walk": true, "swimming": true, "swim": true,
        "went": true, "go": true, "came": true, "come": true,
        "bigger": true, "big": true, "smaller": true, "small": true,
        "best": true, "good": true, "worst": true, "bad": true,
        // Add more words if needed
    }

    // Count how many words we'll actually include
    includedCount := 0
    
    // First pass: count words to include
    for i := 0; i < vocabSize; i++ {
        // Read word
        word, err := readWord(reader)
        if err != nil {
            panic(fmt.Sprintf("Error reading word %d: %v", i, err))
        }
        
        // Check if we want this word
        if wordsToInclude[strings.ToLower(word)] {
            includedCount++
        }
        
        // Skip the vector data
        _, err = reader.Discard(4 * vectorDim) // 4 bytes per float32
        if err != nil {
            panic(fmt.Sprintf("Error skipping vector %d: %v", i, err))
        }
    }
    
    // Reset file position
    srcFile.Seek(0, 0)
    reader = bufio.NewReader(srcFile)
    
    // Re-read header and skip newline
    fmt.Fscanf(reader, "%d %d", &vocabSize, &vectorDim)
    reader.ReadByte()
    
    // Write header to sample file
    fmt.Fprintf(sampleFile, "%d %d\n", includedCount, vectorDim)
    
    // Second pass: copy selected words and vectors
    for i := 0; i < vocabSize; i++ {
        // Read word
        word, err := readWord(reader)
        if err != nil {
            panic(fmt.Sprintf("Error reading word %d: %v", i, err))
        }
        
        // Read vector
        vector := make([]float32, vectorDim)
        for j := 0; j < vectorDim; j++ {
            err := binary.Read(reader, binary.LittleEndian, &vector[j])
            if err != nil {
                panic(fmt.Sprintf("Error reading vector %d component %d: %v", i, j, err))
            }
        }
        
        // If this is a word we want, write it to sample file
        if wordsToInclude[strings.ToLower(word)] {
            // Write word
            sampleFile.WriteString(word)
            sampleFile.WriteByte(' ')
            
            // Write vector
            binary.Write(sampleFile, binary.LittleEndian, vector)
        }
    }
}

func readWord(reader *bufio.Reader) (string, error) {
    var word []byte
    char, err := reader.ReadByte()
    
    for err == nil && char != ' ' {
        word = append(word, char)
        char, err = reader.ReadByte()
    }
    
    if err != nil {
        return "", err
    }
    
    return string(word), nil
}
```

## File Format

The binary Word2Vec format has the following structure:

1. Header line: `<vocab_size> <vector_dimension>\n`
2. For each word:
   - Word as a string followed by a space
   - Vector as binary float32 values (4 bytes each)

## Usage

Place the Word2Vec binary file in this directory, and make sure the filename matches what's expected in the test (`word2vec_sample.bin`).

When running the vector analogy tests, the system will automatically load the embeddings from this file. 