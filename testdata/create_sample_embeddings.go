package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"
)

// This is a standalone utility to create a smaller sample of Word2Vec embeddings
// for use with the vector analogy tests.

func main() {
	// Parse command line arguments
	sourceFile := flag.String("source", "", "Path to the original Word2Vec binary file")
	outputFile := flag.String("output", "word2vec_sample.bin", "Path to output the sample file")
	maxWords := flag.Int("max_words", 0, "Maximum number of words to extract (0 = extract all found words)")
	randomWords := flag.Int("random_words", 0, "Number of random words to include in addition to the predefined list")
	flag.Parse()

	if *sourceFile == "" {
		fmt.Println("Error: You must specify a source file with -source")
		flag.Usage()
		os.Exit(1)
	}

	// Words we want to include in our sample
	// Include all words from our test cases plus some extras for better context
	wordsToInclude := map[string]bool{
		// Gender analogies
		"king": true, "queen": true, "man": true, "woman": true,
		"father": true, "mother": true, "husband": true, "wife": true,
		"uncle": true, "aunt": true, "prince": true, "princess": true,
		"brother": true, "sister": true, "he": true, "she": true,
		"boy": true, "girl": true, "son": true, "daughter": true,

		// Country-capital analogies
		"paris": true, "france": true, "berlin": true, "germany": true,
		"rome": true, "italy": true, "tokyo": true, "japan": true,
		"madrid": true, "spain": true, "london": true, "england": true,
		"moscow": true, "russia": true, "washington": true, "usa": true,
		"beijing": true, "china": true, "ottawa": true, "canada": true,
		"warsaw": true, "poland": true,

		// Verb forms
		"walking": true, "walk": true, "swimming": true, "swim": true,
		"went": true, "go": true, "came": true, "come": true,
		"running": true, "run": true, "jumped": true, "jump": true,
		"lived": true, "live": true, "spoken": true, "speak": true,
		"written": true, "write": true, "eaten": true, "eat": true,

		// Comparative/superlative
		"bigger": true, "big": true, "smaller": true, "small": true,
		"best": true, "good": true, "worst": true, "bad": true,
		"taller": true, "tall": true, "shorter": true, "short": true,
		"faster": true, "fast": true, "slower": true, "slow": true,
		"higher": true, "high": true, "lower": true, "low": true,
		"strongest": true, "stronger": true, "strong": true,
		"weakest": true, "weaker": true, "weak": true,
		"happier": true, "happy": true, "unhappier": true, "unhappy": true,
	}

	fmt.Printf("Creating sample Word2Vec file from %s\n", *sourceFile)
	fmt.Printf("Target words to extract: %d\n", len(wordsToInclude))

	// Open the source file
	srcFile, err := os.Open(*sourceFile)
	if err != nil {
		fmt.Printf("Error opening source file: %v\n", err)
		os.Exit(1)
	}
	defer srcFile.Close()

	// Create the sample file
	sampleFile, err := os.Create(*outputFile)
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		os.Exit(1)
	}
	defer sampleFile.Close()

	// Read header (vocab size and vector dimension)
	reader := bufio.NewReader(srcFile)
	var vocabSize, vectorDim int
	_, err = fmt.Fscanf(reader, "%d %d", &vocabSize, &vectorDim)
	if err != nil {
		fmt.Printf("Error reading header: %v\n", err)
		os.Exit(1)
	}

	// Skip newline
	_, err = reader.ReadByte()
	if err != nil {
		fmt.Printf("Error skipping newline: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Source file has %d words with %d dimensions\n", vocabSize, vectorDim)

	// Count how many words we'll actually include
	includedCount := 0
	foundWords := make(map[string]bool)
	var allWords []string // Store all words for random selection

	// First pass: count words to include and verify they exist
	fmt.Println("Scanning vocabulary...")
	for i := 0; i < vocabSize; i++ {
		// Read word
		word, err := readWord(reader)
		if err != nil {
			fmt.Printf("Error reading word %d: %v\n", i, err)
			os.Exit(1)
		}

		// Store word for potential random selection
		if *randomWords > 0 {
			allWords = append(allWords, word)
		}

		// Check if we want this word
		wordLower := strings.ToLower(word)
		if wordsToInclude[wordLower] {
			includedCount++
			foundWords[wordLower] = true
		}

		// Skip the vector data
		_, err = reader.Discard(4 * vectorDim) // 4 bytes per float32
		if err != nil {
			fmt.Printf("Error skipping vector %d: %v\n", i, err)
			os.Exit(1)
		}

		// Show progress
		if i%100000 == 0 {
			fmt.Printf("Processed %d words...\n", i)
		}
	}

	// Select random words if requested
	randomWordsMap := make(map[string]bool)
	if *randomWords > 0 && len(allWords) > 0 {
		fmt.Printf("Selecting %d random words...\n", *randomWords)

		// Initialize random number generator
		rand.NewSource(time.Now().UnixNano())

		// Select random words
		for i := 0; i < *randomWords && len(allWords) > 0; i++ {
			// Pick a random index
			idx := rand.Intn(len(allWords))
			word := allWords[idx]

			// Skip if already in our predefined list
			wordLower := strings.ToLower(word)
			if wordsToInclude[wordLower] || randomWordsMap[word] {
				// Replace with another word and try again
				i--
				// Remove this word to avoid selecting it again
				allWords[idx] = allWords[len(allWords)-1]
				allWords = allWords[:len(allWords)-1]
				continue
			}

			// Add to our random words map
			randomWordsMap[word] = true

			// Remove this word to avoid selecting it again
			allWords[idx] = allWords[len(allWords)-1]
			allWords = allWords[:len(allWords)-1]
		}

		// Update included count
		includedCount += len(randomWordsMap)
		fmt.Printf("Added %d random words\n", len(randomWordsMap))
	}

	fmt.Printf("Found %d out of %d target words\n", includedCount, len(wordsToInclude))

	// Check which words were not found
	for word := range wordsToInclude {
		if !foundWords[word] {
			fmt.Printf("Warning: Word '%s' not found in vocabulary\n", word)
		}
	}

	// Reset file position
	_, err = srcFile.Seek(0, 0)
	if err != nil {
		fmt.Printf("Error resetting file position: %v\n", err)
		os.Exit(1)
	}
	reader = bufio.NewReader(srcFile)

	// Re-read header and skip newline
	_, err = fmt.Fscanf(reader, "%d %d", &vocabSize, &vectorDim)
	if err != nil {
		fmt.Printf("Error re-reading header: %v\n", err)
		os.Exit(1)
	}
	_, err = reader.ReadByte()
	if err != nil {
		fmt.Printf("Error skipping newline: %v\n", err)
		os.Exit(1)
	}

	// Write header to sample file
	_, err = fmt.Fprintf(sampleFile, "%d %d\n", includedCount, vectorDim)
	if err != nil {
		fmt.Printf("Error writing header: %v\n", err)
		os.Exit(1)
	}

	// Second pass: copy selected words and vectors
	fmt.Println("Extracting vectors...")
	wordsExtracted := 0
	for i := 0; i < vocabSize; i++ {
		// Read word
		word, err := readWord(reader)
		if err != nil {
			fmt.Printf("Error reading word %d: %v\n", i, err)
			os.Exit(1)
		}

		// Read vector
		vector := make([]float32, vectorDim)
		for j := 0; j < vectorDim; j++ {
			err := binary.Read(reader, binary.LittleEndian, &vector[j])
			if err != nil {
				fmt.Printf("Error reading vector %d component %d: %v\n", i, j, err)
				os.Exit(1)
			}
		}

		// If this is a word we want, write it to sample file
		wordLower := strings.ToLower(word)
		if wordsToInclude[wordLower] || randomWordsMap[word] {
			// Check if we've reached the max words limit
			if *maxWords > 0 && wordsExtracted >= *maxWords {
				break
			}

			// Write word
			_, err = sampleFile.WriteString(word)
			if err != nil {
				fmt.Printf("Error writing word: %v\n", err)
				os.Exit(1)
			}
			_, err = sampleFile.Write([]byte{' '})
			if err != nil {
				fmt.Printf("Error writing space: %v\n", err)
				os.Exit(1)
			}

			// Write vector
			err = binary.Write(sampleFile, binary.LittleEndian, vector)
			if err != nil {
				fmt.Printf("Error writing vector: %v\n", err)
				os.Exit(1)
			}

			wordsExtracted++
			if wordsExtracted%10 == 0 {
				fmt.Printf("Extracted %d words...\n", wordsExtracted)
			}
		}

		// Show progress
		if i%100000 == 0 {
			fmt.Printf("Processed %d words...\n", i)
		}
	}

	fmt.Printf("Successfully extracted %d words to %s\n", wordsExtracted, *outputFile)
}

// readWord reads a space-terminated word from the buffer
func readWord(reader *bufio.Reader) (string, error) {
	var word []byte
	char, err := reader.ReadByte()

	for err == nil && char != ' ' {
		word = append(word, char)
		char, err = reader.ReadByte()
	}

	if err != nil && len(word) == 0 {
		return "", err
	}

	return string(word), nil
}
