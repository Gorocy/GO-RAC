package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"vector-db/config"
	"vector-db/db"
)

// Flags for test configuration
var (
	maxVectorsToLoad    = flag.Int("max_vectors", 100000, "Maximum number of vectors to load from embeddings file")
	topK                = flag.Int("top_k", 10, "Number of top results to consider in analogy tests")
	additionalWordsFile = flag.String("add_words", "", "Path to file with additional words to include in test")
	embeddingFile       = flag.String("embedding_file", "testdata/word2vec_sample.bin", "Path to word embeddings file")
)

// AnalogyTest represents a single analogy test case
type AnalogyTest struct {
	A        string // First term in analogy (e.g., "king")
	B        string // Second term in analogy (e.g., "man")
	C        string // Third term in analogy (e.g., "woman")
	Expected string // Expected result (e.g., "queen")
	Category string // Category of the analogy test (optional)
}

// TestVectorAnalogies tests vector analogies using Word2Vec embeddings
func TestVectorAnalogies(t *testing.T) {
	// Parse flags for test configuration
	flag.Parse()

	// Create a new HNSW graph with appropriate parameters for word embeddings
	// Cosine distance is typically best for word embeddings
	graph := db.NewHNSWGraph(16, 200, config.DistanceTypeCosine)

	// Get words to include (either from default or from file)
	wordsToInclude := getWordsToInclude(*additionalWordsFile)
	t.Logf("Words to include in test: %d", len(wordsToInclude))

	// Load Word2Vec embeddings
	embeddingsPath := *embeddingFile // Use the specified embedding file
	words, err := loadWord2VecEmbeddings(embeddingsPath, graph, *maxVectorsToLoad, wordsToInclude)
	if err != nil {
		// If file doesn't exist, skip the test rather than fail
		if os.IsNotExist(err) {
			t.Skip("Skipping test: Word2Vec embeddings file not found:", embeddingsPath)
		}
		t.Fatalf("Failed to load Word2Vec embeddings: %v", err)
	}

	t.Logf("Loaded %d word vectors", len(words))

	// Define analogy tests
	analogyTests := []AnalogyTest{
		// Gender analogies
		{A: "king", B: "man", C: "woman", Expected: "queen", Category: "gender"},
		{A: "father", B: "man", C: "woman", Expected: "mother", Category: "gender"},
		{A: "husband", B: "man", C: "woman", Expected: "wife", Category: "gender"},
		{A: "brother", B: "man", C: "woman", Expected: "sister", Category: "gender"},
		{A: "son", B: "man", C: "woman", Expected: "daughter", Category: "gender"},
		{A: "uncle", B: "man", C: "woman", Expected: "aunt", Category: "gender"},
		{A: "prince", B: "man", C: "woman", Expected: "princess", Category: "gender"},
		{A: "boy", B: "man", C: "woman", Expected: "girl", Category: "gender"},
		{A: "he", B: "man", C: "woman", Expected: "she", Category: "gender"},

		// Country-capital analogies
		{A: "paris", B: "france", C: "germany", Expected: "berlin", Category: "country-capital"},
		{A: "rome", B: "italy", C: "japan", Expected: "tokyo", Category: "country-capital"},
		{A: "madrid", B: "spain", C: "england", Expected: "london", Category: "country-capital"},
		{A: "moscow", B: "russia", C: "usa", Expected: "washington", Category: "country-capital"},
		{A: "beijing", B: "china", C: "canada", Expected: "ottawa", Category: "country-capital"},
		{A: "warsaw", B: "poland", C: "italy", Expected: "rome", Category: "country-capital"},

		// Past tense analogies
		{A: "walking", B: "walk", C: "swim", Expected: "swimming", Category: "verb-form"},
		{A: "went", B: "go", C: "come", Expected: "came", Category: "verb-form"},
		{A: "running", B: "run", C: "walk", Expected: "walking", Category: "verb-form"},
		{A: "jumped", B: "jump", C: "run", Expected: "running", Category: "verb-form"},
		{A: "spoken", B: "speak", C: "write", Expected: "written", Category: "verb-form"},
		{A: "eaten", B: "eat", C: "live", Expected: "lived", Category: "verb-form"},

		// Comparative/superlative analogies
		{A: "bigger", B: "big", C: "small", Expected: "smaller", Category: "comparative"},
		{A: "best", B: "good", C: "bad", Expected: "worst", Category: "superlative"},
		{A: "taller", B: "tall", C: "short", Expected: "shorter", Category: "comparative"},
		{A: "faster", B: "fast", C: "slow", Expected: "slower", Category: "comparative"},
		{A: "higher", B: "high", C: "low", Expected: "lower", Category: "comparative"},
		{A: "stronger", B: "strong", C: "weak", Expected: "weaker", Category: "comparative"},
		{A: "strongest", B: "strong", C: "weak", Expected: "weakest", Category: "superlative"},
	}

	// Add custom analogies from file if provided
	if customTests, err := loadCustomAnalogies(*additionalWordsFile); err == nil && len(customTests) > 0 {
		t.Logf("Loaded %d custom analogies from file", len(customTests))
		analogyTests = append(analogyTests, customTests...)
	}

	// Store results by category
	resultsByCategory := make(map[string]struct {
		total   int
		correct int
	})

	// Run tests
	for _, test := range analogyTests {
		// Skip the test if any of the required words is not in our vocabulary
		if !contains(words, test.A) || !contains(words, test.B) ||
			!contains(words, test.C) || !contains(words, test.Expected) {
			t.Logf("Skipping test '%s - %s + %s = %s': Words not in vocabulary",
				test.A, test.B, test.C, test.Expected)
			continue
		}

		// Perform analogy test
		result, err := testAnalogy(graph, test.A, test.B, test.C, test.Expected, *topK)
		if err != nil {
			t.Errorf("Error testing analogy %s - %s + %s = %s: %v",
				test.A, test.B, test.C, test.Expected, err)
			continue
		}

		// Update results
		cat := resultsByCategory[test.Category]
		cat.total++
		if result.found {
			cat.correct++
			t.Logf("PASSED: %s - %s + %s = %s (Rank: %d, Similarity: %.4f)",
				test.A, test.B, test.C, test.Expected, result.rank, result.similarity)
		} else {
			t.Logf("FAILED: %s - %s + %s = %s (Not found in top %d, Found: %s)",
				test.A, test.B, test.C, test.Expected, result.topK, strings.Join(result.topResults, ", "))
		}
		resultsByCategory[test.Category] = cat
	}

	// Report results by category
	t.Log("\nResults by category:")
	overallTotal := 0
	overallCorrect := 0

	for category, results := range resultsByCategory {
		accuracy := float64(results.correct) / float64(results.total)
		t.Logf("Category %s: %d/%d (%.2f%%)",
			category, results.correct, results.total, accuracy*100)
		overallTotal += results.total
		overallCorrect += results.correct
	}

	// Overall accuracy
	if overallTotal > 0 {
		overallAccuracy := float64(overallCorrect) / float64(overallTotal)
		t.Logf("\nOverall accuracy: %d/%d (%.2f%%)",
			overallCorrect, overallTotal, overallAccuracy*100)
	}
}

// analogyResult holds the result of a single analogy test
type analogyResult struct {
	found      bool     // Whether the expected word was found in top K results
	rank       int      // Rank of the expected word (1-based)
	similarity float32  // Similarity score to the expected vector
	topK       int      // Number of top results considered
	topResults []string // Top K result words
}

// getWordsToInclude returns set of words to prioritize in embedding loading
func getWordsToInclude(additionalWordsFile string) map[string]bool {
	// Start with default words from our test cases
	words := map[string]bool{
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
	}

	// If additional words file is provided, add those words too
	if additionalWordsFile != "" {
		file, err := os.Open(additionalWordsFile)
		if err == nil {
			defer file.Close()
			scanner := bufio.NewScanner(file)
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if line != "" && !strings.HasPrefix(line, "#") {
					// Handle lines with analogy test format: A,B,C,D,Category
					parts := strings.Split(line, ",")
					if len(parts) >= 4 {
						// This is an analogy test line with A,B,C,D,[Category]
						words[strings.ToLower(parts[0])] = true
						words[strings.ToLower(parts[1])] = true
						words[strings.ToLower(parts[2])] = true
						words[strings.ToLower(parts[3])] = true
					} else {
						// Treat as a single word
						words[strings.ToLower(line)] = true
					}
				}
			}
		}
	}

	return words
}

// loadCustomAnalogies loads custom analogy tests from a file
func loadCustomAnalogies(filePath string) ([]AnalogyTest, error) {
	if filePath == "" {
		return nil, fmt.Errorf("no file path provided")
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var tests []AnalogyTest
	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse analogy test (A,B,C,D,Category)
		parts := strings.Split(line, ",")
		if len(parts) < 4 {
			fmt.Printf("Warning: Skipping line %d, invalid format (expected at least 4 comma-separated values)\n", lineNum)
			continue
		}

		category := "custom"
		if len(parts) >= 5 {
			category = strings.TrimSpace(parts[4])
		}

		test := AnalogyTest{
			A:        strings.TrimSpace(parts[0]),
			B:        strings.TrimSpace(parts[1]),
			C:        strings.TrimSpace(parts[2]),
			Expected: strings.TrimSpace(parts[3]),
			Category: category,
		}
		tests = append(tests, test)
	}

	if err := scanner.Err(); err != nil {
		return tests, err
	}

	return tests, nil
}

// testAnalogy tests a single word analogy (A - B + C = D)
func testAnalogy(graph *db.HNSWGraph, wordA, wordB, wordC, expected string, topK int) (analogyResult, error) {
	// Get vectors for words
	vecA, ok := graph.Vectors[wordA]
	if !ok {
		return analogyResult{}, fmt.Errorf("word '%s' not found in vocabulary", wordA)
	}

	vecB, ok := graph.Vectors[wordB]
	if !ok {
		return analogyResult{}, fmt.Errorf("word '%s' not found in vocabulary", wordB)
	}

	vecC, ok := graph.Vectors[wordC]
	if !ok {
		return analogyResult{}, fmt.Errorf("word '%s' not found in vocabulary", wordC)
	}

	// Calculate result vector: A - B + C
	resultVec, err := db.VectorSubtract(vecA.Data, vecB.Data)
	if err != nil {
		return analogyResult{}, err
	}

	resultVec, err = db.VectorAdd(resultVec, vecC.Data)
	if err != nil {
		return analogyResult{}, err
	}

	// Normalize the result vector (important for cosine similarity)
	db.NormalizeVector(resultVec)

	// Search for nearest neighbors
	neighbors, err := graph.Search(resultVec, topK+3) // Add extra to account for A, B, C
	if err != nil {
		return analogyResult{}, err
	}

	// Extract results, filtering out the input words (A, B, C)
	var filteredNeighbors []db.Vector
	for _, neighbor := range neighbors {
		if neighbor.ID != wordA && neighbor.ID != wordB && neighbor.ID != wordC {
			filteredNeighbors = append(filteredNeighbors, neighbor)
			if len(filteredNeighbors) >= topK {
				break
			}
		}
	}

	// Check if expected word is in results
	result := analogyResult{
		found:      false,
		topK:       topK,
		topResults: make([]string, 0, len(filteredNeighbors)),
	}

	for i, neighbor := range filteredNeighbors {
		result.topResults = append(result.topResults, neighbor.ID)

		if neighbor.ID == expected {
			result.found = true
			result.rank = i + 1

			// Calculate similarity with expected vector
			expectedVec, ok := graph.Vectors[expected]
			if ok {
				similarity, _ := db.CosineSimilarity(resultVec, expectedVec.Data)
				result.similarity = similarity
			}
			break
		}
	}

	return result, nil
}

// Helper function to check if a word is in a slice
func contains(words []string, word string) bool {
	for _, w := range words {
		if strings.EqualFold(w, word) {
			return true
		}
	}
	return false
}

// loadWord2VecEmbeddings loads Word2Vec embeddings from a binary file
func loadWord2VecEmbeddings(filePath string, graph *db.HNSWGraph, maxVectors int, priorityWords map[string]bool) ([]string, error) {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	// Open the file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read the file header
	reader := bufio.NewReader(file)

	// Read word count and vector dimension
	var wordCount, vectorDim int
	fmt.Fscanf(reader, "%d %d", &wordCount, &vectorDim)

	// Consume the newline
	_, err = reader.ReadByte()
	if err != nil {
		return nil, fmt.Errorf("error reading newline: %v", err)
	}

	// Limit the number of vectors to load if needed
	if maxVectors > 0 && maxVectors < wordCount {
		fmt.Printf("Limiting to %d vectors out of %d in the file\n", maxVectors, wordCount)
		wordCount = maxVectors
	}

	// Simplified approach: just read all words in a single pass
	words := make([]string, 0, wordCount)
	priorityWordsFound := make(map[string]bool)

	fmt.Println("First pass: Finding priority words...")
	// First load priority words
	for i := 0; i < wordCount; i++ {
		// Read the word
		word, err := readWord(reader)
		if err != nil {
			return words, fmt.Errorf("error reading word %d: %v", i+1, err)
		}

		// Check if this is a priority word
		wordLower := strings.ToLower(word)
		isPriority := priorityWords[wordLower]

		// Read the vector
		vector := make([]float32, vectorDim)
		for j := 0; j < vectorDim; j++ {
			var val float32
			if err := binary.Read(reader, binary.LittleEndian, &val); err != nil {
				return words, fmt.Errorf("error reading vector %d component %d: %v", i+1, j, err)
			}
			vector[j] = val
		}

		// If we've reached our limit, stop
		if len(words) >= maxVectors {
			break
		}

		// Normalize the vector (important for cosine similarity)
		db.NormalizeVector(vector)

		// Insert into graph and words list
		words = append(words, word)

		if isPriority {
			priorityWordsFound[wordLower] = true
		}

		err = graph.Insert(db.Vector{
			ID:   word,
			Data: vector,
		})

		if err != nil {
			return words, fmt.Errorf("error inserting vector for word '%s': %v", word, err)
		}

		// Show progress
		if i > 0 && i%1000 == 0 {
			fmt.Printf("Loaded %d words...\n", i)
		}
	}

	fmt.Printf("Finished loading. Total words loaded: %d (including %d priority words)\n",
		len(words), len(priorityWordsFound))

	return words, nil
}

// readWord reads a word from the binary file
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

// Helper function to download embeddings if not available
func downloadEmbeddings(t *testing.T, targetPath string) error {
	// Check if file already exists
	if _, err := os.Stat(targetPath); err == nil {
		return nil
	}

	t.Log("Downloading Word2Vec embeddings (this might take a while)...")

	// Create directory structure
	dir := filepath.Dir(targetPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Here you would add code to download the embeddings
	// For example using http.Get or exec.Command to use wget/curl
	// This is a placeholder - you would replace this with actual download code

	return fmt.Errorf("automatic download not implemented - please download Word2Vec embeddings manually")
}
