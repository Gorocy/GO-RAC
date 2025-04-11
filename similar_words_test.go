package main

import (
	"bufio"
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
	simWordsMaxVectors    = flag.Int("sim_max_vectors", 100000, "Maximum number of vectors to load from embeddings file")
	simWordsTopK          = flag.Int("sim_top_k", 20, "Number of top results to consider in similarity search")
	simWordsInputFile     = flag.String("sim_input", "", "Path to file with words to test (one word per line)")
	simWordsOutputFile    = flag.String("sim_output", "similar_words_results.csv", "Path to output CSV file with results")
	simWordsEmbeddingFile = flag.String("sim_embedding_file", "testdata/word2vec_sample.bin", "Path to word embeddings file")
)

// TestSimilarWords finds semantically similar words for a set of query words
func TestSimilarWords(t *testing.T) {
	// Parse flags for test configuration
	flag.Parse()

	// Create a new HNSW graph with appropriate parameters for word embeddings
	// Cosine distance is typically best for word embeddings
	graph := db.NewHNSWGraph(16, 200, config.DistanceTypeCosine)

	// Get words to include (we don't need specific priority words for similarity search)
	priorityWords := make(map[string]bool)

	// Load Word2Vec embeddings
	embeddingsPath := *simWordsEmbeddingFile
	words, err := loadWord2VecEmbeddings(embeddingsPath, graph, *simWordsMaxVectors, priorityWords)
	if err != nil {
		// If file doesn't exist, skip the test rather than fail
		if os.IsNotExist(err) {
			t.Skip("Skipping test: Word2Vec embeddings file not found:", embeddingsPath)
		}
		t.Fatalf("Failed to load Word2Vec embeddings: %v", err)
	}

	t.Logf("Loaded %d word vectors", len(words))

	// Define words to test - either from file or use defaults
	var testWords []string
	if *simWordsInputFile != "" {
		// Read words from file
		testWords, err = readWordsFromFile(*simWordsInputFile)
		if err != nil {
			t.Logf("Warning: Could not read words from file: %v", err)
			// Fall back to default words
			testWords = getDefaultTestWords()
		}
	} else {
		// Use default test words
		testWords = getDefaultTestWords()
	}

	t.Logf("Testing similarity for %d words", len(testWords))

	// Create output file for CSV results
	outputFile, err := createOutputFile(*simWordsOutputFile)
	if err != nil {
		t.Logf("Warning: Could not create output file: %v", err)
		t.Logf("Results will only be displayed in console")
	} else {
		defer outputFile.Close()
		// Write CSV header
		outputFile.WriteString("Query,Rank,Similar,Similarity\n")
	}

	// Test each word
	for _, word := range testWords {
		// Skip words not in vocabulary
		if !contains(words, word) {
			t.Logf("Skipping word '%s': Not in vocabulary", word)
			continue
		}

		// Get vector for the word
		vector, ok := graph.Vectors[word]
		if !ok {
			t.Errorf("Word '%s' not found in graph, but was in vocabulary", word)
			continue
		}

		// Search for similar words
		results, err := searchSimilarWords(graph, vector.Data, word, *simWordsTopK)
		if err != nil {
			t.Errorf("Error searching for word '%s': %v", word, err)
			continue
		}

		// Display results in console
		t.Logf("\nSimilar words for '%s':", word)
		for i, result := range results {
			t.Logf("  %2d. %s (%.4f)", i+1, result.Word, result.Similarity)

			// Write to CSV file if available
			if outputFile != nil {
				outputFile.WriteString(fmt.Sprintf("%s,%d,%s,%.6f\n",
					word, i+1, result.Word, result.Similarity))
			}
		}
	}

	if outputFile != nil {
		t.Logf("\nResults saved to '%s'", *simWordsOutputFile)
	}
}

// SimilarWord represents a word and its similarity to the query
type SimilarWord struct {
	Word       string
	Similarity float32
}

// searchSimilarWords finds words similar to the query word
func searchSimilarWords(graph *db.HNSWGraph, queryVector []float32, queryWord string, topK int) ([]SimilarWord, error) {
	// Search for nearest neighbors
	neighbors, err := graph.Search(queryVector, topK+1) // +1 to account for the query word itself
	if err != nil {
		return nil, err
	}

	// Extract results, filtering out the query word
	var results []SimilarWord
	for _, neighbor := range neighbors {
		if neighbor.ID != queryWord {
			// Calculate similarity (1 - distance for cosine, since we want similarity not distance)
			similarity := 1.0 - float32(graph.Distance(queryVector, neighbor.Data))
			results = append(results, SimilarWord{
				Word:       neighbor.ID,
				Similarity: similarity,
			})

			if len(results) >= topK {
				break
			}
		}
	}

	return results, nil
}

// getDefaultTestWords returns a default set of words to test for similarity
func getDefaultTestWords() []string {
	return []string{
		// Common nouns
		"cat", "dog", "house", "car", "book", "computer", "phone", "tree", "water", "food",

		// Abstract concepts
		"love", "happiness", "sadness", "time", "money", "freedom", "justice", "peace",

		// Actions
		"run", "eat", "sleep", "think", "work", "play", "build", "create",

		// Countries
		"france", "usa", "japan", "germany", "russia", "china",

		// Cities
		"paris", "london", "tokyo", "berlin", "moscow", "beijing",

		// Professional terms
		"business", "science", "technology", "engineering", "medicine", "law",

		// Technical terms
		"algorithm", "database", "network", "cloud", "software", "hardware",
	}
}

// readWordsFromFile reads a list of words from a file (one word per line)
func readWordsFromFile(filePath string) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var words []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		word := strings.TrimSpace(scanner.Text())
		if word != "" && !strings.HasPrefix(word, "#") {
			words = append(words, word)
		}
	}

	if err := scanner.Err(); err != nil {
		return words, err
	}

	return words, nil
}

// createOutputFile creates a file for writing CSV results
func createOutputFile(filePath string) (*os.File, error) {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, err
		}
	}

	// Create or truncate the file
	return os.Create(filePath)
}
