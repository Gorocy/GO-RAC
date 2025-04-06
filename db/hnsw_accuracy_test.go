package db

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"
	"time"
	"vector-db/config"
)

// TestHNSWAccuracy creates a CSV file with distances from all vectors to a query vector
// and compares HNSW approximate search results with actual nearest neighbors using a scoring metric.
func TestHNSWAccuracy(t *testing.T) {
	// --- Test Configuration ---

	// Use fixed seed for reproducibility during debugging/testing
	// Change to time.Now().UnixNano() for full randomness in each run
	rand.New(rand.NewSource(time.Now().UnixNano()))
	// rand.New(rand.NewSource(42)) // Fixed seed for reproducibility

	// HNSW Parameters
	m := 16               // Max connections per layer
	efConstruction := 200 // Width of candidate list during construction
	efSearch := 400       // Width of candidate list during search (>= k)

	// Search and Data Parameters
	k := 100            // Number of nearest neighbors to find
	dimensions := 100   // Vector dimensions
	numVectors := 10000 // Number of vectors in the dataset (reduced for faster testing)

	// Initialize HNSW graph
	// Make sure the HNSWGraph structure has an EfSearch field
	// If NewHNSWGraph sets EfSearch, that's good. If not, set it manually.
	graph := NewHNSWGraph(m, efConstruction, config.DistanceTypeEuclidean)
	// Assuming HNSWGraph has an EfSearch field or a method to set it:
	// graph.SetEfSearch(efSearch) // Or directly, if the field is exported
	graph.EfSearch = efSearch // Assuming the field is exported

	// Check if efSearch is at least k
	if efSearch < k {
		t.Logf("Warning: efSearch (%d) is less than k (%d). Consider increasing efSearch for better accuracy.", efSearch, k)
	}

	// --- Data Generation ---
	vectors := make([]Vector, numVectors)
	fmt.Printf("Generating %d vectors with %d dimensions...\n", numVectors, dimensions)
	for i := 0; i < numVectors; i++ {
		vectorData := make([]float32, dimensions) // Changed variable name to avoid shadowing
		for j := 0; j < dimensions; j++ {
			vectorData[j] = rand.Float32() * 100 // Random values in range [0, 100)
		}
		vectors[i] = Vector{
			ID:       fmt.Sprintf("vec_%d", i), // Added prefix for clarity of IDs
			Data:     vectorData,
			Metadata: map[string]interface{}{"index": i},
		}
	}
	fmt.Println("Vector generation complete.")

	// --- Inserting Vectors into the Graph ---
	fmt.Println("Inserting vectors into HNSW graph...")
	startTime := time.Now()
	for i, vector := range vectors {
		// Assuming Insert MAY return an error
		err := graph.Insert(vector) // Using Insert that returns an error
		if err != nil {
			// End the test if insertion fails
			t.Fatalf("Insert failed for vector %s: %v", vector.ID, err)
		}
		// Print progress every 10% or every 1000 vectors
		if (i+1)%(numVectors/10) == 0 || (i+1)%1000 == 0 {
			fmt.Printf("Inserted %d / %d vectors...\n", i+1, numVectors)
		}
	}
	insertDuration := time.Since(startTime)
	insertRate := float64(numVectors) / insertDuration.Seconds()
	fmt.Printf("Insertion finished in %s (%.2f vectors/sec)\n", insertDuration, insertRate)

	// --- Preparation and Execution of Search ---
	// Create query vector
	query := make([]float32, dimensions)
	for j := 0; j < dimensions; j++ {
		query[j] = rand.Float32() * 100
	}

	// Perform HNSW search
	fmt.Printf("Performing HNSW search for k=%d with efSearch=%d...\n", k, graph.EfSearch)
	startTime = time.Now()
	// Assuming Search MAY return an error
	results, err := graph.Search(query, k) // Using Search that returns an error
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	searchDuration := time.Since(startTime)
	fmt.Printf("Search finished in %s, found %d results (expected %d)\n", searchDuration, len(results), k)
	if len(results) > k {
		t.Logf("Warning: Search returned %d results, which is more than k=%d. Truncating to k.", len(results), k)
		results = results[:k]
	} else if len(results) < k && len(vectors) >= k {
		// Only log if there were enough vectors in the graph to find k
		t.Logf("Warning: Search returned %d results, which is less than k=%d.", len(results), k)
	}

	// --- Calculating Actual Distances and Rankings (Ground Truth) ---
	fmt.Println("Calculating exact distances and ranks (ground truth)...")
	type DistanceInfo struct {
		VectorID string
		Distance float32
		Rank     int  // Actual ranking (1 = nearest)
		Found    bool // Whether found by HNSW in results
	}

	// Calculate distances for all vectors
	allDistances := make([]DistanceInfo, numVectors) // Pre-allocation
	for i := 0; i < numVectors; i++ {
		distance := graph.Distance(query, vectors[i].Data) // Use Distance method from graph
		allDistances[i] = DistanceInfo{
			VectorID: vectors[i].ID,
			Distance: distance,
		}
	}

	// Sort by actual distance
	sort.Slice(allDistances, func(i, j int) bool {
		return allDistances[i].Distance < allDistances[j].Distance
	})

	// Assign rankings and build ID -> Rank map
	actualRankMap := make(map[string]int, numVectors)
	for i := range allDistances {
		allDistances[i].Rank = i + 1 // Ranking starts from 1
		actualRankMap[allDistances[i].VectorID] = allDistances[i].Rank
	}
	fmt.Println("Exact ranks calculated.")

	// Mark vectors found by HNSW
	foundIDs := make(map[string]bool, len(results))
	for _, result := range results {
		foundIDs[result.ID] = true
	}
	truePositivesCount := 0
	for i := range allDistances {
		if _, ok := foundIDs[allDistances[i].VectorID]; ok {
			allDistances[i].Found = true
			// Count true positives only those in actual top-k for Recall@k
			if allDistances[i].Rank <= k {
				truePositivesCount++
			}
		}
	}

	// --- Writing to CSV File (Optional) ---
	csvDir := "test_hnsw_results"
	if err := os.MkdirAll(csvDir, 0755); err != nil {
		t.Fatalf("Failed to create directory %s: %v", csvDir, err)
	}
	csvPath := filepath.Join(csvDir, fmt.Sprintf("hnsw_accuracy_k%d_ef%d.csv", k, efSearch))
	fmt.Printf("Saving detailed results (top %d actual) to %s...\n", k*2, csvPath)
	file, err := os.Create(csvPath)
	if err != nil {
		t.Fatalf("Failed to create CSV file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"Actual Rank", "Vector ID", "Distance", "Found By HNSW"}
	writer.Write(header) // Ignoring potential header write error for simplicity

	writeLimit := min(k*2, len(allDistances)) // Write 2*k best actual results
	for i := 0; i < writeLimit; i++ {
		d := allDistances[i]
		row := []string{
			strconv.Itoa(d.Rank),
			d.VectorID,
			fmt.Sprintf("%.6f", d.Distance),
			strconv.FormatBool(d.Found),
		}
		writer.Write(row) // Ignoring potential row write error
	}
	fmt.Println("CSV file saved.")

	// --- Accuracy Analysis ---

	// 1. Recall@k (Classic metric)
	recallAtK := 0.0
	if k > 0 {
		// Using pre-calculated truePositivesCount
		recallAtK = float64(truePositivesCount) / float64(k)
	}

	// 2. Scored Accuracy (Rank-weighted scoring metric)
	var totalScore float64 = 0.0
	numResultsReturned := len(results) // Actual number of returned results

	// Debug: Show top 10 returned by HNSW and their actual ranks
	fmt.Println("\nTop 10 results returned by HNSW and their actual ranks:")
	for i, result := range results[:min(10, numResultsReturned)] {
		actualRank, found := actualRankMap[result.ID]
		rankStr := "Not Found in Map"
		distStr := "N/A"
		if found {
			rankStr = fmt.Sprintf("Actual Rank %d", actualRank)
			distStr = fmt.Sprintf("%.6f", graph.Distance(query, result.Data))
		}
		fmt.Printf("%d. Vector %s: %s, Distance %s\n", i+1, result.ID, rankStr, distStr)
	}

	// Calculate sum of scores
	scoredVectorsCount := 0 // Number of vectors that contributed points (>0)
	for _, result := range results {
		actualRank, found := actualRankMap[result.ID]
		if !found {
			// This shouldn't happen if 'results' come from 'vectors'
			t.Logf("Warning: Result vector %s not found in the original dataset's rank map. Skipping score calculation for this vector.", result.ID)
			continue
		}

		// Award points only if the vector is in actual top-k
		if actualRank <= k {
			var individualScore float64
			// Formula: Linear decay from 1.0 for rank 1 to 0.6 for rank k
			// Formula: 0.6 + 0.4 * (k - actualRank) / (k - 1)
			if k == 1 {
				// Handle edge case k=1
				if actualRank == 1 {
					individualScore = 1.0
				} else {
					individualScore = 0.0 // Shouldn't happen if actualRank <= k
				}
			} else {
				// Use max(0, ...) for numeric errors or unexpected ranks > k
				linearFactor := math.Max(0.0, float64(k-actualRank)/float64(k-1))
				individualScore = 0.6 + 0.4*linearFactor
			}

			totalScore += individualScore
			scoredVectorsCount++ // Count only those that contributed points
		}
		// Vectors outside the actual top-k (actualRank > k) don't add points
	}

	// Calculate average score for the *requested* number of neighbors (k)
	averageScore := 0.0
	if k > 0 {
		averageScore = totalScore / float64(k)
	}

	// --- Displaying Results and Test Assertion ---
	fmt.Printf("\n=== HNSW Search Accuracy Analysis ===\n")
	fmt.Printf("Dataset Size:  %d vectors\n", numVectors)
	fmt.Printf("Dimensions:    %d\n", dimensions)
	fmt.Printf("HNSW Params:   M=%d, efConstruction=%d\n", m, efConstruction)
	fmt.Printf("Search Params: k=%d, efSearch=%d\n", k, efSearch)
	fmt.Printf("Results Found: %d\n", numResultsReturned)
	fmt.Printf("--- Metrics ---\n")
	fmt.Printf("Recall@%d:     %.4f (%.2f%%)  (%d/%d actual top-k vectors found)\n",
		k, recallAtK, recallAtK*100, truePositivesCount, k)
	fmt.Printf("Scored Acc:    %.4f          (Rank-weighted score, max=1.0, min=%.2f*Recall)\n",
		averageScore, 0.6) // Min score contribution is 0.6 per true positive
	fmt.Printf("---------------\n")

	// Assertion based on scoring metric
	// Threshold should be adjusted based on expected quality for given parameters.
	// For this function (0.6-1.0), the result will often be closer to Recall.
	scoreThreshold := 0.45 // Example threshold, adjust as needed
	if averageScore < scoreThreshold {
		t.Errorf("HNSW scored accuracy (%.4f) is below the threshold (%.4f)", averageScore, scoreThreshold)
	} else {
		fmt.Printf("Scored accuracy meets the threshold (>= %.4f)\n", scoreThreshold)
	}

	// Assertion checking consistency: Scored Accuracy shouldn't be drastically lower than Recall
	// Minimum possible SA score is Recall@k * 0.6 (when all hits have rank k)
	// Let's set a threshold slightly higher, e.g., 0.7 * Recall@k, to ensure we're not finding just the worst of top-k.
	expectedMinScore := recallAtK * 0.70                  // Expect average score to be at least 70% of recall
	if averageScore < expectedMinScore && recallAtK > 0 { // Added condition recall > 0 to avoid error when recall = 0
		t.Errorf("Scored accuracy (%.4f) is unexpectedly low compared to Recall@k (%.4f). Expected at least ~%.4f.",
			averageScore, recallAtK, expectedMinScore)
	} else if recallAtK > 0 {
		fmt.Printf("Scored accuracy is consistent with Recall@k (>= ~%.4f based on Recall)\n", expectedMinScore)
	}
}
