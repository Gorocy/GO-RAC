package db

import (
	"fmt"
	"math/rand"
	"testing"

	"vector-db/config"
)

func TestHNSWInsertAndSearch(t *testing.T) {
	// Initialize HNSW graph with smaller parameters for testing
	graph := NewHNSWGraph(8, 100, config.DistanceTypeEuclidean)

	// Generate random vectors with smaller dimensions
	dimensions := 64
	numVectors := 100
	vectors := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vector := make([]float32, dimensions)
		for j := 0; j < dimensions; j++ {
			vector[j] = rand.Float32()
		}
		vectors[i] = Vector{
			ID:   fmt.Sprintf("%d", i),
			Data: vector,
		}
	}

	// Insert vectors
	for _, vector := range vectors {
		err := graph.Insert(vector)
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Test search
	query := make([]float32, dimensions)
	for i := 0; i < dimensions; i++ {
		query[i] = rand.Float32()
	}

	// Search for nearest neighbors with smaller k
	k := 5
	results, err := graph.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Verify results
	if len(results) != k {
		t.Errorf("Expected %d results, got %d", k, len(results))
	}

	// Verify distances are in ascending order
	for i := 1; i < len(results); i++ {
		dist1 := graph.Distance(query, results[i-1].Data)
		dist2 := graph.Distance(query, results[i].Data)
		if dist1 > dist2 {
			t.Errorf("Distances not in ascending order: %f > %f", dist1, dist2)
		}
	}
}

func TestHNSWConcurrentOperations(t *testing.T) {
	graph := NewHNSWGraph(8, 100, config.DistanceTypeEuclidean)
	dimensions := 64
	numVectors := 100
	vectors := make([]Vector, numVectors)

	// Generate random vectors
	for i := 0; i < numVectors; i++ {
		vector := make([]float32, dimensions)
		for j := 0; j < dimensions; j++ {
			vector[j] = rand.Float32()
		}
		vectors[i] = Vector{
			ID:   fmt.Sprintf("%d", i),
			Data: vector,
		}
	}

	// Concurrent insertions
	done := make(chan bool)
	for i := 0; i < 2; i++ {
		go func(start, end int) {
			for j := start; j < end; j++ {
				err := graph.Insert(vectors[j])
				if err != nil {
					t.Errorf("Insert failed: %v", err)
				}
			}
			done <- true
		}(i*50, (i+1)*50)
	}

	// Wait for all insertions to complete
	for i := 0; i < 2; i++ {
		<-done
	}

	// Concurrent searches
	query := make([]float32, dimensions)
	for i := 0; i < dimensions; i++ {
		query[i] = rand.Float32()
	}

	for i := 0; i < 2; i++ {
		go func() {
			results, err := graph.Search(query, 5)
			if err != nil {
				t.Errorf("Search failed: %v", err)
				done <- true
				return
			}
			if len(results) != 5 {
				t.Errorf("Expected 5 results, got %d", len(results))
			}
			done <- true
		}()
	}

	// Wait for all searches to complete
	for i := 0; i < 2; i++ {
		<-done
	}
}

func TestHNSWDifferentDistanceMetrics(t *testing.T) {
	dimensions := 64
	vectors := make([]Vector, 50)
	for i := 0; i < 50; i++ {
		vector := make([]float32, dimensions)
		for j := 0; j < dimensions; j++ {
			vector[j] = rand.Float32()
		}
		vectors[i] = Vector{
			ID:   fmt.Sprintf("%d", i),
			Data: vector,
		}
	}

	// Test different distance metrics
	metrics := []config.DistanceType{
		config.DistanceTypeEuclidean,
		config.DistanceTypeCosine,
		config.DistanceTypeManhattan,
		config.DistanceTypeHamming,
	}

	for _, metric := range metrics {
		graph := NewHNSWGraph(8, 100, metric)
		for _, vector := range vectors {
			err := graph.Insert(vector)
			if err != nil {
				t.Errorf("Insert failed for metric %v: %v", metric, err)
			}
		}

		query := make([]float32, dimensions)
		for i := 0; i < dimensions; i++ {
			query[i] = rand.Float32()
		}

		results, err := graph.Search(query, 5)
		if err != nil {
			t.Errorf("Search failed for metric %v: %v", metric, err)
			continue
		}
		if len(results) != 5 {
			t.Errorf("Expected 5 results for metric %v, got %d", metric, len(results))
		}
	}
}
