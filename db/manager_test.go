package db

import (
	"fmt"
	"testing"

	"vector-db/config"
)

func TestDatabaseManager(t *testing.T) {
	// Create configuration
	cfg := &config.Config{
		DefaultDatabase: config.DatabaseConfig{
			HNSW: config.HNSWConfig{
				M:              8,
				EfConstruction: 100,
				Dimensions:     64,
				DistanceType:   config.DistanceTypeEuclidean,
			},
		},
	}

	// Create manager
	manager := NewManager(cfg)

	// Test database creation
	db1, err := manager.CreateDatabase("test1", cfg.DefaultDatabase)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	if db1 == nil {
		t.Fatal("Created database is nil")
	}

	// Test duplicate database creation
	_, err = manager.CreateDatabase("test1", cfg.DefaultDatabase)
	if err == nil {
		t.Fatal("Expected error when creating duplicate database")
	}

	// Test database retrieval
	db2, err := manager.GetDatabase("test1")
	if err != nil {
		t.Fatalf("Failed to get database: %v", err)
	}
	if db2 != db1 {
		t.Fatal("Retrieved database is not the same as created one")
	}

	// Test database listing
	dbs := manager.ListDatabases()
	if len(dbs) != 1 || dbs[0] != "test1" {
		t.Fatalf("Expected [test1], got %v", dbs)
	}

	// Test database deletion
	err = manager.DeleteDatabase("test1")
	if err != nil {
		t.Fatalf("Failed to delete database: %v", err)
	}

	// Verify deletion
	_, err = manager.GetDatabase("test1")
	if err == nil {
		t.Fatal("Expected error when getting deleted database")
	}
}

func TestVectorOperations(t *testing.T) {
	cfg := &config.Config{
		DefaultDatabase: config.DatabaseConfig{
			HNSW: config.HNSWConfig{
				M:              8,
				EfConstruction: 100,
				Dimensions:     64,
				DistanceType:   config.DistanceTypeEuclidean,
			},
		},
	}

	manager := NewManager(cfg)
	_, _ = manager.CreateDatabase("test", cfg.DefaultDatabase)

	// Create test vectors
	vectors := make([]Vector, 50)
	for i := 0; i < 50; i++ {
		vector := make([]float32, 64)
		for j := 0; j < 64; j++ {
			vector[j] = float32(i) // Simple pattern for testing
		}
		vectors[i] = Vector{
			ID:       fmt.Sprintf("%d", i),
			Data:     vector,
			Metadata: map[string]interface{}{"index": i},
		}
	}

	// Test vector addition
	for _, vector := range vectors {
		err := manager.AddVector("test", vector)
		if err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Test vector retrieval
	for i := 0; i < 50; i++ {
		vector, err := manager.GetVector("test", fmt.Sprintf("%d", i))
		if err != nil {
			t.Fatalf("Failed to get vector: %v", err)
		}
		if vector.ID != fmt.Sprintf("%d", i) {
			t.Fatalf("Expected vector ID %d, got %s", i, vector.ID)
		}
	}

	// Test vector search
	query := make([]float32, 64)
	for i := 0; i < 64; i++ {
		query[i] = 25.0 // Should find vectors close to index 25
	}

	results, err := manager.Search("test", query, 5)
	if err != nil {
		t.Fatalf("Failed to search vectors: %v", err)
	}
	if len(results) != 5 {
		t.Fatalf("Expected 5 results, got %d", len(results))
	}

	// Verify results are close to query
	for _, result := range results {
		index := result.Metadata["index"].(int)
		if index < 20 || index > 30 {
			t.Errorf("Result index %d is not close to expected range [20,30]", index)
		}
	}
}

func TestConcurrentDatabaseOperations(t *testing.T) {
	cfg := &config.Config{
		DefaultDatabase: config.DatabaseConfig{
			HNSW: config.HNSWConfig{
				M:              8,
				EfConstruction: 100,
				Dimensions:     64,
				DistanceType:   config.DistanceTypeEuclidean,
			},
		},
	}

	manager := NewManager(cfg)
	done := make(chan bool)

	// Concurrent database creation
	for i := 0; i < 2; i++ {
		go func(id int) {
			dbName := fmt.Sprintf("%d", id)
			_, err := manager.CreateDatabase(dbName, cfg.DefaultDatabase)
			if err != nil {
				t.Errorf("Failed to create database %s: %v", dbName, err)
			}
			done <- true
		}(i)
	}

	// Wait for all creations to complete
	for i := 0; i < 2; i++ {
		<-done
	}

	// Verify all databases were created
	dbs := manager.ListDatabases()
	if len(dbs) != 2 {
		t.Fatalf("Expected 2 databases, got %d", len(dbs))
	}

	// Concurrent vector operations
	for i := 0; i < 2; i++ {
		go func(dbName string) {
			// Add vector
			vector := Vector{
				ID:   "test",
				Data: make([]float32, 64),
			}
			err := manager.AddVector(dbName, vector)
			if err != nil {
				t.Errorf("Failed to add vector to %s: %v", dbName, err)
			}

			// Search
			_, err = manager.Search(dbName, make([]float32, 64), 5)
			if err != nil {
				t.Errorf("Failed to search in %s: %v", dbName, err)
			}
			done <- true
		}(fmt.Sprintf("%d", i))
	}

	// Wait for all operations to complete
	for i := 0; i < 2; i++ {
		<-done
	}
}
