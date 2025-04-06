package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"vector-db/config"
	"vector-db/db"

	"github.com/gorilla/websocket"
)

func TestAPIServer(t *testing.T) {
	// Create configuration
	cfg := &config.Config{
		DefaultDatabase: config.DatabaseConfig{
			HNSW: config.HNSWConfig{
				M:              16,
				EfConstruction: 200,
				Dimensions:     128,
				DistanceType:   config.DistanceTypeEuclidean,
			},
		},
	}

	// Create manager and server
	manager := db.NewManager(cfg)
	server := NewServer(manager)

	// Start server in a goroutine
	go func() {
		if err := server.Start(":8080"); err != nil {
			t.Errorf("Failed to start server: %v", err)
		}
	}()

	// Wait for server to start
	time.Sleep(100 * time.Millisecond)

	// Test database creation
	reqBody := map[string]interface{}{
		"name": "test",
		"config": map[string]interface{}{
			"hnsw": map[string]interface{}{
				"m":              16,
				"efConstruction": 200,
				"dimensions":     128,
				"distanceType":   "euclidean",
			},
		},
	}
	jsonBody, _ := json.Marshal(reqBody)
	req := httptest.NewRequest("POST", "/api/databases", bytes.NewBuffer(jsonBody))
	w := httptest.NewRecorder()
	server.HandleDatabases(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	// Test database listing
	req = httptest.NewRequest("GET", "/api/databases", nil)
	w = httptest.NewRecorder()
	server.HandleDatabases(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var dbs []string
	if err := json.NewDecoder(w.Body).Decode(&dbs); err != nil {
		t.Errorf("Failed to decode response: %v", err)
	}
	if len(dbs) != 1 || dbs[0] != "test" {
		t.Errorf("Expected [test], got %v", dbs)
	}

	// Test WebSocket connection
	wsURL := "ws://localhost:8080/api/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect to WebSocket: %v", err)
	}
	defer conn.Close()

	// Test vector addition through WebSocket
	addVectorMsg := map[string]interface{}{
		"type":     "add_vector",
		"database": "test",
		"id":       "test_vector",
		"data":     make([]float32, 128),
		"metadata": map[string]interface{}{"test": true},
	}
	if err := conn.WriteJSON(addVectorMsg); err != nil {
		t.Errorf("Failed to send add_vector message: %v", err)
	}

	// Test vector search through WebSocket
	searchMsg := map[string]interface{}{
		"type":     "search",
		"database": "test",
		"query":    make([]float32, 128),
		"k":        5,
	}
	if err := conn.WriteJSON(searchMsg); err != nil {
		t.Errorf("Failed to send search message: %v", err)
	}

	// Read response
	var response map[string]interface{}
	if err := conn.ReadJSON(&response); err != nil {
		t.Errorf("Failed to read response: %v", err)
	}
	if response["error"] != nil {
		t.Errorf("Received error in response: %v", response["error"])
	}
}
