package api

import (
	"encoding/json"
	"net/http"
	"time"

	"vector-db/config"
	"vector-db/db"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins
	},
}

/*
Server represents the API server
*/
type Server struct {
	dbManager *db.Manager
}

/*
NewServer creates a new API server
*/
func NewServer(dbManager *db.Manager) *Server {
	return &Server{
		dbManager: dbManager,
	}
}

/*
Start starts the HTTP server
*/
func (s *Server) Start(addr string) error {
	http.HandleFunc("/api/databases", s.HandleDatabases)
	http.HandleFunc("/api/databases/", s.handleDatabase)
	http.HandleFunc("/api/ws", s.handleWebSocket)
	return http.ListenAndServe(addr, nil)
}

/*
HandleDatabases handles database list and creation
*/
func (s *Server) HandleDatabases(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.handleListDatabases(w, r)
	case http.MethodPost:
		s.handleCreateDatabase(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

/*
handleDatabase handles database operations
*/
func (s *Server) handleDatabase(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.getDatabase(w, r)
	case http.MethodDelete:
		s.deleteDatabase(w, r)
	case http.MethodPost:
		s.addVector(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

/*
handleWebSocket handles WebSocket connections
*/
func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		http.Error(w, "Failed to upgrade connection", http.StatusInternalServerError)
		return
	}
	defer conn.Close()

	// Set read deadline
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	// Handle WebSocket messages
	for {
		messageType, p, err := conn.ReadMessage()
		if err != nil {
			return
		}

		// Process message
		var request map[string]interface{}
		if err := json.Unmarshal(p, &request); err != nil {
			conn.WriteMessage(messageType, []byte(`{"error": "Invalid JSON"}`))
			continue
		}

		// Handle different message types
		switch request["type"] {
		case "search":
			s.handleSearch(conn, messageType, request)
		case "add_vector":
			s.handleAddVector(conn, messageType, request)
		default:
			conn.WriteMessage(messageType, []byte(`{"error": "Unknown message type"}`))
		}
	}
}

/*
Helper methods for HTTP handlers
*/
func (s *Server) handleListDatabases(w http.ResponseWriter, _ *http.Request) {
	databases := s.dbManager.ListDatabases()
	json.NewEncoder(w).Encode(databases)
}

/*
CreateDatabase creates a new database
*/
func (s *Server) handleCreateDatabase(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Name   string
		Config config.DatabaseConfig
	}
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	db, err := s.dbManager.CreateDatabase(request.Name, request.Config)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(db)
}

/*
GetDatabase gets a database by name
*/
func (s *Server) getDatabase(w http.ResponseWriter, r *http.Request) {
	// Extract database name from URL
	dbName := r.URL.Path[len("/api/databases/"):]
	db, err := s.dbManager.GetDatabase(dbName)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	json.NewEncoder(w).Encode(db)
}

/*
DeleteDatabase deletes a database by name
*/
func (s *Server) deleteDatabase(w http.ResponseWriter, r *http.Request) {
	dbName := r.URL.Path[len("/api/databases/"):]
	if err := s.dbManager.DeleteDatabase(dbName); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

/*
AddVector adds a vector to a specific database
*/
func (s *Server) addVector(w http.ResponseWriter, r *http.Request) {
	dbName := r.URL.Path[len("/api/databases/"):]
	var vector db.Vector
	if err := json.NewDecoder(r.Body).Decode(&vector); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if err := s.dbManager.AddVector(dbName, vector); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
}

/*
Helper methods for WebSocket handlers
*/
func (s *Server) handleSearch(conn *websocket.Conn, messageType int, request map[string]interface{}) {
	dbName := request["database"].(string)
	query := request["query"].([]float32)
	k := int(request["k"].(float64))

	results, err := s.dbManager.Search(dbName, query, k)
	if err != nil {
		conn.WriteMessage(messageType, []byte(`{"error": "`+err.Error()+`"}`))
		return
	}

	response, _ := json.Marshal(results)
	conn.WriteMessage(messageType, response)
}

func (s *Server) handleAddVector(conn *websocket.Conn, messageType int, request map[string]interface{}) {
	dbName := request["database"].(string)

	// Convert interface{} to []float32
	data := request["data"].([]interface{})
	vectorData := make([]float32, len(data))
	for i, v := range data {
		vectorData[i] = float32(v.(float64))
	}

	vector := db.Vector{
		ID:       request["id"].(string),
		Data:     vectorData,
		Metadata: request["metadata"].(map[string]interface{}),
	}

	if err := s.dbManager.AddVector(dbName, vector); err != nil {
		conn.WriteMessage(messageType, []byte(`{"error": "`+err.Error()+`"}`))
		return
	}

	conn.WriteMessage(messageType, []byte(`{"status": "success"}`))
}
