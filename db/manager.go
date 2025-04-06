package db

import (
	"sync"
	"vector-db/config"
)

/*
Database represents a single vector database
*/
type Database struct {
	Name    string
	Config  config.DatabaseConfig
	Vectors map[string]Vector
	Graph   *HNSWGraph
	mu      sync.RWMutex
}

/*
Manager handles multiple vector databases
*/
type Manager struct {
	databases map[string]*Database
	mu        sync.RWMutex
	config    *config.Config
}

/*
NewManager creates a new database manager
*/
func NewManager(cfg *config.Config) *Manager {
	return &Manager{
		databases: make(map[string]*Database),
		config:    cfg,
	}
}

/*
CreateDatabase creates a new vector database with the given name and configuration
*/
func (m *Manager) CreateDatabase(name string, dbConfig config.DatabaseConfig) (*Database, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.databases[name]; exists {
		return nil, ErrDatabaseExists
	}

	db := &Database{
		Name:    name,
		Config:  dbConfig,
		Vectors: make(map[string]Vector),
		Graph:   NewHNSWGraph(dbConfig.HNSW.M, dbConfig.HNSW.EfConstruction, dbConfig.HNSW.DistanceType),
	}

	m.databases[name] = db
	return db, nil
}

/*
GetDatabase returns a database by name
*/
func (m *Manager) GetDatabase(name string) (*Database, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	db, exists := m.databases[name]
	if !exists {
		return nil, ErrDatabaseNotFound
	}

	return db, nil
}

/*
DeleteDatabase removes a database by name
*/
func (m *Manager) DeleteDatabase(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.databases[name]; !exists {
		return ErrDatabaseNotFound
	}

	delete(m.databases, name)
	return nil
}

/*
ListDatabases returns a list of all database names
*/
func (m *Manager) ListDatabases() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	names := make([]string, 0, len(m.databases))
	for name := range m.databases {
		names = append(names, name)
	}
	return names
}

/*
AddVector adds a vector to a specific database
*/
func (m *Manager) AddVector(dbName string, vector Vector) error {
	db, err := m.GetDatabase(dbName)
	if err != nil {
		return err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if len(vector.Data) != db.Config.HNSW.Dimensions {
		return ErrInvalidDimensions
	}

	db.Vectors[vector.ID] = vector
	db.Graph.Insert(vector)
	return nil
}

/*
GetVector retrieves a vector from a specific database
*/
func (m *Manager) GetVector(dbName, vectorID string) (Vector, error) {
	db, err := m.GetDatabase(dbName)
	if err != nil {
		return Vector{}, err
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	vector, exists := db.Vectors[vectorID]
	if !exists {
		return Vector{}, ErrVectorNotFound
	}

	return vector, nil
}

/*
DeleteVector removes a vector from a specific database
*/
func (m *Manager) DeleteVector(dbName, vectorID string) error {
	db, err := m.GetDatabase(dbName)
	if err != nil {
		return err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if _, exists := db.Vectors[vectorID]; !exists {
		return ErrVectorNotFound
	}

	delete(db.Vectors, vectorID)
	// TODO: Implement vector deletion from HNSW graph
	return nil
}

/*
Search performs a similarity search in a specific database
*/
func (m *Manager) Search(dbName string, query []float32, k int) ([]Vector, error) {
	db, err := m.GetDatabase(dbName)
	if err != nil {
		return nil, err
	}

	if len(query) != db.Config.HNSW.Dimensions {
		return nil, ErrInvalidDimensions
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	return db.Graph.Search(query, k), nil
}
