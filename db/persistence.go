package db

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"vector-db/config"
)

/*
PersistenceManager handles saving and loading databases
*/
type PersistenceManager struct {
	basePath string
	mu       sync.RWMutex
}

/*
NewPersistenceManager creates a new persistence manager
*/
func NewPersistenceManager(basePath string) *PersistenceManager {
	return &PersistenceManager{
		basePath: basePath,
	}
}

/*
SaveDatabase saves a database to disk
*/
func (p *PersistenceManager) SaveDatabase(db *Database) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Create database directory if it doesn't exist
	dbPath := filepath.Join(p.basePath, db.Name)
	if err := os.MkdirAll(dbPath, 0755); err != nil {
		return err
	}

	// Save database configuration
	configPath := filepath.Join(dbPath, "config.json")
	configFile, err := os.Create(configPath)
	if err != nil {
		return err
	}
	defer configFile.Close()

	if err := json.NewEncoder(configFile).Encode(db.Config); err != nil {
		return err
	}

	// Save vectors
	vectorsPath := filepath.Join(dbPath, "vectors.json")
	vectorsFile, err := os.Create(vectorsPath)
	if err != nil {
		return err
	}
	defer vectorsFile.Close()

	db.mu.RLock()
	defer db.mu.RUnlock()

	if err := json.NewEncoder(vectorsFile).Encode(db.Vectors); err != nil {
		return err
	}

	return nil
}

/*
LoadDatabase loads a database from disk
*/
func (p *PersistenceManager) LoadDatabase(name string) (*Database, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	dbPath := filepath.Join(p.basePath, name)

	// Load database configuration
	configPath := filepath.Join(dbPath, "config.json")
	configFile, err := os.Open(configPath)
	if err != nil {
		return nil, err
	}
	defer configFile.Close()

	var dbConfig config.DatabaseConfig
	if err := json.NewDecoder(configFile).Decode(&dbConfig); err != nil {
		return nil, err
	}

	// Load vectors
	vectorsPath := filepath.Join(dbPath, "vectors.json")
	vectorsFile, err := os.Open(vectorsPath)
	if err != nil {
		return nil, err
	}
	defer vectorsFile.Close()

	var vectors map[string]Vector
	if err := json.NewDecoder(vectorsFile).Decode(&vectors); err != nil {
		return nil, err
	}

	return &Database{
		Name:    name,
		Config:  dbConfig,
		Vectors: vectors,
	}, nil
}

/*
DeleteDatabase removes a database from disk
*/
func (p *PersistenceManager) DeleteDatabase(name string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	dbPath := filepath.Join(p.basePath, name)
	return os.RemoveAll(dbPath)
}

/*
ListDatabases returns a list of all saved databases
*/
func (p *PersistenceManager) ListDatabases() ([]string, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	entries, err := os.ReadDir(p.basePath)
	if err != nil {
		return nil, err
	}

	databases := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			databases = append(databases, entry.Name())
		}
	}

	return databases, nil
}
