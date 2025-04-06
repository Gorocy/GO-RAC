package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
)

/*
Config is the configuration for the application.

Contains the configuration for the server, storage, and multiple vector databases.
*/
type Config struct {
	Server          ServerConfig              `json:"server"`
	Storage         StorageConfig             `json:"storage"`
	Databases       map[string]DatabaseConfig `json:"databases"`
	LogLevel        string                    `json:"log_level"`
	DefaultDatabase DatabaseConfig            `yaml:"default_database"`
}

/*
ServerConfig is the configuration for the server.
*/
type ServerConfig struct {
	Host string `json:"host"`
	Port string `json:"port"`
}

/*
HNSWConfig is the configuration for the HNSW algorithm.
*/
type HNSWConfig struct {
	// number of dimensions
	Dimensions int `json:"dimensions"`
	// number of neighbors
	M int `json:"m"`
	// parameter efConstruction for HNSW
	EfConstruction int `json:"ef_construction"`
	// parameter efSearch for HNSW
	EfSearch int `json:"ef_search"`
	// distance function
	DistanceType DistanceType `json:"distance_type"`
}

/*
DistanceType is the type of distance function.
*/
type DistanceType int

const (
	DistanceTypeEuclidean DistanceType = iota
	DistanceTypeCosine    DistanceType = iota
	DistanceTypeManhattan DistanceType = iota
	DistanceTypeHamming   DistanceType = iota
)

/*
StorageConfig is the configuration for the storage.
*/
type StorageConfig struct {
	// path to the data file
	DataPath string `json:"data_path"`
	// whether to use persistence engine
	PersistenceEngine bool `json:"persistence_engine"`
	// interval to persist data [seconds]
	PersistenceInterval int `json:"persistence_interval"`
}

/*
DatabaseConfig represents the configuration for a single vector database.
*/
type DatabaseConfig struct {
	HNSW HNSWConfig `json:"hnsw"`
	// Additional database-specific settings can be added here
}

/*
Default config
*/
func DefaultConfig() *Config {
	return &Config{
		// server configuration
		Server: ServerConfig{
			Host: "localhost",
			Port: "8080",
		},
		// storage configuration
		Storage: StorageConfig{
			DataPath:            "./data",
			PersistenceEngine:   true,
			PersistenceInterval: 5,
		},
		// default database configuration
		Databases: map[string]DatabaseConfig{
			"default": {
				HNSW: HNSWConfig{
					Dimensions:     128,
					M:              16,
					EfConstruction: 200,
					EfSearch:       100,
					DistanceType:   DistanceTypeEuclidean,
				},
			},
		},
		// logging configuration
		LogLevel: "warn",
	}
}

/*
LoadFromFile loads the configuration from a JSON file.
*/
func LoadFromFile(path string) (*Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	config := DefaultConfig()
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(config); err != nil {
		return nil, err
	}
	return config, nil
}

/*
LoadFromEnv loads the configuration from the environment variables.
*/
func LoadFromEnv() (*Config, error) {
	config := DefaultConfig()

	// Server config
	if host := os.Getenv("GORAC_HOST"); host != "" {
		config.Server.Host = host
	}

	if portStr := os.Getenv("GORAC_PORT"); portStr != "" {
		config.Server.Port = portStr
	}

	// Default database HNSW config
	defaultDB := config.Databases["default"]

	if dimsStr := os.Getenv("GORAC_DIMS"); dimsStr != "" {
		if dims, err := strconv.Atoi(dimsStr); err == nil {
			defaultDB.HNSW.Dimensions = dims
		}
	}

	if mStr := os.Getenv("GORAC_M"); mStr != "" {
		if m, err := strconv.Atoi(mStr); err == nil {
			defaultDB.HNSW.M = m
		}
	}

	if efConstructionStr := os.Getenv("GORAC_EF_CONSTRUCTION"); efConstructionStr != "" {
		if efConstruction, err := strconv.Atoi(efConstructionStr); err == nil {
			defaultDB.HNSW.EfConstruction = efConstruction
		}
	}

	if efSearchStr := os.Getenv("GORAC_EF_SEARCH"); efSearchStr != "" {
		if efSearch, err := strconv.Atoi(efSearchStr); err == nil {
			defaultDB.HNSW.EfSearch = efSearch
		}
	}

	if distType := os.Getenv("GORAC_DISTANCE_TYPE"); distType != "" {
		if distTypeInt, err := strconv.Atoi(distType); err == nil {
			defaultDB.HNSW.DistanceType = DistanceType(distTypeInt)
		}
	}

	config.Databases["default"] = defaultDB

	// Storage config
	if dataPath := os.Getenv("GORAC_DATA_PATH"); dataPath != "" {
		config.Storage.DataPath = dataPath
	}

	if persistStr := os.Getenv("GORAC_PERSISTENCE_ENABLED"); persistStr != "" {
		if persist, err := strconv.ParseBool(persistStr); err == nil {
			config.Storage.PersistenceEngine = persist
		}
	}

	if intervalStr := os.Getenv("GORAC_AUTOSAVE_INTERVAL"); intervalStr != "" {
		if interval, err := strconv.Atoi(intervalStr); err == nil {
			config.Storage.PersistenceInterval = interval
		}
	}

	return config, nil
}

/*
Validate checks if the configuration is valid
*/
func (c *Config) Validate() error {
	if c.DefaultDatabase.HNSW.M <= 0 {
		return fmt.Errorf("invalid M value: %d", c.DefaultDatabase.HNSW.M)
	}
	if c.DefaultDatabase.HNSW.Dimensions <= 0 {
		return fmt.Errorf("invalid dimensions: %d", c.DefaultDatabase.HNSW.Dimensions)
	}
	return nil
}

/*
String returns the string representation of the distance type
*/
func (dt DistanceType) String() string {
	switch dt {
	case DistanceTypeEuclidean:
		return "euclidean"
	case DistanceTypeCosine:
		return "cosine"
	case DistanceTypeManhattan:
		return "manhattan"
	case DistanceTypeHamming:
		return "hamming"
	default:
		return "unknown"
	}
}

/*
ParseDistanceType converts a string to a DistanceType
*/
func ParseDistanceType(s string) DistanceType {
	switch s {
	case "euclidean":
		return DistanceTypeEuclidean
	case "cosine":
		return DistanceTypeCosine
	case "manhattan":
		return DistanceTypeManhattan
	case "hamming":
		return DistanceTypeHamming
	default:
		return DistanceTypeEuclidean
	}
}
