package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	log "github.com/sirupsen/logrus"

	"vector-db/api"
	"vector-db/config"
	"vector-db/db"
)

func main() {
	// load the environment variables
	_ = godotenv.Load()

	// parse the command line arguments
	cfg := parseFlags()

	// Initialize logging
	level, err := log.ParseLevel(cfg.LogLevel)
	if err != nil {
		level = log.WarnLevel
	}
	log.SetLevel(level)

	// Print welcome message
	printWelcome()

	// Create database manager
	dbManager := db.NewManager(cfg)
	persistence := db.NewPersistenceManager(cfg.Storage.DataPath)

	// Load existing databases
	if err := loadDatabases(dbManager, persistence); err != nil {
		log.Fatal("Failed to load databases: ", err)
	}

	// Start persistence worker
	stopPersistence := make(chan struct{})
	go persistenceWorker(dbManager, persistence, cfg.Storage.PersistenceInterval, stopPersistence)

	// Create and start API server
	apiServer := api.NewServer(dbManager)
	go func() {
		addr := fmt.Sprintf("%s:%s", cfg.Server.Host, cfg.Server.Port)
		log.Info("Starting API server on ", addr)
		if err := apiServer.Start(addr); err != nil {
			log.Fatal("Failed to start API server: ", err)
		}
	}()

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for shutdown signal
	<-sigChan
	log.Info("Shutting down...")

	// Stop persistence worker
	close(stopPersistence)

	// Save all databases one last time
	if err := saveAllDatabases(dbManager, persistence); err != nil {
		log.Error("Failed to save databases during shutdown: ", err)
	}
}

func loadDatabases(dbManager *db.Manager, persistence *db.PersistenceManager) error {
	databases, err := persistence.ListDatabases()
	if err != nil {
		return err
	}

	for _, name := range databases {
		db, err := persistence.LoadDatabase(name)
		if err != nil {
			log.Errorf("Failed to load database %s: %v", name, err)
			continue
		}

		if _, err := dbManager.CreateDatabase(name, db.Config); err != nil {
			log.Errorf("Failed to create database %s: %v", name, err)
			continue
		}
	}

	return nil
}

func saveAllDatabases(dbManager *db.Manager, persistence *db.PersistenceManager) error {
	for _, name := range dbManager.ListDatabases() {
		db, err := dbManager.GetDatabase(name)
		if err != nil {
			log.Errorf("Failed to get database %s: %v", name, err)
			continue
		}

		if err := persistence.SaveDatabase(db); err != nil {
			log.Errorf("Failed to save database %s: %v", name, err)
			continue
		}
	}

	return nil
}

func persistenceWorker(dbManager *db.Manager, persistence *db.PersistenceManager, interval int, stop chan struct{}) {
	ticker := time.NewTicker(time.Duration(interval) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := saveAllDatabases(dbManager, persistence); err != nil {
				log.Error("Failed to save databases: ", err)
			}
		case <-stop:
			return
		}
	}
}

func parseFlags() *config.Config {
	// Load default config
	cfg, err := config.LoadFromFile("./config.json")
	if err != nil {
		cfg = config.DefaultConfig()
	}

	// Server flags
	flag.StringVar(&cfg.Server.Host, "host", cfg.Server.Host, "Host address")
	flag.StringVar(&cfg.Server.Port, "port", cfg.Server.Port, "Port number")

	// Storage flags
	flag.StringVar(&cfg.Storage.DataPath, "data-path", cfg.Storage.DataPath, "Path to store data files")
	flag.BoolVar(&cfg.Storage.PersistenceEngine, "persistence", cfg.Storage.PersistenceEngine, "Enable persistence engine")
	flag.IntVar(&cfg.Storage.PersistenceInterval, "persistence-interval", cfg.Storage.PersistenceInterval, "Persistence interval in seconds")

	// Default database HNSW flags
	defaultDB := cfg.Databases["default"]
	flag.IntVar(&defaultDB.HNSW.Dimensions, "dims", defaultDB.HNSW.Dimensions, "Number of dimensions")
	flag.IntVar(&defaultDB.HNSW.M, "neighbors", defaultDB.HNSW.M, "Number of neighbors for HNSW")
	flag.IntVar(&defaultDB.HNSW.EfConstruction, "ef-construction", defaultDB.HNSW.EfConstruction, "Parameter efConstruction for HNSW")
	flag.IntVar(&defaultDB.HNSW.EfSearch, "ef-search", defaultDB.HNSW.EfSearch, "Parameter efSearch for HNSW")
	flag.IntVar((*int)(&defaultDB.HNSW.DistanceType), "distance-type", int(defaultDB.HNSW.DistanceType), "Distance function type (0=euclidean, 1=cosine, 2=manhattan, 3=hamming)")

	// Log level flag
	flag.StringVar(&cfg.LogLevel, "log-level", "warn", "Log level (debug, info, warn, error, fatal)")

	// Parse flags
	flag.Parse()

	// Update default database config
	cfg.Databases["default"] = defaultDB

	return cfg
}

func printWelcome() {
	fmt.Println(" .d8888b.   .d88888b.  8888888b.         d8888  .d8888b.  ")
	fmt.Println("d88P  'Y88b d88P  'Y88b 888   Y88b      d88888 d88P  Y88b ")
	fmt.Println("888    888 888     888 888    888      d88P888 888    888 ")
	fmt.Println("888        888     888 888   d88P     d88P 888 888        ")
	fmt.Println("888  88888 888     888 8888888P'     d88P  888 888        ")
	fmt.Println("888    888 888     888 888 T88b     d88P   888 888    888 ")
	fmt.Println("Y88b  d88P Y88b. .d88P 888  T88b   d8888888888 Y88b  d88P ")
	fmt.Println(" 'Y8888P88  'Y88888P'  888   T88b d88P     888  'Y8888P'  ")
}
