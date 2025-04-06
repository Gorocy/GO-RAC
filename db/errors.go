package db

import "errors"

var (
	// ErrDatabaseExists is returned when trying to create a database that already exists
	ErrDatabaseExists = errors.New("database already exists")

	// ErrDatabaseNotFound is returned when trying to access a non-existent database
	ErrDatabaseNotFound = errors.New("database not found")

	// ErrVectorNotFound is returned when trying to access a non-existent vector
	ErrVectorNotFound = errors.New("vector not found")

	// ErrInvalidDimensions is returned when vector dimensions don't match the database configuration
	ErrInvalidDimensions = errors.New("invalid vector dimensions")
)
