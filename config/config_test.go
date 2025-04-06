package config

import (
	"testing"
)

func TestConfigValidation(t *testing.T) {
	// Test valid config
	validConfig := &Config{
		DefaultDatabase: DatabaseConfig{
			HNSW: HNSWConfig{
				M:              16,
				EfConstruction: 200,
				Dimensions:     128,
				DistanceType:   DistanceTypeEuclidean,
			},
		},
	}

	if err := validConfig.Validate(); err != nil {
		t.Errorf("Valid config should not return error: %v", err)
	}

	// Test invalid M value
	invalidMConfig := &Config{
		DefaultDatabase: DatabaseConfig{
			HNSW: HNSWConfig{
				M:              0, // Invalid
				EfConstruction: 200,
				Dimensions:     128,
				DistanceType:   DistanceTypeEuclidean,
			},
		},
	}

	if err := invalidMConfig.Validate(); err == nil {
		t.Error("Config with invalid M should return error")
	}

	// Test invalid dimensions
	invalidDimConfig := &Config{
		DefaultDatabase: DatabaseConfig{
			HNSW: HNSWConfig{
				M:              16,
				EfConstruction: 200,
				Dimensions:     0, // Invalid
				DistanceType:   DistanceTypeEuclidean,
			},
		},
	}

	if err := invalidDimConfig.Validate(); err == nil {
		t.Error("Config with invalid dimensions should return error")
	}
}

func TestDistanceTypeString(t *testing.T) {
	tests := []struct {
		dt     DistanceType
		expect string
	}{
		{DistanceTypeEuclidean, "euclidean"},
		{DistanceTypeCosine, "cosine"},
		{DistanceTypeManhattan, "manhattan"},
		{DistanceTypeHamming, "hamming"},
		{DistanceType(999), "unknown"},
	}

	for _, test := range tests {
		if got := test.dt.String(); got != test.expect {
			t.Errorf("Expected %s for %v, got %s", test.expect, test.dt, got)
		}
	}
}

func TestParseDistanceType(t *testing.T) {
	tests := []struct {
		input  string
		expect DistanceType
	}{
		{"euclidean", DistanceTypeEuclidean},
		{"cosine", DistanceTypeCosine},
		{"manhattan", DistanceTypeManhattan},
		{"hamming", DistanceTypeHamming},
		{"unknown", DistanceTypeEuclidean}, // Default
	}

	for _, test := range tests {
		if got := ParseDistanceType(test.input); got != test.expect {
			t.Errorf("Expected %v for %s, got %v", test.expect, test.input, got)
		}
	}
}
