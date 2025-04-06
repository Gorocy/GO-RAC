package db

import (
	"math"
	"math/rand"
	"sort"
	"sync"

	"vector-db/config"
)

/*
HNSWGraph represents the HNSW graph structure
*/
type HNSWGraph struct {
	// Maximum number of connections per layer
	M int
	// Size of the dynamic candidate list
	EfConstruction int
	// Maximum layer
	MaxLayer int
	// Entry point
	EntryPoint string
	// Layers of the graph
	Layers []map[string][]string
	// Vector data
	Vectors map[string]Vector
	// Distance function
	DistanceType config.DistanceType
	// Mutex for thread safety
	mu sync.RWMutex
}

/*
NewHNSWGraph creates a new HNSW graph
*/
func NewHNSWGraph(m, efConstruction int, distanceType config.DistanceType) *HNSWGraph {
	return &HNSWGraph{
		M:              m,
		EfConstruction: efConstruction,
		MaxLayer:       0,
		Layers:         []map[string][]string{make(map[string][]string)},
		Vectors:        make(map[string]Vector),
		DistanceType:   distanceType,
	}
}

/*
Insert adds a new vector to the graph
*/
func (g *HNSWGraph) Insert(vector Vector) {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Generate random layer
	layer := int(math.Floor(-math.Log(rand.Float64()) * float64(g.M)))
	if layer > g.MaxLayer {
		g.MaxLayer = layer
		for i := len(g.Layers); i <= layer; i++ {
			g.Layers = append(g.Layers, make(map[string][]string))
		}
	}

	// Store vector
	g.Vectors[vector.ID] = vector

	// If this is the first vector, set it as entry point
	if g.EntryPoint == "" {
		g.EntryPoint = vector.ID
		return
	}

	// Find nearest neighbors in each layer
	nearest := g.searchLayer(vector.Data, g.EntryPoint, 1, g.MaxLayer)
	for l := min(layer, g.MaxLayer); l >= 0; l-- {
		nearest = g.searchLayer(vector.Data, nearest[0], g.EfConstruction, l)
		neighbors := g.selectNeighbors(vector.Data, []string{nearest[0]}, g.M, l)

		// Add bidirectional connections
		g.Layers[l][vector.ID] = neighbors
		for _, neighbor := range neighbors {
			g.Layers[l][neighbor] = append(g.Layers[l][neighbor], vector.ID)
			if len(g.Layers[l][neighbor]) > g.M {
				g.Layers[l][neighbor] = g.selectNeighbors(g.Vectors[neighbor].Data, g.Layers[l][neighbor], g.M, l)
			}
		}
	}
}

/*
Search finds the k nearest neighbors to a query vector
*/
func (g *HNSWGraph) Search(query []float32, k int) []Vector {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.EntryPoint == "" {
		return []Vector{}
	}

	// Start from the top layer
	current := g.EntryPoint
	results := make([]string, 0, k)

	// Search through all layers
	for l := g.MaxLayer; l >= 0; l-- {
		// Search in current layer
		layerResults := g.searchLayer(query, current, k, l)
		if len(layerResults) > 0 {
			results = layerResults
			current = layerResults[0]
		}
	}

	// Convert to vectors
	vectors := make([]Vector, 0, len(results))
	for _, id := range results {
		vectors = append(vectors, g.Vectors[id])
	}

	return vectors
}

/*
searchLayer searches for the nearest neighbors in a specific layer
*/
func (g *HNSWGraph) searchLayer(query []float32, entryPoint string, k int, layer int) []string {
	visited := make(map[string]bool)
	candidates := make([]string, 0, k)
	results := make([]string, 0, k)

	candidates = append(candidates, entryPoint)
	visited[entryPoint] = true

	for len(candidates) > 0 {
		current := candidates[0]
		candidates = candidates[1:]

		// Check if current is better than the worst in results
		if len(results) < k || g.Distance(query, g.Vectors[current].Data) < g.Distance(query, g.Vectors[results[len(results)-1]].Data) {
			// Add to results
			results = append(results, current)
			sort.Slice(results, func(i, j int) bool {
				return g.Distance(query, g.Vectors[results[i]].Data) < g.Distance(query, g.Vectors[results[j]].Data)
			})
			if len(results) > k {
				results = results[:k]
			}

			// Add neighbors to candidates
			for _, neighbor := range g.Layers[layer][current] {
				if !visited[neighbor] {
					visited[neighbor] = true
					candidates = append(candidates, neighbor)
				}
			}
		}
	}

	return results
}

/*
selectNeighbors selects the M nearest neighbors from a set of candidates
*/
func (g *HNSWGraph) selectNeighbors(query []float32, candidates []string, m int, _ int) []string {
	if len(candidates) <= m {
		return candidates
	}

	// Sort candidates by distance
	sort.Slice(candidates, func(i, j int) bool {
		return g.Distance(query, g.Vectors[candidates[i]].Data) < g.Distance(query, g.Vectors[candidates[j]].Data)
	})

	return candidates[:m]
}

/*
Distance calculates the distance between two vectors based on the configured distance type
*/
func (g *HNSWGraph) Distance(a, b []float32) float32 {
	switch g.DistanceType {
	case config.DistanceTypeEuclidean:
		return g.euclideanDistance(a, b)
	case config.DistanceTypeCosine:
		return g.cosineDistance(a, b)
	case config.DistanceTypeManhattan:
		return g.manhattanDistance(a, b)
	case config.DistanceTypeHamming:
		return g.hammingDistance(a, b)
	default:
		return g.euclideanDistance(a, b)
	}
}

/*
Helper functions for different distance metrics
*/
func (g *HNSWGraph) euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

/*
cosineDistance calculates the cosine distance between two vectors
*/
func (g *HNSWGraph) cosineDistance(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	return 1 - dot/(float32(math.Sqrt(float64(normA)))*float32(math.Sqrt(float64(normB))))
}

/*
manhattanDistance calculates the Manhattan distance between two vectors
*/
func (g *HNSWGraph) manhattanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += float32(math.Abs(float64(a[i] - b[i])))
	}
	return sum
}

/*
hammingDistance calculates the Hamming distance between two vectors
*/
func (g *HNSWGraph) hammingDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		if a[i] != b[i] {
			sum++
		}
	}
	return sum
}

/*
min returns the smaller of two integers
*/
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
