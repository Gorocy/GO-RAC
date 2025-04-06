package db

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"vector-db/config"
)

/*
HNSWGraph represents the Hierarchical Navigable Small World graph structure.

HNSW is an approximate nearest neighbor search algorithm that builds a multi-layer
graph with skip-list-like properties. Each layer is a navigable small world graph,
with the number of connections decreasing as you go up the layers. This structure
allows for logarithmic search complexity in practice.

Key parameters:
- M: Controls the maximum number of connections per node in the graph
- EfConstruction: Controls the size of the dynamic candidate list during graph construction
- EfSearch: Controls the size of the dynamic candidate list during search
*/
type HNSWGraph struct {
	// Maximum number of connections per layer
	M int
	// Size of the dynamic candidate list during construction
	EfConstruction int
	// Size of the dynamic candidate list during search
	EfSearch int
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
	// Normalization factor for level generation
	mL float64
}

// Errors
var (
	ErrEmptyVector      = errors.New("vector is empty")
	ErrDifferentDims    = errors.New("vectors have different dimensions")
	ErrInvalidParameter = errors.New("invalid parameter")
)

/*
NewHNSWGraph creates a new HNSW graph with the specified parameters.

Parameters:
- m: Maximum number of connections per node (recommended: 5-48)
- efConstruction: Size of the dynamic candidate list during construction (recommended: 100-200)
- distanceType: The distance metric to use

The mL parameter is calculated as 1/ln(M) and is used for generating the probability
distribution of nodes across layers.
*/
func NewHNSWGraph(m, efConstruction int, distanceType config.DistanceType) *HNSWGraph {
	if m <= 0 {
		m = 16 // Default value if invalid
	}
	if efConstruction <= 0 {
		efConstruction = 200 // Default value if invalid
	}

	// Calculate mL - ensure M > 1 to avoid log(1)=0
	var ml float64
	if m > 1 {
		ml = 1.0 / math.Log(float64(m))
	} else {
		// Handle M=1 case (though HNSW typically uses M > 1)
		ml = 1.0
	}

	return &HNSWGraph{
		M:              m,
		EfConstruction: efConstruction,
		EfSearch:       efConstruction, // Default to same as construction
		MaxLayer:       0,
		Layers:         []map[string][]string{make(map[string][]string)},
		Vectors:        make(map[string]Vector),
		DistanceType:   distanceType,
		mL:             ml,
	}
}

/*
Insert adds a new vector to the graph.

The process works as follows:
1. Randomly assign a layer level to the new vector using the mL normalization factor
2. If this is the first vector, it becomes the entry point
3. Find the best entry point for the target layer by descending from the top layer
4. For each layer from the target down to 0, find neighbors and establish connections
5. For each bidirectional connection, trim neighbor's connections if they exceed M

The random layer assignment is a key feature of HNSW, creating a probabilistic
skip-list-like structure where ~1/e nodes of layer l appear in layer l+1.
*/
func (g *HNSWGraph) Insert(vector Vector) error {
	// Validate vector
	if len(vector.Data) == 0 {
		return ErrEmptyVector
	}
	if vector.ID == "" {
		return ErrInvalidParameter
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	// Check if vector with this ID already exists
	if _, exists := g.Vectors[vector.ID]; exists {
		return fmt.Errorf("vector with ID %s already exists", vector.ID)
	}

	// Calculate layer using mL
	layer := 0
	if g.mL > 0 {
		levelRand := rand.Float64()
		if levelRand == 0 {
			levelRand = math.SmallestNonzeroFloat64
		}
		layer = int(math.Floor(-math.Log(levelRand) * g.mL))
	}

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
		return nil
	}

	// First phase: Find the best entry point for the target layer
	entryPointForLayer := g.EntryPoint
	for l := g.MaxLayer; l > layer; l-- {
		pathCandidates := g.searchLayer(vector.Data, entryPointForLayer, 1, l)
		if len(pathCandidates) > 0 {
			entryPointForLayer = pathCandidates[0]
		}
	}

	// Second phase: Insert the vector in each layer from layer down to 0
	for l := min(layer, g.MaxLayer); l >= 0; l-- {
		// Find potential neighbors in layer l
		nearestCandidates := g.searchLayer(vector.Data, entryPointForLayer, g.EfConstruction, l)

		// Select M best neighbors from the candidates
		neighbors := g.selectNeighbors(vector.Data, nearestCandidates, g.M)

		// Add bidirectional connections
		g.Layers[l][vector.ID] = neighbors
		for _, neighbor := range neighbors {
			if _, ok := g.Layers[l][neighbor]; !ok {
				g.Layers[l][neighbor] = []string{}
			}
			g.Layers[l][neighbor] = append(g.Layers[l][neighbor], vector.ID)

			// Trim neighbor's connections if they exceed M
			if len(g.Layers[l][neighbor]) > g.M {
				neighborVectorData := g.Vectors[neighbor].Data
				g.Layers[l][neighbor] = g.selectNeighbors(neighborVectorData, g.Layers[l][neighbor], g.M)
			}
		}

		// Update entry point for next layer
		if len(nearestCandidates) > 0 {
			entryPointForLayer = nearestCandidates[0]
		}
	}

	return nil
}

/*
Search finds the k nearest neighbors to a query vector.

The search process works in two phases:
1. Descend from the top layer to layer 1, finding the best entry point for each layer
2. Perform a detailed search in layer 0 using efSearch candidates

This approach allows the algorithm to quickly navigate to the approximate region
of interest in higher layers, then perform a more detailed search in the lowest layer.
*/
func (g *HNSWGraph) Search(query []float32, k int) ([]Vector, error) {
	// Validate parameters
	if len(query) == 0 {
		return nil, ErrEmptyVector
	}
	if k <= 0 {
		return nil, ErrInvalidParameter
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.EntryPoint == "" {
		return []Vector{}, nil
	}

	// Phase 1: Descend from top layer to layer 1 (only finding path)
	currentEntryPoint := g.EntryPoint
	for l := g.MaxLayer; l > 0; l-- {
		// Use small number of candidates (1) to find best entry point for next layer
		pathCandidates := g.searchLayer(query, currentEntryPoint, 1, l)
		if len(pathCandidates) > 0 {
			currentEntryPoint = pathCandidates[0]
		} else {
			// If no candidates found, break the descent
			break
		}
	}

	// Phase 2: Detailed search in layer 0
	finalCandidates := g.searchLayer(query, currentEntryPoint, g.EfSearch, 0)

	// Trim to k results
	if len(finalCandidates) > k {
		finalCandidates = finalCandidates[:k]
	}

	// Convert to vectors
	vectors := make([]Vector, 0, len(finalCandidates))
	for _, id := range finalCandidates {
		vectors = append(vectors, g.Vectors[id])
	}

	return vectors, nil
}

// DistanceItem represents an item with its distance to the query vector
type DistanceItem struct {
	ID       string
	Distance float32
}

// MinHeap implementation for candidates (min distance first)
type MinHeap []DistanceItem

func (h MinHeap) Len() int            { return len(h) }
func (h MinHeap) Less(i, j int) bool  { return h[i].Distance < h[j].Distance }
func (h MinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(DistanceItem)) }
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// MaxHeap implementation for results (max distance first, for easy removal of worst element)
type MaxHeap []DistanceItem

func (h MaxHeap) Len() int            { return len(h) }
func (h MaxHeap) Less(i, j int) bool  { return h[i].Distance > h[j].Distance } // Note: > for max heap
func (h MaxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MaxHeap) Push(x interface{}) { *h = append(*h, x.(DistanceItem)) }
func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

/*
searchLayer searches for the nearest neighbors in a specific layer using heap data structures
for better performance with large k or EfConstruction values
*/
func (g *HNSWGraph) searchLayer(query []float32, entryPoint string, k int, layer int) []string {
	// Early return for invalid k
	if k <= 0 {
		return []string{}
	}

	// Initialize visited set and result/candidate heaps
	visited := make(map[string]bool)
	resultSet := &MaxHeap{} // Max heap for results (worst at top for easy removal)
	heap.Init(resultSet)

	// Initialize with entry point
	entryPointDist := g.Distance(query, g.Vectors[entryPoint].Data)
	heap.Push(resultSet, DistanceItem{ID: entryPoint, Distance: entryPointDist})
	visited[entryPoint] = true

	// Min heap for candidates to visit next (best at top)
	candidateSet := &MinHeap{DistanceItem{ID: entryPoint, Distance: entryPointDist}}
	heap.Init(candidateSet)

	// We'll use a dynamic list size based on ef parameter:
	// For construction (efConstruction) or search (efSearch)
	// This improves exploration during search
	ef := k
	if layer == 0 && g.EfSearch > k {
		// Use larger ef for bottom layer search
		ef = g.EfSearch
	} else if layer > 0 && g.EfConstruction > k {
		// Use efConstruction for upper layers during insertion
		ef = g.EfConstruction
	}

	// Use a higher quality threshold for early stopping to ensure better exploration
	qualityThreshold := float32(1.1) // Allow 10% worse candidates to be explored before stopping

	// Continue until we've explored all viable candidates
	for candidateSet.Len() > 0 {
		// Get closest candidate
		current := heap.Pop(candidateSet).(DistanceItem)

		// If the results heap is full and the current candidate is significantly worse than the worst result,
		// we can stop (apply quality threshold to avoid early stopping)
		if resultSet.Len() >= ef && current.Distance > (*resultSet)[0].Distance*qualityThreshold {
			break
		}

		// Explore neighbors of the current candidate
		for _, neighborID := range g.Layers[layer][current.ID] {
			if !visited[neighborID] {
				visited[neighborID] = true

				neighborDist := g.Distance(query, g.Vectors[neighborID].Data)

				// If the results heap is not full or the neighbor is better than the worst result,
				// add it to the result set
				if resultSet.Len() < ef || neighborDist < (*resultSet)[0].Distance {
					// Add to result set
					heap.Push(resultSet, DistanceItem{ID: neighborID, Distance: neighborDist})

					// If result set is too large, remove the worst element
					if resultSet.Len() > ef {
						heap.Pop(resultSet)
					}

					// Always add to candidate set for further exploration, regardless of distance
					// This improves the chance of finding better paths through the graph
					heap.Push(candidateSet, DistanceItem{ID: neighborID, Distance: neighborDist})
				}
			}
		}
	}

	// Convert heap to sorted list of IDs (limit to k results)
	resultIDs := make([]string, 0, k)
	resultItems := make([]DistanceItem, 0, resultSet.Len())

	// Extract all items from the heap
	for resultSet.Len() > 0 {
		resultItems = append(resultItems, heap.Pop(resultSet).(DistanceItem))
	}

	// Sort by distance (ascending)
	sort.Slice(resultItems, func(i, j int) bool {
		return resultItems[i].Distance < resultItems[j].Distance
	})

	// Take top k
	for i := 0; i < k && i < len(resultItems); i++ {
		resultIDs = append(resultIDs, resultItems[i].ID)
	}

	return resultIDs
}

/*
selectNeighbors selects the M nearest neighbors from a set of candidates
using the heuristic selection algorithm from the original HNSW paper
*/
func (g *HNSWGraph) selectNeighbors(query []float32, candidates []string, m int) []string {
	if len(candidates) <= m {
		return candidates
	}

	// First, sort candidates by distance
	type candidateItem struct {
		ID       string
		Distance float32
	}

	items := make([]candidateItem, 0, len(candidates))
	for _, id := range candidates {
		items = append(items, candidateItem{
			ID:       id,
			Distance: g.Distance(query, g.Vectors[id].Data),
		})
	}

	// Sort by distance
	sort.Slice(items, func(i, j int) bool {
		return items[i].Distance < items[j].Distance
	})

	// Select neighbors using heuristic selection
	// This improves the diversity of connections and prevents "dead ends"
	result := make([]string, 0, m)

	// Always include the closest neighbor
	if len(items) > 0 {
		result = append(result, items[0].ID)
		items = items[1:] // Remove the closest neighbor from candidates
	}

	// For the remaining slots, use a heuristic selection
	for len(result) < m && len(items) > 0 {
		// Find the candidate with the maximum distance to any point in result
		maxDist := float32(-1.0)
		maxIdx := 0

		for i, item := range items {
			// Find minimum distance to any point in result
			minDist := float32(math.MaxFloat32)
			for _, resultID := range result {
				dist := g.Distance(g.Vectors[item.ID].Data, g.Vectors[resultID].Data)
				if dist < minDist {
					minDist = dist
				}
			}

			// Update max if this is better
			if minDist > maxDist {
				maxDist = minDist
				maxIdx = i
			}
		}

		// Add the selected candidate to result
		result = append(result, items[maxIdx].ID)

		// Remove the selected candidate from items
		items = append(items[:maxIdx], items[maxIdx+1:]...)
	}

	return result
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

	// Check for zero vectors
	if normA == 0 || normB == 0 {
		return 1.0 // Maximum distance for zero vectors
	}

	sqrtNormA := float32(math.Sqrt(float64(normA)))
	sqrtNormB := float32(math.Sqrt(float64(normB)))

	// Avoid division by zero for very small norms
	if sqrtNormA == 0 || sqrtNormB == 0 {
		return 1.0
	}

	similarity := dot / (sqrtNormA * sqrtNormB)

	// Clamp similarity to [-1, 1] due to floating point precision
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}

	return 1.0 - similarity // Distance = 1 - similarity
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
