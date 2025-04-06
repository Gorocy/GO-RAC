package db

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"vector-db/config"
)

func TestHNSW3DVisualization(t *testing.T) {
	// Initialize random seed
	rand.New(rand.NewSource(time.Now().UnixNano()))

	// Test different sizes
	sizes := []int{200, 300, 400, 500}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size%d", size), func(t *testing.T) {
			testHNSW3DVisualizationWithSize(t, size)
		})
	}
}

func testHNSW3DVisualizationWithSize(t *testing.T, numVectors int) {
	// Initialize HNSW graph with more layers for better visualization
	graph := NewHNSWGraph(8, 100, config.DistanceTypeEuclidean)

	// Generate random 2D vectors for visualization
	dimensions := 2
	vectors := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vector := make([]float32, dimensions)
		for j := 0; j < dimensions; j++ {
			vector[j] = rand.Float32() * 100 // Scale to 0-100 range
		}
		vectors[i] = Vector{
			ID:       fmt.Sprintf("%d", i),
			Data:     vector,
			Metadata: map[string]interface{}{"x": vector[0], "y": vector[1]},
		}
	}


	// Print graph structure
	fmt.Printf("\nGraph structure:\n")
	fmt.Printf("Entry point: %s\n", graph.EntryPoint)
	fmt.Printf("Max layer: %d\n", graph.MaxLayer)

	// Create query vector at random position
	query := []float32{
		rand.Float32() * 100,
		rand.Float32() * 100,
	}

	// Print visualization header
	fmt.Printf("\n=== HNSW 3D Search Visualization (Size: %d) ===\n", numVectors)
	fmt.Printf("Query point: (%.2f, %.2f)\n", query[0], query[1])

	// Visualize search process layer by layer
	visualize3DSearch(graph, vectors, query, 5)
}

func visualize3DSearch(graph *HNSWGraph, vectors []Vector, query []float32, k int) {
	const width = 30
	const height = 15
	const depth = 5 // Number of layers to visualize
	const scaleX = width / 100.0
	const scaleY = height / 100.0

	// Create 3D grid
	grid := make([][][]rune, depth)
	for z := range grid {
		grid[z] = make([][]rune, height)
		for y := range grid[z] {
			grid[z][y] = make([]rune, width)
			for x := range grid[z][y] {
				grid[z][y][x] = ' '
			}
		}
	}

	// Plot all vectors in each layer
	for z := 0; z < depth; z++ {
		for _, v := range vectors {
			x := int(v.Data[0] * scaleX)
			y := int(v.Data[1] * scaleY)
			if x >= 0 && x < width && y >= 0 && y < height {
				grid[z][y][x] = '.'
			}
		}
	}

	// Track search path for each layer
	searchPath := make([][]string, depth)
	for i := range searchPath {
		searchPath[i] = make([]string, 0)
	}

	// Perform search and track path
	current := graph.EntryPoint
	visited := make(map[string]bool)
	visited[current] = true

	// Track search process
	for z := graph.MaxLayer; z >= 0 && z < depth; z-- {
		// Get neighbors in current layer
		neighbors := graph.Layers[z][current]
		if neighbors == nil {
			continue
		}

		// Add current point to search path
		searchPath[z] = append(searchPath[z], current)

		// Find best neighbor
		bestNeighbor := current
		minDist := graph.Distance(query, graph.Vectors[current].Data)

		// Add all neighbors to search path
		for _, neighbor := range neighbors {
			if !visited[neighbor] {
				visited[neighbor] = true
				searchPath[z] = append(searchPath[z], neighbor)
				dist := graph.Distance(query, graph.Vectors[neighbor].Data)
				if dist < minDist {
					minDist = dist
					bestNeighbor = neighbor
				}
			}
		}

		// Move to best neighbor if it's better than current
		if bestNeighbor != current {
			current = bestNeighbor
		}
	}

	// Plot search path
	for z, path := range searchPath {
		if len(path) > 0 {
			// Plot current point (first in path)
			x := int(graph.Vectors[path[0]].Data[0] * scaleX)
			y := int(graph.Vectors[path[0]].Data[1] * scaleY)
			if x >= 0 && x < width && y >= 0 && y < height {
				grid[z][y][x] = 'O'
			}

			// Plot neighbors (rest of path)
			for _, point := range path[1:] {
				x := int(graph.Vectors[point].Data[0] * scaleX)
				y := int(graph.Vectors[point].Data[1] * scaleY)
				if x >= 0 && x < width && y >= 0 && y < height {
					grid[z][y][x] = '*'
				}
			}
		}
	}

	// Plot query point
	queryX := int(query[0] * scaleX)
	queryY := int(query[1] * scaleY)
	if queryX >= 0 && queryX < width && queryY >= 0 && queryY < height {
		for z := 0; z < depth; z++ {
			grid[z][queryY][queryX] = 'X'
		}
	}

	// Print 3D visualization
	fmt.Println("\n3D Visualization (Layer by Layer):")
	for z := depth - 1; z >= 0; z-- {
		fmt.Printf("\nLayer %d:\n", z)
		fmt.Println("Y")
		for y := height - 1; y >= 0; y-- {
			fmt.Printf("%2d|", y)
			for x := 0; x < width; x++ {
				fmt.Printf("%c", grid[z][y][x])
			}
			fmt.Println()
		}
		fmt.Println("  +" + string(make([]rune, width)))
		fmt.Println("   0" + string(make([]rune, width-2)) + "X")
	}

	// Print search statistics
	fmt.Println("\nSearch Statistics:")
	for z, path := range searchPath {
		if len(path) > 0 {
			fmt.Printf("Layer %d: Visited %d points\n", z, len(path))
			fmt.Printf("  Start: Vector %s (%.2f, %.2f)\n",
				path[0],
				graph.Vectors[path[0]].Data[0],
				graph.Vectors[path[0]].Data[1])
			if len(path) > 1 {
				fmt.Printf("  End: Vector %s (%.2f, %.2f)\n",
					path[len(path)-1],
					graph.Vectors[path[len(path)-1]].Data[0],
					graph.Vectors[path[len(path)-1]].Data[1])
			}
		}
	}

	fmt.Println("\nLegend:")
	fmt.Println("  . - Vector in database")
	fmt.Println("  O - Current search point")
	fmt.Println("  * - Neighbor being explored")
	fmt.Println("  X - Query point")
}
