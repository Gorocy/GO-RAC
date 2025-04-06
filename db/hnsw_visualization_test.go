package db

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
	"vector-db/config"
)

func TestHNSWVisualization(t *testing.T) {
	// Initialize random seed
	rand.New(rand.NewSource(time.Now().UnixNano()))

	// Test different sizes
	sizes := [][]int{{100, 10}, {200, 20}, {300, 30}, {400, 40}, {500, 50}}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size%d", size[0]), func(t *testing.T) {
			testHNSWVisualizationWithSize(t, size[0], size[1])
		})
	}
}

func testHNSWVisualizationWithSize(t *testing.T, numVectors int, height int) {
	// Initialize HNSW graph
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

	// Insert vectors
	for _, vector := range vectors {
		graph.Insert(vector)
	}

	// Create query vector at random position
	query := []float32{
		rand.Float32() * 100,
		rand.Float32() * 100,
	}

	// Search for nearest neighbors
	k := 5
	results := graph.Search(query, k)

	// Print visualization
	fmt.Printf("\n=== HNSW Search Visualization (Size: %d) ===\n", numVectors)
	fmt.Printf("Query point: (%.2f, %.2f)\n", query[0], query[1])
	fmt.Println("Found nearest neighbors:")
	for i, result := range results {
		x := result.Data[0]
		y := result.Data[1]
		distance := graph.Distance(query, result.Data)
		fmt.Printf("%d. Vector %s: (%.2f, %.2f) - Distance: %.2f\n",
			i+1, result.ID, x, y, distance)
	}

	// Generate ASCII visualization
	fmt.Println("\nASCII Visualization:")
	visualize2D(vectors, results, query, height)
}

func visualize2D(vectors []Vector, results []Vector, query []float32, height int) {
	width := 2 * height
	scaleX := float32(width) / 100.0
	scaleY := float32(height) / 100.0

	// Create empty grid
	grid := make([][]rune, height)
	for i := range grid {
		grid[i] = make([]rune, width)
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}

	// Plot all vectors
	for _, v := range vectors {
		x := int(v.Data[0] * scaleX)
		y := int(v.Data[1] * scaleY)
		if x >= 0 && x < width && y >= 0 && y < height {
			grid[y][x] = '.'
		}
	}

	// Plot results
	for _, v := range results {
		x := int(v.Data[0] * scaleX)
		y := int(v.Data[1] * scaleY)
		if x >= 0 && x < width && y >= 0 && y < height {
			grid[y][x] = 'O'
		}
	}

	// Plot query
	queryX := int(query[0] * scaleX)
	queryY := int(query[1] * scaleY)
	if queryX >= 0 && queryX < width && queryY >= 0 && queryY < height {
		grid[queryY][queryX] = 'X'
	}

	// Print grid
	fmt.Println("Y")
	for y := height - 1; y >= 0; y-- {
		fmt.Printf("%2d|", y)
		for x := 0; x < width; x++ {
			fmt.Printf("%c", grid[y][x])
		}
		fmt.Println()
	}
	fmt.Println("  +" + string(make([]rune, width)))
	fmt.Println("   0" + string(make([]rune, width-2)) + "X")
	fmt.Println("Legend:")
	fmt.Println("  . - Vector in database")
	fmt.Println("  O - Nearest neighbor")
	fmt.Println("  X - Query point")
}
