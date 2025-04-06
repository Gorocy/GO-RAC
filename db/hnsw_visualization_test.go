package db

import (
	"fmt"
	"math"
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

	// Generate random 2D vectors for visualization with better distribution
	dimensions := 2
	vectors := make([]Vector, numVectors)

	// Find min/max coordinates to track bounds
	var minX, minY float32 = 100, 100
	var maxX, maxY float32 = 0, 0

	// First pass - generate vectors and track bounds
	for i := 0; i < numVectors; i++ {
		vector := make([]float32, dimensions)
		for j := 0; j < dimensions; j++ {
			vector[j] = rand.Float32() * 100 // Scale to 0-100 range
		}

		// Track bounds
		if vector[0] < minX {
			minX = vector[0]
		}
		if vector[0] > maxX {
			maxX = vector[0]
		}
		if vector[1] < minY {
			minY = vector[1]
		}
		if vector[1] > maxY {
			maxY = vector[1]
		}

		vectors[i] = Vector{
			ID:       fmt.Sprintf("%d", i),
			Data:     vector,
			Metadata: map[string]interface{}{"x": vector[0], "y": vector[1]},
		}
	}

	// Create query vector at random position within the bounds
	query := []float32{
		minX + rand.Float32()*(maxX-minX),
		minY + rand.Float32()*(maxY-minY),
	}

	// Insert vectors
	for _, vector := range vectors {
		err := graph.Insert(vector)
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Search for nearest neighbors
	k := 5
	results, err := graph.Search(query, k)
	if err != nil {
		t.Errorf("Search failed: %v", err)
	}

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

	// Find min/max coordinates for proper scaling
	var minX, minY float32 = float32(math.MaxFloat32), float32(math.MaxFloat32)
	var maxX, maxY float32 = float32(math.SmallestNonzeroFloat32), float32(math.SmallestNonzeroFloat32)

	for _, v := range vectors {
		if v.Data[0] < minX {
			minX = v.Data[0]
		}
		if v.Data[0] > maxX {
			maxX = v.Data[0]
		}
		if v.Data[1] < minY {
			minY = v.Data[1]
		}
		if v.Data[1] > maxY {
			maxY = v.Data[1]
		}
	}

	// Include query point in bounds
	if query[0] < minX {
		minX = query[0]
	}
	if query[0] > maxX {
		maxX = query[0]
	}
	if query[1] < minY {
		minY = query[1]
	}
	if query[1] > maxY {
		maxY = query[1]
	}

	// Add padding to bounds
	padding := float32(0.05) // 5% padding
	rangeX := maxX - minX
	rangeY := maxY - minY
	minX -= padding * rangeX
	maxX += padding * rangeX
	minY -= padding * rangeY
	maxY += padding * rangeY

	// Create scaling factors
	scaleX := float32(width-1) / (maxX - minX)
	scaleY := float32(height-1) / (maxY - minY)

	// Create empty grid
	grid := make([][]rune, height)
	for i := range grid {
		grid[i] = make([]rune, width)
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}

	// Transform function to map coordinates to grid
	transform := func(x, y float32) (int, int) {
		gridX := int((x - minX) * scaleX)
		gridY := int((y - minY) * scaleY)

		// Constrain to grid boundaries
		if gridX < 0 {
			gridX = 0
		}
		if gridX >= width {
			gridX = width - 1
		}
		if gridY < 0 {
			gridY = 0
		}
		if gridY >= height {
			gridY = height - 1
		}

		return gridX, gridY
	}

	// Plot all vectors
	for _, v := range vectors {
		x, y := transform(v.Data[0], v.Data[1])
		grid[y][x] = '.'
	}

	// Get query point coordinates
	queryX, queryY := transform(query[0], query[1])

	// Highlight results with priority
	resultMap := make(map[string]bool)
	for _, r := range results {
		resultMap[r.ID] = true
	}

	// Plot results with a larger marker to ensure visibility
	for _, v := range results {
		x := int(v.Data[0] * scaleX)
		y := int(v.Data[1] * scaleY)
		if x >= 0 && x < width && y >= 0 && y < height {
			grid[y][x] = 'O'
		}
	}

	// Plot query
	queryX = int(query[0] * scaleX)
	queryY = int(query[1] * scaleY)
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

	// Print x-axis
	fmt.Print("  +")
	for x := 0; x < width; x++ {
		fmt.Print("-")
	}
	fmt.Println()

	// Print axis labels
	fmt.Printf("   %-*d%*d X\n", width/2, 0, width/2, width)

	fmt.Println("Legend:")
	fmt.Println("  . - Vector in database")
	fmt.Println("  O - Nearest neighbor")
	fmt.Println("  X - Query point")

	// Print coordinate ranges for reference
	fmt.Printf("X range: %.2f - %.2f\n", minX, maxX)
	fmt.Printf("Y range: %.2f - %.2f\n", minY, maxY)
}
