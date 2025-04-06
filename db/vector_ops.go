package db

import (
	"errors"
	"math"
)

/*
VectorAdd adds two vectors element-wise

Parameters:
a, b: []float32 - The vectors to add
*/

func VectorAdd(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, errors.New("vector dimensions do not match")
	}

	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result, nil
}

/*
VectorSubtract subtracts vector b from vector a element-wise

Parameters:
a, b: []float32 - The vectors to subtract
*/

func VectorSubtract(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, errors.New("vector dimensions do not match")
	}

	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result, nil
}

/*
NormalizeVector normalizes a vector to unit length

Parameters:
v: []float32 - The vector to normalize
*/

func NormalizeVector(v []float32) {
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

/*
CosineSimilarity calculates the cosine similarity between two vectors

Returns:
float32 - A value between -1 and 1, where 1 means identical direction,
0 means orthogonal, and -1 means opposite directions
*/

func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector dimensions do not match")
	}

	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0, nil
	}

	similarity := dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))

	// Correct for floating point precision issues
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}

	return similarity, nil
}

/*
DotProduct computes the dot product of two vectors

Parameters:
a, b: []float32 - The vectors to compute the dot product of
*/

func DotProduct(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector dimensions do not match")
	}

	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}

	return dot, nil
}

/*
VectorMagnitude computes the magnitude (length) of a vector

Parameters:
v: []float32 - The vector to compute the magnitude of
*/

func VectorMagnitude(v []float32) float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return float32(math.Sqrt(float64(sum)))
}

/*
ScalarMultiply multiplies a vector by a scalar value

Parameters:
v: []float32 - The vector to multiply
scalar: float32 - The scalar value to multiply the vector by
*/

func ScalarMultiply(v []float32, scalar float32) []float32 {
	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = val * scalar
	}
	return result
}
