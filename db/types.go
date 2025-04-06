package db

/*
Vector represents a vector in the database
*/
type Vector struct {
	ID       string                 `json:"id"`
	Data     []float32              `json:"data"`
	Metadata map[string]interface{} `json:"metadata"`
}
