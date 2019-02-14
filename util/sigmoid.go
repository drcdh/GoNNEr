package util

import (
	"math"
)

func Sigmoid(sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}
