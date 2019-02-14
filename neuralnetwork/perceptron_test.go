package neuralnetwork

import "fmt"
import "math"
import "testing"

func TestStep(t *testing.T) {
	andGate := Perceptron{[]float64{2, 2}, -3}
	orGate := Perceptron{[]float64{2, 2}, -1}
	notGate := Perceptron{[]float64{-2}, 1}

	for x := 0.; x <= 1.; x++ {
		for y := 0.; y <= 1.; y++ {
			if andGate.Step([]float64{x, y}) != math.Min(x, y) {
				t.Error(fmt.Sprintf(`andGate.Step(%d, %d) incorrect`, x, y))
			}
			if orGate.Step([]float64{x, y}) != math.Max(x, y) {
				t.Error(fmt.Sprintf(`orGate.Step(%d, %d) incorrect`, x, y))
			}
		}
		if notGate.Step([]float64{x}) != 1.-x {
			t.Error(fmt.Sprintf(`notGate.Step(%d) incorrect`, x))
		}
	}
}
