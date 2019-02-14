package main

import (
	nn "GoNNEr/neuralnetwork"
	//"GoNNEr/util"
	"fmt"
)

func main() {

	/*
		andGate := nn.Perceptron{[]float64{2., 2.}, -3.}
		orGate := nn.Perceptron{[]float64{2., 2.}, -1.}
		notGate := nn.Perceptron{[]float64{-2.}, 1.}

		fmt.Println(andGate.Step([]float64{0., 1.}))
		fmt.Println(orGate.Step([]float64{0., 1.}))
		fmt.Println(notGate.Step([]float64{0.}))
	*/
	var l, l2 nn.NeuralNetworkLayer
	l = nn.CreateNeuralNetworkLayer(1, 2, 2)
	l.SetWeightsBias([][]float64{[]float64{2, 2}, []float64{-1, -1}}, []float64{-1, 1.5})
	l2 = nn.CreateNeuralNetworkLayer(1, 1, 2)
	l2.SetWeightsBias([][]float64{[]float64{1, 1}}, []float64{-1.5})
	var n nn.NeuralNetwork
	n.Layers = []nn.NeuralNetworkLayer{l, l2}

	fmt.Println()

	for x := 0.; x <= 1; x++ {
		for y := 0.; y <= 1; y++ {
			o1 := n.Layers[0].Output([][]float64{[]float64{x, y}, []float64{x, y}}, 1)
			o2 := n.Layers[1].Output(o1, 1)
			fmt.Printf("XOR(%.0f, %.0f) = %.0f\n", x, y, o2[0][0])
		}
	}

}
