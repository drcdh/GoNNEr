package neuralnetwork

import (
//"GoNNEr/util"
//	"fmt"
)

type NeuralNetworkLayer struct {
	nRow    int
	nCol    int
	nCha    int
	neurons []Perceptron
}

func CreateNeuralNetworkLayer(nRow, nCol, nCha int) (layer NeuralNetworkLayer) { // TODO: Activation
	neurons := make([]Perceptron, nRow*nCol)
	for i := 0; i < len(neurons); i++ {
		neurons[i] = NewPerceptron(nCha)
	}
	layer = NeuralNetworkLayer{nRow, nCol, nCha, neurons}
	return layer
}

func (l *NeuralNetworkLayer) SetWeightsBias(weights [][]float64, biases []float64) {
	// Assert that the shapes are the same
	for i := 0; i < len(l.neurons); i++ {
		for c := 0; c < l.nCha; c++ {
			l.neurons[i].Weights[c] = weights[i][c]
		}
		l.neurons[i].Bias = biases[i]
	}
}

func (l *NeuralNetworkLayer) Output(input [][]float64, outSha int) (output [][]float64) {
	var o float64
	// Assert that input shape matchs output
	pOutput := make([]float64, len(l.neurons))
	for i := 0; i < len(l.neurons); i++ {
		o = l.neurons[i].Step(input[i])
		//o = l.neurons[i].Activate(input[i])
		pOutput[i] = o
	}
	for c := 0; c < outSha; c++ {
		output = append(output, pOutput)
	}
	return output
}

type NeuralNetwork struct {
	Layers []NeuralNetworkLayer
}

//func (nn NeuralNetwork) FeedForward(inputs [][]
