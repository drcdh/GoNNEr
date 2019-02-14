package neuralnetwork

import (
	"GoNNEr/util"
)

type Perceptron struct {
	Weights []float64
	Bias    float64
}

func NewPerceptron(nChannels int) (p Perceptron) {
	p = Perceptron{Weights: make([]float64, nChannels)}
	return p
}

func (p *Perceptron) output(input []float64) (output float64) {
	// Check number of channels is consistent
	for i := 0; i < len(input); i++ {
		output += p.Weights[i] * input[i]
	}
	output += p.Bias
	return output
}

func (p *Perceptron) Step(input []float64) float64 {
	if p.output(input) >= 0 {
		return 1.
	}
	return 0.
}

func (p *Perceptron) Activate(input []float64) float64 {
	return util.Sigmoid(p.output(input))
}
