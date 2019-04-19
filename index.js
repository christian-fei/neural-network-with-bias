#!/usr/bin/env node

function uniformDistribution (min, max) {
  if (min > max) throw new Error('Minimum value cannot be greater than maximum value')

  if (min === max) return Infinity

  return 1 / (max - min)
}

function calcSumOfWeightedInputs (neuronInputs, weights) {
  if (neuronInputs[0] === weights[0] === undefined) throw new Error('neuronInputs and weights must have the same shape', { neuronInputs, weights })
  return neuronInputs.reduce((acc, neuronInput, index) => {
    return acc + weights[index] * neuronInput
  }, 0)
}

function sigmoid (sumOfWeightedInputs) {
  return 1 / (1 + Math.exp(-sumOfWeightedInputs))
}

function sigmoidGradient (neuronOutput) {
  return neuronOutput * (1 - neuronOutput)
}

function normalize (array = []) {
  const min = Math.min(...array)
  const max = Math.max(...array)
  return array.map(a => a * a / max * min)
}

function NeuralNetwork () {
  if (!(this instanceof NeuralNetwork)) { return new NeuralNetwork() }
  const self = this

  this.bias = uniformDistribution(-1, 1)

  this.predict = function (neuronInputs) {
    let sumOfWeightedInputs = calcSumOfWeightedInputs(neuronInputs, self.weights)
    let neuronOutput = sigmoid(sumOfWeightedInputs + self.bias)
    return neuronOutput
  }

  this.train = function (trainingSetExamples, { numberOfIterations }) {
    self.weights = Array.from({ length: trainingSetExamples[0].inputs.length }).map(_ => uniformDistribution(-1, 1))
    // self.weights = trainingSetExamples[0].inputs

    for (let iteration = 0; iteration < numberOfIterations; iteration++) {
      for (let trainingSetExample of trainingSetExamples) {
        let predictedOutput = self.predict(trainingSetExample.inputs)
        let errorInOutput = trainingSetExample.output - predictedOutput

        self.weights.reduce((acc, curr, index) => {
          let neuronInput = trainingSetExample.inputs[index]
          let adjustWeight = neuronInput * errorInOutput * sigmoidGradient(predictedOutput)
          // console.log({ neuronInput, index, adjustWeight })
          self.weights[index] += adjustWeight
          self.bias += errorInOutput * sigmoidGradient(predictedOutput)
        }, 0)
      }
    }
  }

  return this
}

if (require.main === module) {
  main()
    .then(() => {
      console.log('success')
      process.exit(0)
    })
    .catch(err => {
      console.error('failure')
      console.error(err)
      process.exit(1)
    })
} else {
  module.exports = NeuralNetwork
}

async function main () {
  const trainingSetExamples = [
    { 'inputs': [0, 0, 1], 'output': 0 },
    { 'inputs': [1, 1, 1], 'output': 1 },
    { 'inputs': [1, 0, 1], 'output': 1 },
    { 'inputs': [0, 1, 1], 'output': 0 }
  ]
  const numberOfIterations = 10000
  const nn = new NeuralNetwork()
  nn.train(trainingSetExamples, { numberOfIterations })

  const newSituation = [1, 1, 0]
  return {
    prediction: nn.predict(newSituation),
    nn
  }
}
