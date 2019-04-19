#!/usr/bin/env node

function uniformDistribution (min, max) {
  if (min > max) throw new Error('Minimum value cannot be greater than maximum value')

  if (min === max) return Infinity

  return 1 / (max - min)
}

function calcSumOfWeightedInputs (neuronInputs, weights) {
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

function NeuralNetwork () {
  if (!(this instanceof NeuralNetwork)) { return new NeuralNetwork() }
  const self = this

  this.weights = [uniformDistribution(-1, 1), uniformDistribution(-1, 1), uniformDistribution(-1, 1)]

  this.bias = uniformDistribution(-1, 1)

  this.predict = function (neuronInputs) {
    let sumOfWeightedInputs = calcSumOfWeightedInputs(neuronInputs, self.weights)
    let neuronOutput = sigmoid(sumOfWeightedInputs + self.bias)
    return neuronOutput
  }

  this.train = function (trainingSetExamples, { numberOfIterations }) {
    for (let iteration = 0; iteration < numberOfIterations; iteration++) {
      for (let trainingSetExample of trainingSetExamples) {
        let predictedOutput = self.predict(trainingSetExample.inputs)
        // console.log('predictedOutput', predictedOutput)

        let errorInOutput = trainingSetExample['output'] - predictedOutput
        // console.log('errorInOutput', errorInOutput)

        self.weights.reduce((acc, curr, index) => {
          let neuronInput = trainingSetExample.inputs[index]
          let adjustWeight = neuronInput * errorInOutput * sigmoidGradient(predictedOutput)
          self.weights[index] += adjustWeight
          self.bias += errorInOutput * sigmoidGradient(predictedOutput)
        }, 0)
      }
    }
  }

  return this
}

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

async function main () {
  const trainingSetExamples = [
    { 'inputs': [0, 0, 1], 'output': 0 },
    { 'inputs': [1, 1, 1], 'output': 1 },
    { 'inputs': [1, 0, 1], 'output': 1 },
    { 'inputs': [0, 1, 1], 'output': 0 }
  ]
  const numberOfIterations = 100000
  const nn = new NeuralNetwork()
  nn.train(trainingSetExamples, { numberOfIterations })

  const newSituation = [1, 1, 0]
  let prediction = nn.predict(newSituation)
  console.log('prediction', prediction, newSituation)

  return nn
}
