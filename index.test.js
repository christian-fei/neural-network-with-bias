const test = require('ava')

const NeuralNetwork = require('.')

const trainingSetExamples = [
  { 'inputs': [0, 0, 1], 'output': 0 },
  { 'inputs': [1, 1, 1], 'output': 1 },
  { 'inputs': [1, 0, 1], 'output': 1 },
  { 'inputs': [0, 1, 1], 'output': 0 }
]
const numberOfIterations = 1000

let nn
test.beforeEach(() => {
  nn = new NeuralNetwork()
  nn.train(trainingSetExamples, { numberOfIterations })
})

test('< 0.5 for [0, 0, 0]', t => {
  const newSituation = [0, 0, 0]
  let prediction = nn.predict(newSituation)
  t.true(prediction < 0.5)
})

test('> 0.5 for [1, 0, 0]', t => {
  const newSituation = [1, 0, 0]
  let prediction = nn.predict(newSituation)
  t.true(prediction > 0.5)
})

test('> 0.5 for [1, 1, 0]', t => {
  const newSituation = [1, 1, 0]
  let prediction = nn.predict(newSituation)
  t.true(prediction > 0.5)
})

test('> 0.5 for [1, 1, 1]', t => {
  const newSituation = [1, 1, 1]
  let prediction = nn.predict(newSituation)
  t.true(prediction > 0.5)
})

test('< 0.5 for [0, 1, 1]', t => {
  const newSituation = [0, 1, 1]
  let prediction = nn.predict(newSituation)
  t.true(prediction < 0.5)
})

test('< 0.5 for [0, 0, 1]', t => {
  const newSituation = [0, 0, 1]
  let prediction = nn.predict(newSituation)
  t.true(prediction < 0.5)
})
