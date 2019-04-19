const test = require('ava')

const ml = require('ml-regression')
const SLR = ml.SLR // Simple Linear Regression

const trainingSetExamples = [
  [300, 0],
  [280, 0.1],
  [230, 0.2],
  [220, 0.3],
  [220, 0.4],
  [160, 0.5],
  [140, 0.6],
  [120, 0.7],
  [100, 0.8],
  [50, 0.9],
  [25, 1],
  [350, 0.05],
  [330, 0.1],
  [320, 0.2],
  [300, 0.3],
  [280, 0.4]
]

let regressionModel
test.beforeEach(() => {
  regressionModel = new SLR(trainingSetExamples.map(e => e[0]), trainingSetExamples.map(e => e[1]))
})

test('> 0.5 for 50', t => {
  const prediction = regressionModel.predict(parseFloat(50))
  console.log({ prediction })
  t.true(prediction > 0.5)
})
test('> 0.5 for 100', t => {
  const prediction = regressionModel.predict(parseFloat(100))
  console.log({ prediction })
  t.true(prediction > 0.5)
})
test('< 0.5 for 250', t => {
  const prediction = regressionModel.predict(parseFloat(250))
  console.log({ prediction })
  t.true(prediction < 0.5)
})
test('< 0.5 for 200', t => {
  const prediction = regressionModel.predict(parseFloat(200))
  console.log({ prediction })
  t.true(prediction < 0.5)
})
