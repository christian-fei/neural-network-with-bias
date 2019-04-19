
function performRegression (X = [], y = [], x = 0) {
  regressionModel.predict(parseFloat(x))
  console.log(regressionModel.toString(3))
}

module.exports = { performRegression }
