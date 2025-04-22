GenerateDataSDS <- function(seed) {
  set.seed(seed)
  nCases <- 10^4
  nNoiseVar <- 200

  # generate X
  x1 <- runif(nCases, -2, 2)
  x2 <- rnorm(nCases, mean = 4, sd = sqrt(2))
  x3 <- rnorm(nCases, mean = 0, sd = 1)
  xnoise <- matrix(runif(nCases * nNoiseVar, -2, 2), nCases, nNoiseVar)

  # generate Y
  eta <- -1.5 + (1.5 * x1) + (0.85 * x1^2) - (0.20 * x1^3) + (2.5 * I(x2 < 0)) + I(x2 > 3) + (0.3 * x3)
  pi <- exp(eta) / (1 + exp(eta))
  y <- rbinom(nCases, 1, pi)

  # combine data and split into training and test data
  full <- data.frame(y, x1, x2, x3, xnoise)
  names(full) <- c("Y", paste("X", 1:(nNoiseVar + 3), sep = ""))

  return(full)
}