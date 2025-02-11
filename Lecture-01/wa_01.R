library(tidyverse)
library(patchwork)

sineFunction <- function(n) {
  x <- runif(n, min=-3, max=3)
  y <- 8 * sin(x)+ rnorm(n)
  return(data.frame(x,y))
}

trainTest <- function(train_set, test_set, deg) {
  fit <- lm(y ~ poly(x, degree = deg), data = train_set)
  test_pred <- predict(fit, newdata=test_set)
  test_err <- mean((test_set$y - test_pred)^2)
  
  return(list(
    "data"=data.frame(
      x=test_set$x,
      y=test_set$y,
      yhat=test_pred
    ),
    "error"=test_err
    )
  )
}

calcTestError <- function(train_size) {
  train_set <- sineFunction(train_size)
  test_set <- sineFunction(10000)
  
  poly3 <- trainTest(train_set, test_set, 3)
  poly15 <- trainTest(train_set, test_set, 15)
  
  return(
    list(
      "data"=list(poly3$data, poly15$data), 
      "errors"=list(poly3$error, poly15$error)
    )
  )
}

train_50 <- calcTestError(50)
train_10k <- calcTestError(10^4)

tibble(
  degree=c("degree 3", "degree 15"),
  "n=50"=as.numeric(train_50$errors),
  "n=10^4"=as.numeric(train_10k$errors)
)

p1 <- ggplot(train_50$data[[1]], aes(x=x, y=y)) +
  geom_point(alpha=0.1) +
  geom_line(aes(y=yhat), color="red") +
  ggtitle("Degree 3, train n=50")

p2 <- ggplot(train_50$data[[2]], aes(x=x, y=y)) +
  geom_point(alpha=0.1) +
  geom_line(aes(y=yhat), color="red") +
  ggtitle("Degree 15, train n=50")

p3 <- ggplot(train_10k$data[[1]], aes(x=x, y=y)) +
  geom_point(alpha=0.1) +
  geom_line(aes(y=yhat), color="red") +
  ggtitle("Degree 3, train n=10,000")

p4 <- ggplot(train_10k$data[[2]], aes(x=x, y=y)) +
  geom_point(alpha=0.1) +
  geom_line(aes(y=yhat), color="red") +
  ggtitle("Degree 15, train n=10,000")

(p1 + p2) / (p3 + p4)


train_set <- sineFunction(10000)
test_set <- sineFunction(10000)

poly3 <- trainTest(train_set, test_set, 3)
poly15 <- trainTest(train_set, test_set, 15)

yhat_sine <- 8*sin(test_set$x)
sine_mse <- mean((test_set$y - yhat_sine)^2)

tibble(
  degree=c("degree 3", "degree 15", "sine"),
  "MSE"=c(poly3$error, poly15$error, sine_mse)
)

