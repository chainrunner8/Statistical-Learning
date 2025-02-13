
library(tidyverse)
library(ggplot2)

set.seed(42)  # set random seed for reproducibility

# VARIABLES

n <- 100
training_pct <- 0.8
ma_window <- 5


# FUNCTIONS

generate_data <- function(n, type) {
  x <- runif(n, min=0, max=100)  # generate x-values randomly
  if (type == "parabola") {
    y <- 3 + x * sqrt(x) / 100 + rnorm(n, mean=0, sd=0.5)  # define a noisy parabolic relationship
  } else if (type == "sine"){
    y <- 4.18 + 3.6 * sin(x / 12.7) + rnorm(n, mean=0, sd=0.5)
  } else if (type == 'sinh'){
    y <- 12 + 3 * sin((x + 14) / 9.5) + sinh(-x/10 + 5) + rnorm(n, mean=0, sd=5)
  }
  return(data.frame(x, y))
}


moving_average <- function(x, y, window) {
  y_hat <- sapply(x, function(xi) {
    neighbours <- abs(x - xi) <= window
    mean(y[neighbours])
  })
  return(y_hat)
}


split_data <- function(data) {
  train_indices <- sample(1:n, size=training_pct*n)
  train_test_sets <- list(
    train=data.frame(
      x=data$x[train_indices],
      y=data$y[train_indices]
    ) %>% arrange(x),
    test=data.frame(
      x=data$x[-train_indices],
      y=data$y[-train_indices]
    ) %>% arrange(x)
  )
  return(train_test_sets)
}


train_and_test <- function(train_test_sets, model_type, h) {
  train_x <- train_test_sets$train$x
  train_y <- train_test_sets$train$y
  test_x <- train_test_sets$test$x
  test_y <- train_test_sets$test$y
  if (model_type == "lm") {
    model <- lm(y ~ x, data=data.frame(x=train_x, y=train_y))
    yhat.Train <- predict(model)
    yhat.Test <- predict(model, newdata=data.frame(x=test_x))
  } else if (model_type == "spline") {
    model <- smooth.spline(x=train_x, y=train_y)
    yhat.Train <- predict(model)$y
    yhat.Test <- predict(model, x=test_x)$y
  } else if (model_type == "nna") {
    yhat.Train <- moving_average(train_x, train_y, window=h)
    yhat.Test <- moving_average(test_x, test_y, window=h)
  } else {
    stop("Unknown model type")
  }
  return(list(train=yhat.Train, test=yhat.Test))
}


calculate_preds <- function(data) {
  models <- list(
    list(model_type="lm", h=NULL),
    list(model_type="spline", h=NULL),
    list(model_type="nna", h=5)
  )
  predictions <- lapply(models, function(params) {
    train_and_test(
      data,
      params$model_type,
      params$h
    )}
  )
  return(predictions)
}


# challenge: how to store the data for each distro?
# training and test sets are of different lengths.
create_dfs <- function(data, predictions) {
  train_results <- data.frame(
    x=data$train$x,
    y=data$train$y,
    yhat_lm=predictions[[1]]$train,
    yhat_spline=predictions[[2]]$train,
    yhat_nna=predictions[[3]]$train
  )
  test_results <- data.frame(
    x=data$test$x,
    y=data$test$y,
    yhat_lm=predictions[[1]]$test,
    yhat_spline=predictions[[2]]$test,
    yhat_nna=predictions[[3]]$test
  )
  return(list(train=train_results, test=test_results))
}


calculate_mse <- function(data) {
  train_mse <- data.frame(
    x=c(2, 6, 30),
    y=c(
      mean((data$train$y - data$train$yhat_lm)^2),
      mean((data$train$y - data$train$yhat_spline)^2),
      mean((data$train$y - data$train$yhat_nna)^2)
    )
  )
  test_mse <- data.frame(
    x=c(2, 6, 30),
    y=c(
      mean((data$test$y - data$test$yhat_lm)^2),
      mean((data$test$y - data$test$yhat_spline)^2),
      mean((data$test$y - data$test$yhat_nna)^2)
    )
  )
  return(list(train=train_mse, test=test_mse))
}


plot_predictions <- function(df, dataset_name) {
  ggplot(df$train, aes(x=x, y=y)) +
    geom_point(shape=21, fill="white", color="black", stroke=0.8, size=3) +
    geom_line(aes(y=yhat_lm, color="Linear Model"), linewidth=1) +
    geom_line(aes(y=yhat_spline, color="Spline"), linewidth=1) +
    geom_line(aes(y=yhat_nna, color="Moving Avg"), linewidth=1) +
    scale_color_manual(values=c("Linear Model"="orange", "Spline"="cyan", "Moving Avg"="green")) +
    labs(title=paste("Predictions on", dataset_name, "Data"),
         x="x", y="y",
         color="Model") +
    theme_minimal()
}


plot_mse <- function(mse_df, dataset_name) {
  mse_long <- rbind(
    data.frame(mse_df$train, set="Train"),
    data.frame(mse_df$test, set="Test")
  )
  ggplot(mse_long, aes(x=x, y=y, color=set)) +
    geom_line(linewidth=1, aes(linetype=set)) +
    geom_point(size=4, shape=22, fill="white", stroke=0.8) +
    scale_color_manual(values=c("Train"="blue", "Test"="red")) +
    labs(title=paste("MSE vs. Flexibility for", dataset_name),
         x="Flexibility", y="MSE (log10 scale)", color="Dataset") +
    scale_y_log10() +
    theme_minimal()
}


# PROGRAMME

data_parabola <- generate_data(n, type="parabola")
data_sine <- generate_data(n, type="sine")
data_sinh <- generate_data(n, type="sinh")

data_parabola.split <- split_data(data_parabola)
data_sine.split <- split_data(data_sine)
data_sinh.split <- split_data(data_sinh)

# now train every model on each data set

yhat_parabola <- calculate_preds(data_parabola.split)
yhat_sine <- calculate_preds(data_sine.split)
yhat_sinh <- calculate_preds(data_sinh.split)

df_parabola <- create_dfs(data_parabola.split, yhat_parabola)
df_sine <- create_dfs(data_sine.split, yhat_sine)
df_sinh <- create_dfs(data_sinh.split, yhat_sinh)

# then calculate the MSE for both training and test, for all 3 models every time

mse_parabola <- calculate_mse(df_parabola)
mse_sine <- calculate_mse(df_sine)
mse_sinh <- calculate_mse(df_sinh)

dev.off()

# plot_predictions(df_parabola, "Parabola")
# plot_mse(mse_parabola, "Parabola")
# plot_predictions(df_sine, "Sine")
# plot_mse(mse_sine, "Sine")
# plot_predictions(df_sinh, "Sinh")
plot_mse(mse_sinh, "Sinh")
