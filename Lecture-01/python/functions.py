## Warning: the functions here were automatically translated from R and not tested
## If you find bugs, please fix them and send a corrected version to j.d.karch@fsw.leidenuniv.nl,
## including a short summary of what you had to change

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def make_ex_1(beta=0.1, n_train=50):
    n_reps = 100
    shrinkage = np.arange(0, 1.1, 0.1)
    mse = np.zeros((n_reps, len(shrinkage)))
    beta_hats = np.zeros((n_reps, len(shrinkage)))
    np.random.seed(123)
    for i in range(n_reps):
        # generate training data
        x = np.random.uniform(-3, 3, n_train)
        y = beta * x + np.random.normal(0, 1, n_train)

        # fit OLS and get parameter estimates
        fit = LinearRegression(fit_intercept=False).fit(x.reshape(-1, 1), y)
        b_ols = fit.coef_[0]

        # generate test data:
        xtest = np.random.uniform(-3, 3, 10000)
        ytest = beta * xtest + np.random.normal(0, 1, 10000)

        # apply shrinkage and obtain predictions
        for s in range(len(shrinkage)):
            # generate predictions for test observations
            ypred = xtest * shrinkage[s] * b_ols
            mse[i, s] = np.mean((ytest - ypred)**2)
            beta_hats[i, s] = shrinkage[s] * b_ols

    min_id = np.argmin(np.mean(mse, axis=0))

    # Plot MSE versus shrinkage
    plt.subplot(1, 2, 1)
    plt.plot(shrinkage, np.mean(mse, axis=0), 'r', label='test MSE')
    plt.xlabel('shrinkage')
    plt.ylabel('test MSE')
    plt.title(f'beta = {beta}; N = {n_train};\n optimal shrinkage factor: {shrinkage[min_id]}')

    # Plot distributions of beta estimates (add line for true value)
    plt.subplot(1, 2, 2)
    plt.boxplot(beta_hats)
    plt.xlabel('shrinkage')
    plt.ylabel('beta_hat')
    plt.plot([1, 11], [beta, beta], 'r')

    # Compute variance of the estimates
    var_est = np.round(np.var(beta_hats, axis=0), decimals=3)

    return shrinkage[min_id], var_est
