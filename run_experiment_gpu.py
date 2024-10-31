import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import pandas as pd
import pickle
from functools import partial

# Generate points on a sphere
def generate_points_on_sphere(key, n, d):
    points = jax.random.normal(key, shape=(n, d))
    norms = jnp.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

# Gaussian kernel matrix
def gauss_kernel_matrix(X1, X2, tau):
    sq_dists = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-sq_dists / (tau ** 2))

def reg_m(m, d):
    return 0  # Adjust if regularization is needed

def current_tau_m(m, d, alpha=0.01):
    return m ** (-(1 / alpha) * 1 / (d - 1))

@jit
def mean_squared_error(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

def compute_prediction(X_train, X_test, y_train, y_test, tau, reg):
    K = gauss_kernel_matrix(X_train, X_train, tau)
    f = gauss_kernel_matrix(X_train, X_test, tau)
    inv = K + reg * jnp.eye(K.shape[0])
    coeffs = jax.scipy.linalg.solve(inv, y_train, assume_a='pos')
    y_pred_test = jnp.dot(f.T, coeffs)
    test_error = mean_squared_error(y_test, y_pred_test)
    return y_pred_test, test_error

def test_sphere(key, ms, d, tau_m, noise_level=0.5, alpha=1, index=0):
    test_errors = []
    for m in ms:
        key, subkey = jax.random.split(key)
        test_size = int(0.5 * m)

        X_train = generate_points_on_sphere(subkey, m, d)
        key, subkey = jax.random.split(key)
        X_test = generate_points_on_sphere(subkey, test_size, d)

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=(m,)) * jnp.sqrt(noise_level)
        y_train = 10 * jnp.ones(m) + noise

        key, subkey = jax.random.split(key)
        test_noise = jax.random.normal(subkey, shape=(test_size,)) * jnp.sqrt(noise_level)
        y_test = 10 * jnp.ones(test_size) + test_noise

        tau = tau_m(m, d, alpha)
        reg = reg_m(m, d)
        y_pred_test, test_error = compute_prediction(X_train, X_test, y_train, y_test, tau, reg)

        test_errors.append(test_error.item())

    result = {'m': ms, 'test_error': test_errors}
    return result

def main():
    number_of_runs = 100
    ds = [6, 4, 6]
    noise_levels = [1, 10, 10000]
    alphas = [0.01, 1000, 1]
    sequence = [100 * i for i in range(1, 10)] + [500 * i for i in range(2, 10)] + [1000 * i for i in range(5, 12)]
    ms_sequences = [sequence for _ in ds]

    for experiment_number, d in enumerate(ds):
        noise_level = noise_levels[experiment_number]
        alpha = alphas[experiment_number]
        ms = ms_sequences[experiment_number]

        aggregated_results = []
        for index in range(number_of_runs):
            key = jax.random.PRNGKey(index)
            result = test_sphere(key, ms, d, current_tau_m, noise_level, alpha, index)
            aggregated_results.append(result)
            print(f"Experiment {experiment_number}, Iteration {index} completed.")

        dataframes = [pd.DataFrame(d) for d in aggregated_results]
        results = pd.concat(dataframes, ignore_index=True)
        path = 'results_' + str(noise_level) + "_" + str(d) + "_" + str(max(ms)) + "_" + str(
            alpha) + "_aggregated" + '.pkl'
        print(f"the path is {path}")
        with open(path, 'wb') as f:
            pickle.dump(results, f)
