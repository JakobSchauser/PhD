"""
Minimal ridge-regression probe for predicting target RGB from pre RGB.
Measures linear decodability of target information from pre-SIREN features.
"""

import numpy as np


def normalize_float(img):
    """Normalize image to float in [0, 1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def make_pixel_dataset(imgs, add_coords=True):
    """Flatten images to a pixel-level design matrix."""
    n, h, w, c = imgs.shape
    rgb_flat = imgs.reshape(n * h * w, c)

    if not add_coords:
        return rgb_flat

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([xx, yy], axis=-1)
    coords = np.broadcast_to(coords, (n, h, w, 2)).reshape(n * h * w, 2)
    return np.concatenate([rgb_flat, coords], axis=1)


def fit_ridge_probe(X, Y, alpha=1e-3):
    """Fit ridge regression with explicit intercept in closed form."""
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    gram = X_c.T @ X_c + alpha * np.eye(X_c.shape[1], dtype=np.float32)
    rhs = X_c.T @ Y_c
    W = np.linalg.solve(gram, rhs)
    b = (Y_mean - X_mean @ W).ravel()
    return W, b


def predict_ridge_probe(X, W, b):
    """Predict RGB with clipping to [0, 1]."""
    return np.clip(X @ W + b[np.newaxis, :], 0.0, 1.0)


def probe_image_decodability(pre_imgs, target_imgs, alpha=1e-3, add_coords=True):
    """Leave-one-image-out linear decodability of target RGB from pre RGB."""
    pre_norm = normalize_float(pre_imgs)
    target_norm = normalize_float(target_imgs)

    n, h, w, c = pre_norm.shape
    predictions = np.zeros((n, h, w, c), dtype=np.float32)
    mse_values = np.zeros(n, dtype=np.float32)
    r2_values = np.zeros(n, dtype=np.float32)

    for k in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[k] = False

        X_train = make_pixel_dataset(pre_norm[train_mask], add_coords=add_coords)
        Y_train = target_norm[train_mask].reshape(-1, c)

        X_test = make_pixel_dataset(pre_norm[k:k + 1], add_coords=add_coords)
        Y_test = target_norm[k:k + 1].reshape(-1, c)

        W, b = fit_ridge_probe(X_train, Y_train, alpha=alpha)
        pred_flat = predict_ridge_probe(X_test, W, b)
        predictions[k] = pred_flat.reshape(h, w, c)

        resid = Y_test - pred_flat
        mse_values[k] = float(np.mean(resid ** 2))

        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((Y_test - Y_test.mean(axis=0, keepdims=True)) ** 2))
        r2_values[k] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "predictions": predictions,
        "mse_values": mse_values,
        "r2_values": r2_values,
        "mean_mse": float(mse_values.mean()),
        "mean_r2": float(r2_values.mean()),
        "std_mse": float(mse_values.std()),
        "std_r2": float(r2_values.std()),
    }
