"""
Nonlinear pixel-level probe for predicting target RGB from pre RGB.
Uses random Fourier features (RBF kernel approximation) + ridge regression.
"""

import numpy as np

from src.probe import fit_ridge_probe, make_pixel_dataset, normalize_float, predict_ridge_probe


def _rff_map(X, W_rff, b_rff):
	"""Map inputs to random Fourier features for an RBF kernel approximation."""
	proj = X @ W_rff + b_rff[np.newaxis, :]
	scale = np.sqrt(2.0 / W_rff.shape[1])
	return scale * np.cos(proj)


def _make_rff_parameters(input_dim, n_features=256, gamma=10.0, rng=None):
	"""Create random weights and phase offsets for the RFF map."""
	if rng is None:
		rng = np.random.default_rng()
	# For k(x, y) = exp(-gamma ||x-y||^2), sample w ~ N(0, 2*gamma I).
	W_rff = rng.normal(
		loc=0.0,
		scale=np.sqrt(2.0 * gamma),
		size=(input_dim, n_features),
	).astype(np.float32)
	b_rff = rng.uniform(0.0, 2.0 * np.pi, size=(n_features,)).astype(np.float32)
	return W_rff, b_rff


def nonlinear_probe_image_decodability(
	pre_imgs,
	target_imgs,
	alpha=1e-3,
	add_coords=True,
	n_features=256,
	gamma=10.0,
	seed=0,
):
	"""Leave-one-image-out nonlinear decodability of target RGB from pre RGB."""
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

		rng = np.random.default_rng(seed + k)
		W_rff, b_rff = _make_rff_parameters(
			input_dim=X_train.shape[1],
			n_features=n_features,
			gamma=gamma,
			rng=rng,
		)

		Z_train = _rff_map(X_train, W_rff, b_rff)
		Z_test = _rff_map(X_test, W_rff, b_rff)

		W, b = fit_ridge_probe(Z_train, Y_train, alpha=alpha)
		pred_flat = predict_ridge_probe(Z_test, W, b)
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
		"n_features": int(n_features),
		"gamma": float(gamma),
	}
