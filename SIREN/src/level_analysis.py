from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

from src.ME import ImageMetrics
from src.gregor import gregor_stage_comparison
from src.nonlinear_probe import nonlinear_probe_image_decodability
from src.process_images import load_image_set_from_dir, load_rgb
from src.probe import probe_image_decodability


def _pairwise_ssim_values(imgs):
	n = imgs.shape[0]
	vals = []
	for i in range(n):
		for j in range(i + 1, n):
			vals.append(ssim(imgs[i], imgs[j], channel_axis=-1, data_range=255))
	return np.array(vals, dtype=np.float64)


def discover_level_dirs(base_dir):
	"""Discover preprocessing level folders (e.g. lvl1, lvl2, ...)."""
	base_dir = Path(base_dir)
	level_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
	return sorted(level_dirs, key=lambda p: p.name)


def load_levels(base_dir, target_dir=None):
	"""Load all discovered level folders and pair each pre set with canonical targets."""
	base_dir = Path(base_dir)
	if target_dir is None:
		target_dir = base_dir.parent.parent / "targets"
	else:
		target_dir = Path(target_dir)

	level_dirs = discover_level_dirs(base_dir)
	if len(level_dirs) == 0:
		raise ValueError(f"No level folders found in {base_dir}")

	# Use the first level to define canonical target stack shape.
	pre_ref = load_image_set_from_dir(level_dirs[0], glob_pattern="siren-examples_best_*.png")
	target_h, target_w = pre_ref.shape[1:3]

	target_paths = sorted(target_dir.glob("*.png"), key=lambda p: p.name)
	if len(target_paths) == 0:
		raise ValueError(f"No images found in {target_dir} matching *.png")
	if len(target_paths) != pre_ref.shape[0]:
		raise ValueError(
			f"Mismatched number of images between {target_dir} and {level_dirs[0]}: "
			f"target={len(target_paths)}, pre={pre_ref.shape[0]}"
		)

	target_imgs = []
	for p in target_paths:
		img = load_rgb(str(p))
		if img.shape[0] != target_h or img.shape[1] != target_w:
			img = (resize(img, (target_h, target_w), anti_aliasing=True) * 255).astype(np.uint8)
		target_imgs.append(img)
	target = np.stack(target_imgs)

	level_sets = {}
	for level_dir in level_dirs:
		pre = load_image_set_from_dir(level_dir, glob_pattern="siren-examples_best_*.png")
		if pre.shape[0] != target.shape[0]:
			raise ValueError(
				f"Mismatched number of images between {level_dir} and {target_dir}: "
				f"pre={pre.shape[0]}, target={target.shape[0]}"
			)
		if pre.shape[1:] != target.shape[1:]:
			raise ValueError(
				f"Mismatched image shapes between {level_dir} and {target_dir}: "
				f"pre={pre.shape[1:]}, target={target.shape[1:]}"
			)
		level_sets[level_dir.name] = {"pre": pre, "target": target}
	return level_sets


def compute_level_results(
	base_dir,
	target_dir=None,
	bins=128,
	probe_alpha=1e-3,
	nonlinear_probe_alpha=1e-3,
	nonlinear_probe_features=256,
	nonlinear_probe_gamma=10.0,
):
	"""Compute the same analysis metrics for each preprocessing level."""
	level_sets = load_levels(base_dir, target_dir=target_dir)
	level_names = list(level_sets.keys())

	# Assume the target set is shared across levels; validate and keep first one.
	shared_target = level_sets[level_names[0]]["target"]
	for level_name in level_names[1:]:
		if level_sets[level_name]["target"].shape != shared_target.shape:
			raise ValueError("Target shapes differ across levels")

	target_internal_ssim = _pairwise_ssim_values(shared_target)

	results = {
		"level_names": level_names,
		"target": {
			"internal_ssim_values": target_internal_ssim,
			"internal_ssim_mean": float(target_internal_ssim.mean()),
		},
		"levels": {},
	}

	for level_name in level_names:
		pre = level_sets[level_name]["pre"]
		target = level_sets[level_name]["target"]

		metrics = ImageMetrics(pre, target, bins=bins)
		cross = metrics.cross_stage_metrics()
		gregor = gregor_stage_comparison(pre, target, rgb_bins=32)
		probe = probe_image_decodability(pre, target, alpha=probe_alpha, add_coords=False)
		nonlinear_probe = nonlinear_probe_image_decodability(
			pre,
			target,
			alpha=nonlinear_probe_alpha,
			add_coords=False,
			n_features=nonlinear_probe_features,
			gamma=nonlinear_probe_gamma,
		)

		pre_internal_ssim = _pairwise_ssim_values(pre)

		results["levels"][level_name] = {
			"pre": pre,
			"target": target,
			"cross": cross,
			"pre_internal_ssim_values": pre_internal_ssim,
			"pre_internal_ssim_mean": float(pre_internal_ssim.mean()),
			"gregor": gregor,
			"probe": probe,
			"nonlinear_probe": nonlinear_probe,
		}

	return results


def print_level_summary(results):
	"""Print a compact table of main metrics per level."""
	header = (
		f"{'level':<8} {'pair_ssim':>10} {'pair_nmi':>10} {'int_ssim':>10} "
		f"{'int_nmi':>10} {'fourier_nmi':>12} {'gregor_mi':>10} {'probe_r2':>10}"
	)
	print(header)
	print("-" * len(header))
	for level_name in results["level_names"]:
		r = results["levels"][level_name]
		cross = r["cross"]
		print(
			f"{level_name:<8} "
			f"{cross['mean_pair_ssim']:>10.3f} "
			f"{cross['mean_pair_nmi']:>10.3f} "
			f"{r['pre_internal_ssim_mean']:>10.3f} "
			f"{cross['pre_internal_nmi_mean']:>10.3f} "
			f"{cross['pre_fourier_internal_nmi_mean']:>12.3f} "
			f"{r['gregor']['pre_mi_mean']:>10.3f} "
			f"{r['probe']['mean_r2']:>10.3f}"
		)


def plot_level_scalar_summary(results):
	"""Plot scalar metrics across all preprocessing levels."""
	level_names = results["level_names"]
	x = np.arange(len(level_names))

	pair_ssim = [results["levels"][k]["cross"]["mean_pair_ssim"] for k in level_names]
	pair_nmi = [results["levels"][k]["cross"]["mean_pair_nmi"] for k in level_names]
	internal_ssim = [results["levels"][k]["pre_internal_ssim_mean"] for k in level_names]
	internal_nmi = [results["levels"][k]["cross"]["pre_internal_nmi_mean"] for k in level_names]
	fourier_nmi = [results["levels"][k]["cross"]["pre_fourier_internal_nmi_mean"] for k in level_names]
	gregor_mi = [results["levels"][k]["gregor"]["pre_mi_mean"] for k in level_names]
	probe_r2 = [results["levels"][k]["probe"]["mean_r2"] for k in level_names]
	probe_mse = [results["levels"][k]["probe"]["mean_mse"] for k in level_names]

	fig, ax = plt.subplots(2, 4, figsize=(16, 8))
	series = [
		(pair_ssim, "Cross-stage SSIM"),
		(pair_nmi, "Cross-stage NMI"),
		(internal_ssim, "Internal SSIM"),
		(internal_nmi, "Internal pixel NMI"),
		(fourier_nmi, "Internal Fourier NMI"),
		(gregor_mi, "Gregor positional MI"),
		(probe_r2, "Probe R^2"),
		(probe_mse, "Probe MSE"),
	]

	for axis, (vals, title) in zip(ax.flat, series):
		axis.plot(x, vals, "o-", linewidth=2, markersize=7)
		axis.set_xticks(x)
		axis.set_xticklabels(level_names)
		axis.set_title(title)
		axis.grid(alpha=0.3, axis="y")

	fig.tight_layout()
	return fig


def plot_level_pairwise_trajectories(results):
	"""Plot pairwise trajectories across levels and final target for key cohort metrics."""
	level_names = results["level_names"]
	xlabels = level_names + ["target"]
	xs = np.arange(len(xlabels))

	pre_internal_nmi = [results["levels"][k]["cross"]["pre_pairwise_nmi"] for k in level_names]
	target_pairwise_nmi = results["levels"][level_names[0]]["cross"]["target_pairwise_nmi"]
	i_idx, j_idx = np.triu_indices_from(target_pairwise_nmi, k=1)

	pre_internal_ssim = [results["levels"][k]["pre_internal_ssim_values"] for k in level_names]
	target_internal_ssim = results["target"]["internal_ssim_values"]

	pre_fourier_nmi = [results["levels"][k]["cross"]["pre_fourier_pairwise_nmi"] for k in level_names]
	target_fourier_nmi = results["levels"][level_names[0]]["cross"]["target_fourier_pairwise_nmi"]

	fig, ax = plt.subplots(1, 3, figsize=(16, 5))

	# Pixel-space internal NMI
	for idx in range(len(i_idx)):
		vals = [m[i_idx[idx], j_idx[idx]] for m in pre_internal_nmi] + [target_pairwise_nmi[i_idx[idx], j_idx[idx]]]
		ax[0].plot(xs, vals, "o-", alpha=0.15, linewidth=1, markersize=3, color="#666")
	mean_vals = [results["levels"][k]["cross"]["pre_internal_nmi_mean"] for k in level_names] + [results["levels"][level_names[0]]["cross"]["target_internal_nmi_mean"]]
	ax[0].plot(xs, mean_vals, "o-", linewidth=3, markersize=8, color="#ef4444", label="mean")
	ax[0].set_title("Internal NMI")

	# Internal SSIM
	for idx in range(len(target_internal_ssim)):
		vals = [m[idx] for m in pre_internal_ssim] + [target_internal_ssim[idx]]
		ax[1].plot(xs, vals, "o-", alpha=0.15, linewidth=1, markersize=3, color="#666")
	mean_vals = [results["levels"][k]["pre_internal_ssim_mean"] for k in level_names] + [results["target"]["internal_ssim_mean"]]
	ax[1].plot(xs, mean_vals, "o-", linewidth=3, markersize=8, color="#22c55e", label="mean")
	ax[1].set_title("Internal SSIM")

	# Fourier internal NMI
	for idx in range(len(i_idx)):
		vals = [m[i_idx[idx], j_idx[idx]] for m in pre_fourier_nmi] + [target_fourier_nmi[i_idx[idx], j_idx[idx]]]
		ax[2].plot(xs, vals, "o-", alpha=0.15, linewidth=1, markersize=3, color="#666")
	mean_vals = [results["levels"][k]["cross"]["pre_fourier_internal_nmi_mean"] for k in level_names] + [results["levels"][level_names[0]]["cross"]["target_fourier_internal_nmi_mean"]]
	ax[2].plot(xs, mean_vals, "o-", linewidth=3, markersize=8, color="#a855f7", label="mean")
	ax[2].set_title("Internal Fourier NMI")

	for axis in ax:
		axis.set_xticks(xs)
		axis.set_xticklabels(xlabels)
		axis.grid(alpha=0.3, axis="y")
		axis.legend()

	fig.tight_layout()
	return fig