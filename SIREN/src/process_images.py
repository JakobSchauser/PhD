import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


def save_image_set_to_dir(
	imgs,
	dir_path,
	prefix="img",
	ext=".png",
	start_idx=0,
):
	"""Save a batch of RGB images with numbered filenames into a directory."""
	dir_path = Path(dir_path)
	dir_path.mkdir(parents=True, exist_ok=True)
	for idx, img in enumerate(np.asarray(imgs), start=start_idx):
		img_path = dir_path / f"{prefix}_{idx:02d}{ext}"
		plt.imsave(img_path, img.astype(np.uint8))
	return sorted(dir_path.glob(f"{prefix}_*{ext}"), key=_sort_key_numeric_suffix)


def load_rgb(path):
	x = plt.imread(path)[..., :3]
	return (x * 255).astype(np.uint8) if x.dtype != np.uint8 else x


def crop_edges(img, top=0, bottom=0, left=0, right=0):
	h, w = img.shape[:2]
	return img[top : h - bottom, left : w - right]


def split_grid(img, rows=4, cols=5):
	h, w = img.shape[:2]
	ch, cw = h // rows, w // cols
	return np.stack([
		img[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
		for r in range(rows)
		for c in range(cols)
	])


def load_pre_target(
	pre_path="data/siren-examples_best.png",
	target_path="data/examples_best.png",
	rows=4,
	cols=5,
	top=0,
	bottom=0,
	left=0,
	right=0,
):
	pre = split_grid(crop_edges(load_rgb(pre_path), top=top, bottom=bottom, left=left, right=right), rows=rows, cols=cols)
	target = split_grid(crop_edges(load_rgb(target_path), top=top, bottom=bottom, left=left, right=right), rows=rows, cols=cols)
	return pre, target


def _sort_key_numeric_suffix(path_obj):
	"""Sort by trailing integer in stem when present, fallback to stem."""
	stem = path_obj.stem
	m = re.search(r"(\d+)$", stem)
	if m:
		return (stem[: m.start()], int(m.group(1)))
	return (stem, -1)


def _extract_numeric_suffix(stem):
	m = re.search(r"(\d+)$", stem)
	return int(m.group(1)) if m else None


def load_image_set_from_dir(dir_path, glob_pattern="*.png"):
	"""Load already-separated images from a directory into shape (N, H, W, 3)."""
	dir_path = Path(dir_path)
	paths = sorted(dir_path.glob(glob_pattern), key=_sort_key_numeric_suffix)
	if len(paths) == 0:
		raise ValueError(f"No images found in {dir_path} matching {glob_pattern}")
	imgs = np.stack([load_rgb(str(p)) for p in paths])
	return imgs


def load_pre_target_from_dirs(
	pre_dir,
	target_dir,
	pre_glob_pattern="*.png",
	target_glob_pattern="*.png",
):
	"""Load paired pre/target image sets from separate directories."""
	pre = load_image_set_from_dir(pre_dir, glob_pattern=pre_glob_pattern)
	target = load_image_set_from_dir(target_dir, glob_pattern=target_glob_pattern)
	if pre.shape[0] != target.shape[0]:
		raise ValueError(f"Mismatched number of images: pre={pre.shape[0]}, target={target.shape[0]}")
	if pre.shape[1:] != target.shape[1:]:
		raise ValueError(f"Mismatched image shapes: pre={pre.shape[1:]}, target={target.shape[1:]}")
	return pre, target


def load_pre_target_from_individual_dir(
	dir_path,
	pre_prefix="siren-examples_best_",
	target_prefix="examples_best_",
	ext=".png",
):
	"""
	Load paired pre/target images from one directory by matching numeric suffixes.

	Example filenames:
	- pre:    siren-examples_best_00.png
	- target: examples_best_00.png

	Only shared indices are loaded, sorted by index.
	"""
	dir_path = Path(dir_path)

	pre_paths = list(dir_path.glob(f"{pre_prefix}*{ext}"))
	target_paths = list(dir_path.glob(f"{target_prefix}*{ext}"))

	if len(pre_paths) == 0:
		raise ValueError(f"No pre images found in {dir_path} matching {pre_prefix}*{ext}")
	if len(target_paths) == 0:
		raise ValueError(f"No target images found in {dir_path} matching {target_prefix}*{ext}")

	pre_map = {}
	for p in pre_paths:
		idx = _extract_numeric_suffix(p.stem)
		if idx is not None:
			pre_map[idx] = p

	target_map = {}
	for p in target_paths:
		idx = _extract_numeric_suffix(p.stem)
		if idx is not None:
			target_map[idx] = p

	shared_idx = sorted(set(pre_map.keys()) & set(target_map.keys()))
	if len(shared_idx) == 0:
		raise ValueError(
			f"No shared numeric suffix between {pre_prefix}* and {target_prefix}* in {dir_path}"
		)

	pre = np.stack([load_rgb(str(pre_map[i])) for i in shared_idx])
	target = np.stack([load_rgb(str(target_map[i])) for i in shared_idx])

	if pre.shape[1:] != target.shape[1:]:
		raise ValueError(f"Mismatched image shapes: pre={pre.shape[1:]}, target={target.shape[1:]}")

	return pre, target


def load_noise_sets_from_dir(base_dir, set_glob_pattern="*", image_glob_pattern="*.png"):
	"""Load multiple individual noise sets from sibling directories under a base directory."""
	base_dir = Path(base_dir)
	set_dirs = sorted([p for p in base_dir.glob(set_glob_pattern) if p.is_dir()], key=lambda p: p.name)
	if len(set_dirs) == 0:
		raise ValueError(f"No noise set folders found in {base_dir}")

	noise_sets = []
	set_names = []
	ref_shape = None
	for set_dir in set_dirs:
		imgs = load_image_set_from_dir(set_dir, glob_pattern=image_glob_pattern)
		if ref_shape is None:
			ref_shape = imgs.shape
		elif imgs.shape != ref_shape:
			raise ValueError(f"Mismatched noise set shape in {set_dir}: expected {ref_shape}, got {imgs.shape}")
		noise_sets.append(imgs)
		set_names.append(set_dir.name)

	return np.stack(noise_sets), set_names
