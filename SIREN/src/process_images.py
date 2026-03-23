import matplotlib.pyplot as plt
import numpy as np


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
