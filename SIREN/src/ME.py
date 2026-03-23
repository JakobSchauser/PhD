import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _channels(img):
	if img.ndim == 2:
		return [img]
	return [img[..., c] for c in range(img.shape[-1])]


def _entropy_1d(x, bins=256):
	p = np.histogram(x.ravel(), bins=bins)[0].astype(np.float64)
	p /= p.sum()
	p = p[p > 0]
	return -(p * np.log2(p)).sum()


def _joint_entropy_2d(x, y, bins=256):
	p = np.histogram2d(x.ravel(), y.ravel(), bins=bins)[0].astype(np.float64)
	p /= p.sum()
	p = p[p > 0]
	return -(p * np.log2(p)).sum()


def entropy(img, bins=256):
	return float(np.mean([_entropy_1d(ch, bins=bins) for ch in _channels(img)]))


def mutual_information(a, b, bins=256):
	vals = []
	for ac, bc in zip(_channels(a), _channels(b)):
		ha = _entropy_1d(ac, bins=bins)
		hb = _entropy_1d(bc, bins=bins)
		hab = _joint_entropy_2d(ac, bc, bins=bins)
		vals.append(ha + hb - hab)
	return float(np.mean(vals))


def normalized_mi(a, b, bins=256):
	ha = entropy(a, bins=bins)
	hb = entropy(b, bins=bins)
	return float((2.0 * mutual_information(a, b, bins=bins)) / (ha + hb))


def ssim_pair(a, b):
	return float(ssim(a, b, channel_axis=-1, data_range=a.max() - a.min()))


def _fft_magnitudes(img):
	return [np.abs(np.fft.rfft2(img[..., c])).ravel() for c in range(img.shape[-1])]


def fourier_entropy(img, bins=256):
	return float(np.mean([_entropy_1d(m, bins=bins) for m in _fft_magnitudes(img)]))


def fourier_mutual_information(a, b, bins=256):
	vals = []
	for ma, mb in zip(_fft_magnitudes(a), _fft_magnitudes(b)):
		ha = _entropy_1d(ma, bins=bins)
		hb = _entropy_1d(mb, bins=bins)
		hab = _joint_entropy_2d(ma, mb, bins=bins)
		vals.append(ha + hb - hab)
	return float(np.mean(vals))


def fourier_normalized_mi(a, b, bins=256):
	ha = fourier_entropy(a, bins=bins)
	hb = fourier_entropy(b, bins=bins)
	return float((2.0 * fourier_mutual_information(a, b, bins=bins)) / (ha + hb))


def fourier_pairwise_mi_matrix(imgs, bins=256):
	n = imgs.shape[0]
	m = np.zeros((n, n), dtype=np.float64)
	for i in range(n):
		m[i, i] = 1.0
		for j in range(i + 1, n):
			v = fourier_normalized_mi(imgs[i], imgs[j], bins=bins)
			m[i, j] = v
			m[j, i] = v
	return m


def pairwise_mi_matrix(imgs, bins=256):
	n = imgs.shape[0]
	m = np.zeros((n, n), dtype=np.float64)
	for i in range(n):
		m[i, i] = 1.0
		for j in range(i + 1, n):
			v = normalized_mi(imgs[i], imgs[j], bins=bins)
			m[i, j] = v
			m[j, i] = v
	return m


def pixel_variance_map(imgs):
	return imgs.var(axis=0)


def _upper_triangle_mean(m):
	i, j = np.triu_indices_from(m, k=1)
	return float(m[i, j].mean())


class ImageMetrics:
	def __init__(self, pre_imgs, target_imgs, bins=256):
		self.pre = pre_imgs
		self.target = target_imgs
		self.bins = bins

	def entropy(self, img):
		return entropy(img, bins=self.bins)

	def mutual_information(self, a, b):
		return mutual_information(a, b, bins=self.bins)

	def normalized_mi(self, a, b):
		return normalized_mi(a, b, bins=self.bins)

	def ssim_pair(self, a, b):
		return ssim_pair(a, b)

	def pairwise_mi_matrix(self, imgs):
		return pairwise_mi_matrix(imgs, bins=self.bins)

	def fourier_entropy(self, img):
		return fourier_entropy(img, bins=self.bins)

	def fourier_normalized_mi(self, a, b):
		return fourier_normalized_mi(a, b, bins=self.bins)

	def fourier_pairwise_mi_matrix(self, imgs):
		return fourier_pairwise_mi_matrix(imgs, bins=self.bins)

	def pixel_variance_map(self, imgs):
		return pixel_variance_map(imgs)

	def cross_stage_metrics(self):
		pair_nmi = np.array(
			[self.normalized_mi(self.pre[k], self.target[k]) for k in range(self.pre.shape[0])]
		)
		pair_ssim = np.array([self.ssim_pair(self.pre[k], self.target[k]) for k in range(self.pre.shape[0])])

		pre_pairwise = self.pairwise_mi_matrix(self.pre)
		tgt_pairwise = self.pairwise_mi_matrix(self.target)

		pre_fourier_pairwise = self.fourier_pairwise_mi_matrix(self.pre)
		tgt_fourier_pairwise = self.fourier_pairwise_mi_matrix(self.target)

		return {
			"pair_nmi": pair_nmi,
			"pair_ssim": pair_ssim,
			"mean_pair_nmi": float(pair_nmi.mean()),
			"mean_pair_ssim": float(pair_ssim.mean()),
			"pre_entropy_mean": float(np.mean([self.entropy(x) for x in self.pre])),
			"target_entropy_mean": float(np.mean([self.entropy(x) for x in self.target])),
			"pre_pairwise_nmi": pre_pairwise,
			"target_pairwise_nmi": tgt_pairwise,
			"pre_internal_nmi_mean": _upper_triangle_mean(pre_pairwise),
			"target_internal_nmi_mean": _upper_triangle_mean(tgt_pairwise),
			"pre_pixel_variance": self.pixel_variance_map(self.pre),
			"target_pixel_variance": self.pixel_variance_map(self.target),
			"pre_fourier_entropy_mean": float(np.mean([self.fourier_entropy(x) for x in self.pre])),
			"target_fourier_entropy_mean": float(np.mean([self.fourier_entropy(x) for x in self.target])),
			"pre_fourier_pairwise_nmi": pre_fourier_pairwise,
			"target_fourier_pairwise_nmi": tgt_fourier_pairwise,
			"pre_fourier_internal_nmi_mean": _upper_triangle_mean(pre_fourier_pairwise),
			"target_fourier_internal_nmi_mean": _upper_triangle_mean(tgt_fourier_pairwise),
		}

	@staticmethod
	def plot_mi_heatmap(matrix, title="Pairwise NMI"):
		plt.figure(figsize=(5, 4))
		plt.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
		plt.colorbar(label="NMI")
		plt.title(title)
		plt.xlabel("Image index")
		plt.ylabel("Image index")
		plt.tight_layout()

	@staticmethod
	def plot_pixel_variance(var_map, title="Pixel Variance"):
		vis = var_map.mean(axis=-1) if var_map.ndim == 3 else var_map
		plt.figure(figsize=(5, 4))
		plt.imshow(vis, cmap="magma")
		plt.colorbar(label="Variance")
		plt.title(title)
		plt.axis("off")
		plt.tight_layout()

	def plot_summary(self):
		r = self.cross_stage_metrics()
		
		fig, ax = plt.subplots(2, 2, figsize=(10, 6))
		
		ax[0, 0].bar(["pre", "target"], [r["pre_internal_nmi_mean"], r["target_internal_nmi_mean"]], color=["#0ea5e9", "#f97316"])
		ax[0, 0].set_title("Internal Pairwise NMI")
		ax[0, 0].set_ylim(0, 1)
		
		ax[0, 1].bar(["NMI", "SSIM"], [r["mean_pair_nmi"], r["mean_pair_ssim"]], color=["#ec4899", "#22c55e"])
		ax[0, 1].set_title("Cross-stage Metrics (mean)")
		ax[0, 1].set_ylim(0, 1)
		
		ax[1, 0].bar(["pre", "target"], [r["pre_entropy_mean"], r["target_entropy_mean"]], color=["#0ea5e9", "#f97316"])
		ax[1, 0].set_title("Entropy")
		
		ax[1, 1].axis("off")
		txt = f"pre_internal_nmi: {r['pre_internal_nmi_mean']:.3f}\ntarget_internal_nmi: {r['target_internal_nmi_mean']:.3f}\nmean_pair_nmi: {r['mean_pair_nmi']:.3f}\nmean_pair_ssim: {r['mean_pair_ssim']:.3f}\npre_entropy: {r['pre_entropy_mean']:.3f}\ntarget_entropy: {r['target_entropy_mean']:.3f}"
		ax[1, 1].text(0.1, 0.5, txt, fontsize=11, family="monospace", verticalalignment="center")
		
		fig.tight_layout()
		return r

	def summarize(self, show=True):
		r = self.plot_summary()
		if show:
			plt.show()
		return r


def cross_stage_metrics(pre_imgs, target_imgs, bins=256):
	return ImageMetrics(pre_imgs, target_imgs, bins=bins).cross_stage_metrics()


def plot_mi_heatmap(matrix, title="Pairwise NMI"):
	ImageMetrics.plot_mi_heatmap(matrix, title=title)


def plot_pixel_variance(var_map, title="Pixel Variance"):
	ImageMetrics.plot_pixel_variance(var_map, title=title)


def plot_summary(pre_imgs, target_imgs, bins=256):
	return ImageMetrics(pre_imgs, target_imgs, bins=bins).plot_summary()


def summarize(pre_imgs, target_imgs, bins=256, show=True):
	return ImageMetrics(pre_imgs, target_imgs, bins=bins).summarize(show=show)

