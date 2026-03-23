"""
Gregor-inspired positional information metrics.

Compute how much the RGB color at a pixel encodes its spatial position.
This measures whether the image has spatially-localized, structured colors
vs. random or uniform coloring.
"""

import numpy as np


def rgb_position_mi(img, rgb_bins=32):
    """
    Compute mutual information between RGB color and spatial position.
    
    High MI → colors are location-specific (structured image)
    Low MI  → colors appear uniformly across all positions (noisy/uniform)
    
    Args:
        img: Array of shape (H, W, 3) with pixel values 0-255 (uint8)
        rgb_bins: Number of bins for RGB discretization
        
    Returns:
        mi: Mutual information I(RGB; Position) in nats
        h_rgb: Entropy of RGB distribution
        h_pos: Entropy of position distribution  
        h_joint: Joint entropy H(RGB, Position)
    """
    h, w, c = img.shape
    assert c == 3, "Expected RGB image"
    n_pixels = h * w
    
    # Discretize RGB into bins for each channel independently
    rgb_flat = img.reshape(-1, c).astype(np.float32)
    rgb_digits = np.zeros((n_pixels, c), dtype=np.int32)
    for ch in range(c):
        # Digitize to get bin indices
        rgb_digits[:, ch] = np.digitize(rgb_flat[:, ch], bins=np.linspace(0, 255, rgb_bins + 1)[1:-1]) 
    
    # Convert RGB triplet to single categorical index
    # E.g., (r, g, b) bins → r * rgb_bins^2 + g * rgb_bins + b
    rgb_code = (rgb_digits[:, 0] * rgb_bins * rgb_bins + 
                rgb_digits[:, 1] * rgb_bins + 
                rgb_digits[:, 2]).astype(np.int32)
    
    # Position index: 0 to n_pixels-1 (each pixel is its own position)
    pos_code = np.arange(n_pixels, dtype=np.int32)
    
    # Build joint histogram: P(rgb_code, pos_code)
    n_rgb = rgb_code.max() + 1
    n_pos = pos_code.max() + 1
    
    joint_hist = np.zeros((n_rgb, n_pos), dtype=np.float64)
    for i in range(n_pixels):
        joint_hist[rgb_code[i], pos_code[i]] += 1
    
    # Normalize to probability distribution
    joint_hist /= n_pixels
    
    # Marginal distributions
    p_rgb = joint_hist.sum(axis=1)
    p_pos = joint_hist.sum(axis=0)
    
    # Entropies in nats
    h_rgb = -np.sum(p_rgb[p_rgb > 0] * np.log(p_rgb[p_rgb > 0]))
    h_pos = -np.sum(p_pos[p_pos > 0] * np.log(p_pos[p_pos > 0]))
    h_joint = -np.sum(joint_hist[joint_hist > 0] * np.log(joint_hist[joint_hist > 0]))
    
    # Mutual information: I(RGB; Pos) = H(RGB) + H(Pos) - H(RGB, Pos)
    mi = h_rgb + h_pos - h_joint
    
    return mi, h_rgb, h_pos, h_joint


def batch_rgb_position_mi(imgs, rgb_bins=32):
    """
    Compute MI(RGB → position) for a batch of images.
    
    Args:
        imgs: Array of shape (N, H, W, 3) with pixel values 0-255
        rgb_bins: Number of bins for RGB discretization
        
    Returns:
        mis: Array of shape (N,) with MI values per image
        mean_mi: Mean MI across batch
        std_mi: Std dev across batch
    """
    mis = np.array([rgb_position_mi(imgs[k], rgb_bins=rgb_bins)[0] for k in range(len(imgs))])
    return mis, mis.mean(), mis.std()


def position_uniqueness_map(img, rgb_bins=32):
    """
    For each pixel, compute how uniquely identifiable its position is by its RGB.
    
    Uniqueness = 1 / (# of pixels with same RGB value)
    - Value 1.0 → position is uniquely identified by its color (no other pixel has that color)
    - Value < 1.0 → multiple pixels share the same color
    - Mean over all pixels measures avg position predictability
    
    Args:
        img: Array of shape (H, W, 3) with pixel values 0-255
        rgb_bins: Number of bins for RGB discretization
        
    Returns:
        uniqueness_map: Array of shape (H, W) with uniqueness scores 0-1
    """
    h, w, c = img.shape
    assert c == 3, "Expected RGB image"
    n_pixels = h * w
    
    # Discretize RGB
    rgb_flat = img.reshape(-1, c).astype(np.float32)
    rgb_digits = np.zeros((n_pixels, c), dtype=np.int32)
    for ch in range(c):
        rgb_digits[:, ch] = np.digitize(rgb_flat[:, ch], bins=np.linspace(0, 255, rgb_bins + 1)[1:-1])
    
    # Encode RGB as single integer
    rgb_code = (rgb_digits[:, 0] * rgb_bins * rgb_bins + 
                rgb_digits[:, 1] * rgb_bins + 
                rgb_digits[:, 2]).astype(np.int32)
    
    # Count occurrences of each color
    unique_colors, inverse_indices, counts = np.unique(rgb_code, return_inverse=True, return_counts=True)
    
    # Uniqueness = 1 / count for each color
    uniqueness = 1.0 / counts[inverse_indices]
    uniqueness_map = uniqueness.reshape(h, w)
    
    return uniqueness_map


def gregor_stage_comparison(pre_imgs, target_imgs, rgb_bins=32):
    """
    Compare MI(RGB → position) between pre-SIREN and post-SIREN images.
    
    Args:
        pre_imgs: Array of shape (N, H, W, 3) — pre-SIREN intermediate images
        target_imgs: Array of shape (N, H, W, 3) — post-SIREN final images
        rgb_bins: RGB discretization bins
        
    Returns:
        results: Dict with MI comparisons and uniqueness metrics
    """
    pre_mis, pre_mean, pre_std = batch_rgb_position_mi(pre_imgs, rgb_bins=rgb_bins)
    tgt_mis, tgt_mean, tgt_std = batch_rgb_position_mi(target_imgs, rgb_bins=rgb_bins)
    
    # Uniqueness maps
    pre_unique_means = np.array([position_uniqueness_map(pre_imgs[k], rgb_bins).mean() for k in range(len(pre_imgs))])
    tgt_unique_means = np.array([position_uniqueness_map(target_imgs[k], rgb_bins).mean() for k in range(len(target_imgs))])
    
    return {
        "pre_mi_values": pre_mis,
        "target_mi_values": tgt_mis,
        "pre_mi_mean": float(pre_mean),
        "target_mi_mean": float(tgt_mean),
        "pre_mi_std": float(pre_std),
        "target_mi_std": float(tgt_std),
        "delta_mi": float(tgt_mean - pre_mean),
        "ratio_mi": float(tgt_mean / (pre_mean + 1e-8)),
        "pre_uniqueness_values": pre_unique_means,
        "target_uniqueness_values": tgt_unique_means,
        "pre_uniqueness_mean": float(pre_unique_means.mean()),
        "target_uniqueness_mean": float(tgt_unique_means.mean()),
        "delta_uniqueness": float(tgt_unique_means.mean() - pre_unique_means.mean()),
    }
