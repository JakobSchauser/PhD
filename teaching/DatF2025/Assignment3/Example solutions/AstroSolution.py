#!/usr/bin/env python3
"""
Combine Gaia + SDSS + UKIDSS + WISE photometry into a single table.

Make colour-colour plots.

Usage example:
  python combine_photometry.py 
"""

import pandas as pd
from astropy.table import Table
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

t = Table.read('1761463744013O-result_newselection.fits', format="fits")
df_gaia = t.to_pandas()

df_sdss = pd.read_csv('1762095020857A_sdss.csv', comment="#")
sdss_cols = [
        "source_id",
        "umag", "e_umag",
        "gmag", "e_gmag",
        "rmag", "e_rmag",
        "imag", "e_imag",
        "zmag", "e_zmag",
]

df_ukidss = pd.read_csv('1762095199241A_ukidss.csv', comment="#")
ukidss_cols = [
        "source_id",
        "yAperMag3",   "yAperMag3Err",
        "j_1AperMag3", "j_1AperMag3Err",
        "hAperMag3",   "hAperMag3Err",
        "kAperMag3",   "kAperMag3Err",
]

df_wise = pd.read_csv('1762095353124A_WISE.csv', comment="#")
wise_cols = [
        "source_id",
        "W1mag", "e_W1mag",
        "W2mag", "e_W2mag",
        "W3mag", "e_W3mag",
        "W4mag", "e_W4mag",
]

#Merge the tables based on source_id
merged = df_gaia.merge(df_sdss[sdss_cols], on="source_id", how="left")
merged = merged.merge(df_ukidss[ukidss_cols], on="source_id", how="left")
merged = merged.merge(df_wise[wise_cols], on="source_id", how="left")

# Find rows with information in all bands
cols_needed = ["W1mag", "W2mag",            # WISE
               "j_1AperMag3", "kAperMag3",  # UKIDSS (or 'J', 'K' etc.)
               "gmag", "rmag"]              # SDSS  (or 'gmag', 'rmag')

# Keep only rows where ALL these are non-NaN and finite
mask = np.isfinite(merged[cols_needed]).all(axis=1)
clean = merged[mask]

#Make plots
J_minus_K  = clean["j_1AperMag3"] - clean["kAperMag3"]
W1_minus_W2 = clean["W1mag"] - clean["W2mag"]
g_minus_r  = clean["gmag"] - clean["rmag"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

# Panel 1: W1 - W2 vs J - K
axes[0].scatter(J_minus_K, W1_minus_W2, s=5, alpha=0.5)
axes[0].set_xlabel("J - K")
axes[0].set_ylabel("W1 - W2")
axes[0].set_title("WISE vs UKIDSS colors")

# Panel 2: g - r vs J - K
axes[1].scatter(J_minus_K, g_minus_r, s=5, alpha=0.5)
axes[1].set_xlabel("J - K")
axes[1].set_ylabel("g - r")
axes[1].set_xlim(0.0,3.0)
axes[1].set_ylim(-0.5,2.0)
axes[1].set_title("SDSS vs UKIDSS colors")

plt.tight_layout()
plt.show()

#More fancey plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

# -------- Panel 1 (unchanged): W1 - W2 vs J - K --------
axes[0].scatter(J_minus_K, W1_minus_W2, s=5, alpha=0.5)
axes[0].set_xlabel("J - K")
axes[0].set_ylabel("W1 - W2")
axes[0].set_title("WISE vs UKIDSS colors")

# -------- Panel 2 (color-coded): g - r vs J - K --------
scatter = axes[1].scatter(J_minus_K, g_minus_r, s=5, alpha=0.6,
                          c=W1_minus_W2, cmap="plasma")  # <-- color by W1-W2

axes[1].set_xlabel("J - K")
axes[1].set_ylabel("g - r")
axes[1].set_xlim(0.0,3.0)
axes[1].set_ylim(-0.5,2.0)
axes[1].set_title("SDSS vs UKIDSS colors")

# Add colorbar for W1 - W2
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label("W1 - W2")

plt.tight_layout()
plt.show()
