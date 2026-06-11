#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

CSV = os.path.join(os.path.dirname(__file__),
                   "stack_master/maps/final/global_waypoints.csv")

df = pd.read_csv(CSV)
x, y = df["x_m"].values, df["y_m"].values

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("global_waypoints.csv — final map", fontsize=14)

# ── 1. Track geometry (path + track width) ──────────────────────────────────
ax = axes[0]
ax.set_title("Track geometry")
if {"w_tr_right_m", "w_tr_left_m", "psi_rad"}.issubset(df.columns):
    psi   = df["psi_rad"].values
    n_x   = -np.sin(psi)   # normal vector (left)
    n_y   =  np.cos(psi)
    r     = df["w_tr_right_m"].values
    l     = df["w_tr_left_m"].values
    ax.fill(
        np.concatenate([x + n_x * l, (x - n_x * r)[::-1]]),
        np.concatenate([y + n_y * l, (y - n_y * r)[::-1]]),
        alpha=0.15, color="gray", label="track width"
    )
    ax.plot(x - n_x * r, y - n_y * r, "k--", lw=0.8, label="right boundary")
    ax.plot(x + n_x * l, y + n_y * l, "k-",  lw=0.8, label="left boundary")
ax.plot(x, y, "b-", lw=1.5, label="centre line")
ax.plot(x[0], y[0], "go", ms=10, label="start")
ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

# ── 2. Speed profile ─────────────────────────────────────────────────────────
ax = axes[1]
ax.set_title("Speed profile")
if "vx_mps" in df.columns:
    sc = ax.scatter(x, y, c=df["vx_mps"], cmap="RdYlGn",
                    s=8, vmin=df["vx_mps"].min(), vmax=df["vx_mps"].max())
    plt.colorbar(sc, ax=ax, label="vx [m/s]")
ax.plot(x[0], y[0], "ko", ms=8)
ax.set_aspect("equal"); ax.grid(True, alpha=0.4)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

# ── 3. Curvature profile ──────────────────────────────────────────────────────
ax = axes[2]
ax.set_title("Curvature profile")
if "kappa_radpm" in df.columns:
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    ax.plot(s, df["kappa_radpm"], color="purple", lw=1.2)
    ax.axhline(0, color="k", lw=0.5, linestyle="--")
    ax.set_xlabel("s [m]"); ax.set_ylabel("κ [rad/m]")
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "waypoints_plot.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved → waypoints_plot.png")
