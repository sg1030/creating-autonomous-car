from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from Optimization.dynamics import VehicleKinematics
from Optimization.model_params import opt_params_kin
from Optimization.track_opt import MiminumTimeOptimization
from Track.track import Track


# =========================
# User Settings
# =========================
# Edit only this block when you want to rerun the script manually.
CENTERLINE_CSV = Path("/home/hmcl/data needed/final/centerline.csv")
BOUNDARY_LEFT_CSV = Path("/home/hmcl/data needed/final/boundary_left.csv")
BOUNDARY_RIGHT_CSV = Path("/home/hmcl/data needed/final/boundary_right.csv")
OUTPUT_DIR = Path("outputs/kinematic_min_laptime_initstyle")

NS = 300
DT = 0.05
TRACK_WIDTH_SCALE = 0.3

# Use the mean width from `centerline.csv` when None.
# Set a number like `1.0` if you want to force a fixed width manually.
TRACK_WIDTH_OVERRIDE_M = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class SolverConfig:
    ns: int = 300
    dt: float = 0.05
    track_width_scale: float = 0.8
    track_width_override_m: float | None = None


def save_track_assets(
    centerline_csv: Path,
    boundary_left_csv: Path,
    boundary_right_csv: Path,
    traj_dir: Path,
) -> tuple[float, float]:
    ensure_dir(traj_dir)

    center_df = pd.read_csv(centerline_csv)
    left_df = pd.read_csv(boundary_left_csv)
    right_df = pd.read_csv(boundary_right_csv)

    center_xy = center_df[["x_m", "y_m"]].to_numpy(dtype=float)
    left_xy = left_df[["x_m", "y_m"]].to_numpy(dtype=float)
    right_xy = right_df[["x_m", "y_m"]].to_numpy(dtype=float)

    np.savetxt(traj_dir / "centerline.txt", center_xy, delimiter=",")
    np.savetxt(traj_dir / "innerwall.txt", left_xy, delimiter=",")
    np.savetxt(traj_dir / "outerwall.txt", right_xy, delimiter=",")

    center_df.to_csv(traj_dir / "centerline_source.csv", index=False)
    left_df.to_csv(traj_dir / "boundary_left_source.csv", index=False)
    right_df.to_csv(traj_dir / "boundary_right_source.csv", index=False)

    if {"w_tr_right_m", "w_tr_left_m"}.issubset(center_df.columns):
        width_series = center_df["w_tr_right_m"].to_numpy(dtype=float) + center_df["w_tr_left_m"].to_numpy(dtype=float)
        track_width = float(np.mean(width_series))
        track_width_std = float(np.std(width_series))
    else:
        track_width = 2.0
        track_width_std = 0.0

    return track_width, track_width_std


def run_min_laptime(track: Track, cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dynamics = VehicleKinematics(track, dt=cfg.dt, N=cfg.ns)
    opt = MiminumTimeOptimization(dynamics, track, opt_params_kin, "Kin")
    x_pred, u_pred = opt.solve_optimization()

    sdot = x_pred[:-1, 2] * np.cos(x_pred[:-1, 1]) / (1.0 - x_pred[:-1, 0] * track.curv_center)
    x = np.zeros((len(track.s_center),), dtype=float)
    y = np.zeros((len(track.s_center),), dtype=float)
    psi = np.zeros((len(track.s_center),), dtype=float)
    for i in range(len(track.s_center)):
        x[i], y[i], psi[i] = track.local_to_global(np.array([track.s_center[i], x_pred[i, 0], x_pred[i, 1]]))

    xy = np.column_stack([x, y, psi, x_pred[:-1, 2]])
    return x_pred, u_pred, sdot, xy


def save_outputs(
    track: Track,
    x_pred: np.ndarray,
    u_pred: np.ndarray,
    sdot: np.ndarray,
    xy: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    opt_traj_fren = np.zeros((len(x_pred) - 1, x_pred.shape[1] + 3), dtype=float)
    opt_traj_fren[:, 0] = track.s_center
    opt_traj_fren[:, 1 : x_pred.shape[1]] = x_pred[:-1, :-1]
    opt_traj_fren[:, x_pred.shape[1] : -1] = u_pred
    opt_traj_fren[:, -1] = sdot

    opt_traj_xy = xy

    frenet_txt = output_dir / "optimized_traj0_frenet.txt"
    xy_txt = output_dir / "optimized_traj0.txt"
    np.savetxt(frenet_txt, opt_traj_fren, delimiter=",")
    np.savetxt(xy_txt, opt_traj_xy, delimiter=",")

    frenet_df = pd.DataFrame(
        opt_traj_fren,
        columns=["s_m", "ey_m", "epsi_rad", "v_mps", "a_mps2", "delta_rad", "sdot_mps"],
    )
    frenet_df["t_s"] = x_pred[:-1, 3]
    xy_df = pd.DataFrame(opt_traj_xy, columns=["x_m", "y_m", "psi_rad", "v_mps"])
    combined_df = pd.concat([frenet_df, xy_df], axis=1)

    frenet_df.to_csv(output_dir / "optimized_traj0_frenet.csv", index=False)
    xy_df.to_csv(output_dir / "optimized_traj0.csv", index=False)
    combined_df.to_csv(output_dir / "optimized_traj0_combined.csv", index=False)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.axis("equal")

    points = np.zeros((len(track.s_center), 1, 2))
    points[:, 0, 0] = xy[:, 0]
    points[:, 0, 1] = xy[:, 1]
    speed = xy[:, 3]

    ax.plot(track.center[:, 0], track.center[:, 1], "--k", linewidth=1.0, label="centerline")
    ax.plot(track.inner[:, 0], track.inner[:, 1], "k", linewidth=1.0, label="innerwall")
    ax.plot(track.outer[:, 0], track.outer[:, 1], color="dimgray", linewidth=1.0, label="outerwall")

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(float(np.min(speed)), float(np.max(speed)))
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(speed)
    lc.set_linewidth(5)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label="speed [m/s]")
    ax.set_title("Init-Style Kinematic Minimum Lap Time")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "optimal_traj_kin.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(track.s_center, frenet_df["ey_m"], color="tab:blue")
    axes[0].set_ylabel("ey [m]")
    axes[0].set_title("Optimized Frenet Profile")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(track.s_center, frenet_df["epsi_rad"], color="tab:red")
    axes[1].set_ylabel("epsi [rad]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(track.s_center, frenet_df["v_mps"], color="tab:green")
    axes[2].set_ylabel("v [m/s]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(track.s_center, frenet_df["delta_rad"], color="tab:purple", label="delta")
    axes[3].plot(track.s_center, frenet_df["a_mps2"], color="tab:orange", linestyle="--", label="a")
    axes[3].set_ylabel("input")
    axes[3].set_xlabel("s [m]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "optimal_profile_kin.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return xy_txt, frenet_txt
def main() -> None:
    cfg = SolverConfig(
        ns=NS,
        dt=DT,
        track_width_scale=TRACK_WIDTH_SCALE,
        track_width_override_m=TRACK_WIDTH_OVERRIDE_M,
    )

    ensure_dir(OUTPUT_DIR)
    traj_dir = OUTPUT_DIR / "Traj"
    track_width, track_width_std = save_track_assets(CENTERLINE_CSV, BOUNDARY_LEFT_CSV, BOUNDARY_RIGHT_CSV, traj_dir)
    if cfg.track_width_override_m is not None:
        track_width = float(cfg.track_width_override_m)

    track = Track(f"{traj_dir}/")
    track.track_width = track_width
    track.track_width_scale = cfg.track_width_scale
    track.linspace_s(N=cfg.ns)

    x_pred, u_pred, sdot, xy = run_min_laptime(track, cfg)
    xy_txt, frenet_txt = save_outputs(track, x_pred, u_pred, sdot, xy, OUTPUT_DIR)

    summary = {
        "track_length_m": float(track.track_length),
        "track_width_mean_m": float(track_width),
        "track_width_std_m": float(track_width_std),
        "track_width_scale": float(track.track_width_scale),
        "effective_half_width_m": float(track.track_width * track.track_width_scale / 2.0),
        "ns": int(cfg.ns),
        "dt": float(cfg.dt),
        "optimized_lap_time_s": float(x_pred[-1, 3]),
        "ey_min_m": float(np.min(x_pred[:-1, 0])),
        "ey_max_m": float(np.max(x_pred[:-1, 0])),
        "v_min_mps": float(np.min(x_pred[:-1, 2])),
        "v_max_mps": float(np.max(x_pred[:-1, 2])),
    }
    payload = {
        "config": asdict(cfg),
        "summary": summary,
        "saved_files": {
            "optimized_xy_txt": str(xy_txt),
            "optimized_frenet_txt": str(frenet_txt),
            "optimized_xy_csv": str(OUTPUT_DIR / "optimized_traj0.csv"),
            "optimized_frenet_csv": str(OUTPUT_DIR / "optimized_traj0_frenet.csv"),
            "combined_csv": str(OUTPUT_DIR / "optimized_traj0_combined.csv"),
            "traj_plot": str(OUTPUT_DIR / "optimal_traj_kin.png"),
            "profile_plot": str(OUTPUT_DIR / "optimal_profile_kin.png"),
            "track_dir": str(traj_dir),
        },
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print("[Init-Style Kin Minimum Lap Time]")
    print(json.dumps(payload, indent=2))
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
