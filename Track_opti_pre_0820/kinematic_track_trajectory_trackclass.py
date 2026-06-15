from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from Track.track import Track


# Map CSVs live in <repo>/stack_master/maps/final/ ; resolve relative to this file.
_MAP_DIR = Path(__file__).resolve().parent.parent / "stack_master" / "maps" / "final"
DEFAULT_CENTERLINE = _MAP_DIR / "centerline.csv"
DEFAULT_BOUNDARY_LEFT = _MAP_DIR / "boundary_left.csv"
DEFAULT_BOUNDARY_RIGHT = _MAP_DIR / "boundary_right.csv"
DEFAULT_OUTPUT_DIR = Path("outputs/kinematic_track_trajectory_trackclass")


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class KinematicConfig:
    ds: float = 0.05
    wheelbase: float = 0.33
    v_max: float = 4.0
    a_lat_max: float = 3.5
    a_lon_max: float = 2.5
    a_brake_max: float = 4.0
    speed_iterations: int = 30
    min_speed: float = 0.5


def save_track_assets(
    centerline_csv: Path,
    boundary_left_csv: Path,
    boundary_right_csv: Path,
    track_dir: Path,
) -> None:
    ensure_dir(track_dir)

    center_df = pd.read_csv(centerline_csv)
    left_df = pd.read_csv(boundary_left_csv)
    right_df = pd.read_csv(boundary_right_csv)

    center_xy = center_df[["x_m", "y_m"]].to_numpy(dtype=float)
    left_xy = left_df[["x_m", "y_m"]].to_numpy(dtype=float)
    right_xy = right_df[["x_m", "y_m"]].to_numpy(dtype=float)

    np.savetxt(track_dir / "centerline.txt", center_xy, delimiter=",")
    np.savetxt(track_dir / "innerwall.txt", left_xy, delimiter=",")
    np.savetxt(track_dir / "outerwall.txt", right_xy, delimiter=",")

    center_df.to_csv(track_dir / "centerline_source.csv", index=False)
    left_df.to_csv(track_dir / "boundary_left_source.csv", index=False)
    right_df.to_csv(track_dir / "boundary_right_source.csv", index=False)


def build_closed_speed_profile(curvature: np.ndarray, ds: float, cfg: KinematicConfig) -> np.ndarray:
    curv_abs = np.abs(curvature)
    v_curv = np.sqrt(cfg.a_lat_max / np.maximum(curv_abs, 1e-6))
    v_profile = np.clip(v_curv, cfg.min_speed, cfg.v_max)

    for _ in range(cfg.speed_iterations):
        for i in range(len(v_profile)):
            j = (i + 1) % len(v_profile)
            v_allowed = np.sqrt(max(v_profile[i] ** 2 + 2.0 * cfg.a_lon_max * ds, 0.0))
            v_profile[j] = min(v_profile[j], v_allowed)

        for i in range(len(v_profile) - 1, -1, -1):
            j = (i - 1) % len(v_profile)
            v_allowed = np.sqrt(max(v_profile[i] ** 2 + 2.0 * cfg.a_brake_max * ds, 0.0))
            v_profile[j] = min(v_profile[j], v_allowed)

    return np.clip(v_profile, cfg.min_speed, cfg.v_max)


def compute_controls_and_time(
    v_profile: np.ndarray,
    ds: float,
    curvature: np.ndarray,
    cfg: KinematicConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(v_profile)
    sdot = v_profile.copy()
    dt = np.zeros(n, dtype=float)
    accel = np.zeros(n, dtype=float)

    for i in range(n):
        j = (i + 1) % n
        v_avg = max(0.5 * (v_profile[i] + v_profile[j]), 1e-6)
        dt[i] = ds / v_avg
        accel[i] = (v_profile[j] - v_profile[i]) / dt[i]

    delta = np.arctan(cfg.wheelbase * curvature)
    time = np.concatenate(([0.0], np.cumsum(dt[:-1])))
    return sdot, dt, accel, delta, time


def build_reference_from_track(track: Track, cfg: KinematicConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    curvature = track.curv_center.astype(float)
    s = track.s_center.astype(float)
    ds = float(track.d_dist)

    v_profile = build_closed_speed_profile(curvature, ds, cfg)
    sdot, dt, accel, delta, time = compute_controls_and_time(v_profile, ds, curvature, cfg)

    x = np.zeros_like(s)
    y = np.zeros_like(s)
    psi = np.zeros_like(s)
    for i in range(len(s)):
        x[i], y[i], psi[i] = track.local_to_global(np.array([s[i], 0.0, 0.0]))

    frenet_df = pd.DataFrame(
        {
            "s_m": s,
            "ey_m": np.zeros_like(s),
            "epsi_rad": np.zeros_like(s),
            "v_mps": v_profile,
            "a_mps2": accel,
            "delta_rad": delta,
            "sdot_mps": sdot,
            "dt_s": dt,
            "t_s": time,
            "curvature_1pm": curvature,
        }
    )

    xy_df = pd.DataFrame(
        {
            "x_m": x,
            "y_m": y,
            "psi_rad": psi,
            "v_mps": v_profile,
        }
    )

    return frenet_df, xy_df


def global_kinematic_rhs(state: np.ndarray, accel: float, delta: float, wheelbase: float) -> np.ndarray:
    x, y, psi, v = state
    return np.array(
        [
            v * np.cos(psi),
            v * np.sin(psi),
            v * np.tan(delta) / wheelbase,
            accel,
        ],
        dtype=float,
    )


def rollout_global_kinematic(reference_frenet: pd.DataFrame, reference_xy: pd.DataFrame, wheelbase: float) -> pd.DataFrame:
    n = len(reference_frenet)
    states = np.zeros((n, 4), dtype=float)
    states[0] = np.array(
        [
            reference_xy.loc[0, "x_m"],
            reference_xy.loc[0, "y_m"],
            reference_xy.loc[0, "psi_rad"],
            reference_xy.loc[0, "v_mps"],
        ],
        dtype=float,
    )

    for i in range(n - 1):
        accel = float(reference_frenet.loc[i, "a_mps2"])
        delta = float(reference_frenet.loc[i, "delta_rad"])
        dt = float(reference_frenet.loc[i, "dt_s"])

        k1 = global_kinematic_rhs(states[i], accel, delta, wheelbase)
        k2 = global_kinematic_rhs(states[i] + 0.5 * dt * k1, accel, delta, wheelbase)
        k3 = global_kinematic_rhs(states[i] + 0.5 * dt * k2, accel, delta, wheelbase)
        k4 = global_kinematic_rhs(states[i] + dt * k3, accel, delta, wheelbase)
        states[i + 1] = states[i] + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    rollout_df = pd.DataFrame(states, columns=["x_rollout_m", "y_rollout_m", "psi_rollout_rad", "v_rollout_mps"])
    rollout_df["psi_err_rad"] = wrap_to_pi(rollout_df["psi_rollout_rad"].to_numpy() - reference_xy["psi_rad"].to_numpy())
    rollout_df["xy_err_m"] = np.hypot(
        rollout_df["x_rollout_m"].to_numpy() - reference_xy["x_m"].to_numpy(),
        rollout_df["y_rollout_m"].to_numpy() - reference_xy["y_m"].to_numpy(),
    )
    rollout_df["v_err_mps"] = rollout_df["v_rollout_mps"].to_numpy() - reference_xy["v_mps"].to_numpy()
    return rollout_df


def make_colored_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cmap: str,
    linewidth: float,
    linestyle: str = "solid",
    alpha: float = 1.0,
    norm: Normalize | None = None,
) -> LineCollection:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    line.set_array(values[:-1])
    line.set_linestyle(linestyle)
    ax.add_collection(line)
    return line


def save_artifacts(
    track: Track,
    reference_frenet: pd.DataFrame,
    reference_xy: pd.DataFrame,
    rollout_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    combined_df = pd.concat([reference_frenet, reference_xy, rollout_df], axis=1)
    combined_df.to_csv(output_dir / "kinematic_reference_combined.csv", index=False)
    reference_frenet.to_csv(output_dir / "kinematic_reference_frenet.csv", index=False)
    reference_xy.to_csv(output_dir / "kinematic_reference_xy.csv", index=False)
    rollout_df.to_csv(output_dir / "kinematic_rollout.csv", index=False)

    np.savetxt(output_dir / "kinematic_reference_frenet.txt", reference_frenet.to_numpy(), delimiter=",")
    np.savetxt(output_dir / "kinematic_reference_xy.txt", reference_xy.to_numpy(), delimiter=",")

    speed = reference_xy["v_mps"].to_numpy()
    norm = Normalize(vmin=float(speed.min()), vmax=float(speed.max()))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("equal")
    ax.plot(track.center[:, 0], track.center[:, 1], "--k", linewidth=1.0, label="centerline")
    ax.plot(track.inner[:, 0], track.inner[:, 1], color="black", linewidth=1.0, label="innerwall")
    ax.plot(track.outer[:, 0], track.outer[:, 1], color="dimgray", linewidth=1.0, label="outerwall")
    colored_ref_only = make_colored_line(
        ax=ax,
        x=reference_xy["x_m"].to_numpy(),
        y=reference_xy["y_m"].to_numpy(),
        values=speed,
        cmap="viridis",
        norm=norm,
        linewidth=4.0,
    )
    ax.scatter(reference_xy.loc[0, "x_m"], reference_xy.loc[0, "y_m"], color="tab:green", s=50, label="start")
    ax.set_title("Reference Trajectory Only (Speed Colormap)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    colorbar = fig.colorbar(colored_ref_only, ax=ax)
    colorbar.set_label("reference speed [m/s]")
    fig.tight_layout()
    fig.savefig(output_dir / "reference_only_speed_colormap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("equal")
    ax.plot(track.center[:, 0], track.center[:, 1], "--k", linewidth=1.0)
    ax.plot(track.inner[:, 0], track.inner[:, 1], color="black", linewidth=1.0)
    ax.plot(track.outer[:, 0], track.outer[:, 1], color="dimgray", linewidth=1.0)

    colored_ref = make_colored_line(
        ax=ax,
        x=reference_xy["x_m"].to_numpy(),
        y=reference_xy["y_m"].to_numpy(),
        values=speed,
        cmap="viridis",
        norm=norm,
        linewidth=4.0,
    )
    make_colored_line(
        ax=ax,
        x=rollout_df["x_rollout_m"].to_numpy(),
        y=rollout_df["y_rollout_m"].to_numpy(),
        values=speed,
        cmap="viridis",
        norm=norm,
        linewidth=2.0,
        linestyle="dashed",
        alpha=0.9,
    )

    ax.scatter(reference_xy.loc[0, "x_m"], reference_xy.loc[0, "y_m"], color="tab:green", s=50)
    ax.set_title("Track-Class Kinematic Trajectory (Speed Colormap)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.0, label="centerline"),
        Line2D([0], [0], color="black", linewidth=1.0, label="innerwall"),
        Line2D([0], [0], color="dimgray", linewidth=1.0, label="outerwall"),
        Line2D([0], [0], color=plt.cm.viridis(0.8), linewidth=4.0, label="reference"),
        Line2D([0], [0], color=plt.cm.viridis(0.5), linewidth=2.0, linestyle="--", label="rollout"),
        Line2D([0], [0], marker="o", color="tab:green", linestyle="None", markersize=7, label="start"),
    ]
    ax.legend(handles=legend_handles, loc="best")

    colorbar = fig.colorbar(colored_ref, ax=ax)
    colorbar.set_label("reference speed [m/s]")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_map_speed_colormap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(reference_frenet["t_s"], reference_frenet["v_mps"], color="tab:blue")
    axes[0].set_ylabel("v [m/s]")
    axes[0].set_title("Track-Class Kinematic Profile")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(reference_frenet["t_s"], reference_frenet["curvature_1pm"], color="tab:red")
    axes[1].set_ylabel("curvature [1/m]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(reference_frenet["t_s"], reference_frenet["delta_rad"], color="tab:purple")
    axes[2].set_ylabel("delta [rad]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(reference_frenet["t_s"], reference_frenet["a_mps2"], color="tab:green", label="a_ref")
    axes[3].plot(reference_frenet["t_s"], rollout_df["xy_err_m"], color="tab:orange", linestyle="--", label="xy error")
    axes[3].set_ylabel("a / error")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary(track: Track, cfg: KinematicConfig, reference_frenet: pd.DataFrame, rollout_df: pd.DataFrame, n_nodes: int) -> dict[str, float | int]:
    lap_time = float(reference_frenet["dt_s"].sum())
    return {
        "track_length_m": float(track.track_length),
        "n_nodes": int(n_nodes),
        "d_dist_m": float(track.d_dist),
        "wheelbase_m": float(cfg.wheelbase),
        "v_max_mps": float(reference_frenet["v_mps"].max()),
        "v_mean_mps": float(reference_frenet["v_mps"].mean()),
        "lap_time_s": lap_time,
        "max_abs_curvature_1pm": float(np.abs(reference_frenet["curvature_1pm"]).max()),
        "max_abs_delta_rad": float(np.abs(reference_frenet["delta_rad"]).max()),
        "rollout_rmse_xy_m": float(np.sqrt(np.mean(rollout_df["xy_err_m"].to_numpy() ** 2))),
        "rollout_rmse_psi_rad": float(np.sqrt(np.mean(rollout_df["psi_err_rad"].to_numpy() ** 2))),
        "rollout_rmse_v_mps": float(np.sqrt(np.mean(rollout_df["v_err_mps"].to_numpy() ** 2))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track-class based kinematic trajectory generator.")
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--boundary-left", type=Path, default=DEFAULT_BOUNDARY_LEFT)
    parser.add_argument("--boundary-right", type=Path, default=DEFAULT_BOUNDARY_RIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ds", type=float, default=0.05, help="Target spatial step [m] for Track.linspace_s.")
    parser.add_argument("--wheelbase", type=float, default=0.28)
    parser.add_argument("--v-max", type=float, default=4.0)
    parser.add_argument("--a-lat-max", type=float, default=3.5)
    parser.add_argument("--a-lon-max", type=float, default=2.5)
    parser.add_argument("--a-brake-max", type=float, default=4.0)
    parser.add_argument("--min-speed", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = KinematicConfig(
        ds=args.ds,
        wheelbase=args.wheelbase,
        v_max=args.v_max,
        a_lat_max=args.a_lat_max,
        a_lon_max=args.a_lon_max,
        a_brake_max=args.a_brake_max,
        min_speed=args.min_speed,
    )

    ensure_dir(args.output_dir)
    track_dir = args.output_dir / "track_assets"
    save_track_assets(args.centerline, args.boundary_left, args.boundary_right, track_dir)

    track = Track(f"{track_dir}/")
    n_nodes = max(50, int(round(track.track_length / cfg.ds)))
    track.linspace_s(N=n_nodes)

    reference_frenet, reference_xy = build_reference_from_track(track, cfg)
    rollout_df = rollout_global_kinematic(reference_frenet, reference_xy, cfg.wheelbase)
    save_artifacts(track, reference_frenet, reference_xy, rollout_df, args.output_dir)

    summary = build_summary(track, cfg, reference_frenet, rollout_df, n_nodes)
    payload = {
        "config": asdict(cfg),
        "summary": summary,
        "saved_files": {
            "track_assets": str(track_dir),
            "reference_frenet_csv": str(args.output_dir / "kinematic_reference_frenet.csv"),
            "reference_xy_csv": str(args.output_dir / "kinematic_reference_xy.csv"),
            "rollout_csv": str(args.output_dir / "kinematic_rollout.csv"),
            "combined_csv": str(args.output_dir / "kinematic_reference_combined.csv"),
            "map_png": str(args.output_dir / "trajectory_map_speed_colormap.png"),
            "profile_png": str(args.output_dir / "trajectory_profile.png"),
        },
    }

    summary_path = args.output_dir / "trajectory_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print("[Track-Class Kinematic Trajectory]")
    print(json.dumps(payload, indent=2))
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
