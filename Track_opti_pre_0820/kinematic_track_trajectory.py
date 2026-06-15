from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


# Map CSVs live in <repo>/stack_master/maps/final/ ; resolve relative to this file
# so it works regardless of the machine / current working directory.
_MAP_DIR = Path(__file__).resolve().parent.parent / "stack_master" / "maps" / "final"
DEFAULT_CENTERLINE = _MAP_DIR / "centerline.csv"
DEFAULT_BOUNDARY_LEFT = _MAP_DIR / "boundary_left.csv"
DEFAULT_BOUNDARY_RIGHT = _MAP_DIR / "boundary_right.csv"
DEFAULT_OUTPUT_DIR = Path("outputs/kinematic_track_trajectory")


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_odd_window(n_points: int, target: int = 31) -> int:
    if n_points < 5:
        return 0
    window = min(target, n_points if n_points % 2 == 1 else n_points - 1)
    if window < 5:
        return 0
    return window


def smooth_periodic(values: np.ndarray, target_window: int = 31, polyorder: int = 3) -> np.ndarray:
    window = choose_odd_window(len(values), target_window)
    if window == 0:
        return values.copy()

    pad = window // 2
    extended = np.concatenate([values[-pad:], values, values[:pad]])
    smoothed = savgol_filter(extended, window_length=window, polyorder=min(polyorder, window - 2), mode="interp")
    return smoothed[pad:-pad]


@dataclass
class KinematicConfig:
    ds: float = 0.05
    wheelbase: float = 0.28
    v_max: float = 4.0
    a_lat_max: float = 3.5
    a_lon_max: float = 2.5
    a_brake_max: float = 4.0
    speed_iterations: int = 30
    min_speed: float = 0.5


class ClosedTrackSpline:
    def __init__(self, centerline_csv: Path) -> None:
        center_df = pd.read_csv(centerline_csv)
        xy = center_df[["x_m", "y_m"]].to_numpy(dtype=float)
        widths_r = center_df["w_tr_right_m"].to_numpy(dtype=float)
        widths_l = center_df["w_tr_left_m"].to_numpy(dtype=float)

        # Drop consecutive duplicate points (ds≈0). Raw map CSVs contain repeated
        # samples, which would make the spline knots non-monotonic and crash
        # CubicSpline ("x must be strictly increasing").
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        keep = np.concatenate(([True], seg > 1e-6))
        xy = xy[keep]
        widths_r = widths_r[keep]
        widths_l = widths_l[keep]

        if np.linalg.norm(xy[0] - xy[-1]) > 1e-9:
            xy = np.vstack([xy, xy[0]])
            widths_r = np.concatenate([widths_r, widths_r[:1]])
            widths_l = np.concatenate([widths_l, widths_l[:1]])

        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        self.s_nodes = np.concatenate(([0.0], np.cumsum(ds)))
        self.length = float(self.s_nodes[-1])
        self.centerline_xy = xy
        self.x_spline = CubicSpline(self.s_nodes, xy[:, 0], bc_type="periodic")
        self.y_spline = CubicSpline(self.s_nodes, xy[:, 1], bc_type="periodic")
        self.wr_spline = CubicSpline(self.s_nodes, widths_r, bc_type="periodic")
        self.wl_spline = CubicSpline(self.s_nodes, widths_l, bc_type="periodic")

    def wrap_s(self, s: np.ndarray | float) -> np.ndarray | float:
        return np.mod(s, self.length)

    def sample(self, s: np.ndarray) -> dict[str, np.ndarray]:
        s_wrapped = self.wrap_s(s)
        x = self.x_spline(s_wrapped)
        y = self.y_spline(s_wrapped)
        dx = self.x_spline(s_wrapped, 1)
        dy = self.y_spline(s_wrapped, 1)
        ddx = self.x_spline(s_wrapped, 2)
        ddy = self.y_spline(s_wrapped, 2)
        psi = np.unwrap(np.arctan2(dy, dx))
        curvature = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)

        return {
            "s": s_wrapped,
            "x": x,
            "y": y,
            "psi": psi,
            "curvature": curvature,
            "w_tr_right_m": self.wr_spline(s_wrapped),
            "w_tr_left_m": self.wl_spline(s_wrapped),
        }


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


def compute_time_and_accel(v_profile: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(v_profile)
    dt = np.zeros(n, dtype=float)
    accel = np.zeros(n, dtype=float)

    for i in range(n):
        j = (i + 1) % n
        v_avg = max(0.5 * (v_profile[i] + v_profile[j]), 1e-6)
        dt[i] = ds / v_avg
        accel[i] = (v_profile[j] - v_profile[i]) / dt[i]

    time = np.concatenate(([0.0], np.cumsum(dt[:-1])))
    return time, dt, accel


def generate_reference_trajectory(track: ClosedTrackSpline, cfg: KinematicConfig) -> pd.DataFrame:
    s_ref = np.arange(0.0, track.length, cfg.ds)
    sampled = track.sample(s_ref)
    curvature = smooth_periodic(sampled["curvature"], target_window=41)
    psi_unwrapped = np.unwrap(sampled["psi"])
    v_profile = build_closed_speed_profile(curvature, cfg.ds, cfg)
    time, dt, accel = compute_time_and_accel(v_profile, cfg.ds)
    yaw_rate = v_profile * curvature
    delta = np.arctan(cfg.wheelbase * curvature)

    return pd.DataFrame(
        {
            "s_m": sampled["s"],
            "x_m": sampled["x"],
            "y_m": sampled["y"],
            "psi_unwrapped_rad": psi_unwrapped,
            "psi_rad": wrap_to_pi(psi_unwrapped),
            "curvature_1pm": curvature,
            "v_ref_mps": v_profile,
            "a_ref_mps2": accel,
            "yaw_rate_ref_radps": yaw_rate,
            "delta_ref_rad": delta,
            "dt_s": dt,
            "t_s": time,
            "ey_ref_m": np.zeros_like(s_ref),
            "epsi_ref_rad": np.zeros_like(s_ref),
            "w_tr_right_m": sampled["w_tr_right_m"],
            "w_tr_left_m": sampled["w_tr_left_m"],
        }
    )


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


def rollout_kinematic(reference_df: pd.DataFrame, wheelbase: float) -> pd.DataFrame:
    n = len(reference_df)
    states = np.zeros((n, 4), dtype=float)
    states[0] = np.array(
        [
            reference_df.loc[0, "x_m"],
            reference_df.loc[0, "y_m"],
            reference_df.loc[0, "psi_unwrapped_rad"],
            reference_df.loc[0, "v_ref_mps"],
        ],
        dtype=float,
    )

    for i in range(n - 1):
        accel = float(reference_df.loc[i, "a_ref_mps2"])
        delta = float(reference_df.loc[i, "delta_ref_rad"])
        dt = float(reference_df.loc[i, "dt_s"])

        k1 = global_kinematic_rhs(states[i], accel, delta, wheelbase)
        k2 = global_kinematic_rhs(states[i] + 0.5 * dt * k1, accel, delta, wheelbase)
        k3 = global_kinematic_rhs(states[i] + 0.5 * dt * k2, accel, delta, wheelbase)
        k4 = global_kinematic_rhs(states[i] + dt * k3, accel, delta, wheelbase)
        states[i + 1] = states[i] + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    rollout_df = pd.DataFrame(states, columns=["x_rollout_m", "y_rollout_m", "psi_rollout_unwrapped_rad", "v_rollout_mps"])
    rollout_df["psi_rollout_rad"] = wrap_to_pi(rollout_df["psi_rollout_unwrapped_rad"].to_numpy())
    rollout_df["psi_err_rad"] = wrap_to_pi(
        rollout_df["psi_rollout_rad"].to_numpy() - reference_df["psi_rad"].to_numpy()
    )
    rollout_df["xy_err_m"] = np.hypot(
        rollout_df["x_rollout_m"].to_numpy() - reference_df["x_m"].to_numpy(),
        rollout_df["y_rollout_m"].to_numpy() - reference_df["y_m"].to_numpy(),
    )
    rollout_df["v_err_mps"] = rollout_df["v_rollout_mps"].to_numpy() - reference_df["v_ref_mps"].to_numpy()
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


def save_outputs(
    reference_df: pd.DataFrame,
    rollout_df: pd.DataFrame,
    boundary_left_csv: Path,
    boundary_right_csv: Path,
    output_dir: Path,
) -> None:
    left_df = pd.read_csv(boundary_left_csv)
    right_df = pd.read_csv(boundary_right_csv)

    combined_df = pd.concat([reference_df, rollout_df], axis=1)
    combined_df.to_csv(output_dir / "kinematic_reference_trajectory.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(left_df["x_m"], left_df["y_m"], color="black", linewidth=1.0, label="boundary left")
    ax.plot(right_df["x_m"], right_df["y_m"], color="dimgray", linewidth=1.0, label="boundary right")

    speed_values = reference_df["v_ref_mps"].to_numpy()
    speed_norm = Normalize(vmin=float(speed_values.min()), vmax=float(speed_values.max()))
    speed_line = make_colored_line(
        ax=ax,
        x=reference_df["x_m"].to_numpy(),
        y=reference_df["y_m"].to_numpy(),
        values=speed_values,
        cmap="viridis",
        norm=speed_norm,
        linewidth=3.0,
    )
    make_colored_line(
        ax=ax,
        x=rollout_df["x_rollout_m"].to_numpy(),
        y=rollout_df["y_rollout_m"].to_numpy(),
        values=speed_values,
        cmap="viridis",
        norm=speed_norm,
        linewidth=2.0,
        linestyle="dashed",
        alpha=0.9,
    )
    ax.scatter(reference_df.loc[0, "x_m"], reference_df.loc[0, "y_m"], color="tab:green", s=40, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Kinematic Track Trajectory (Speed Colormap)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    colorbar = fig.colorbar(speed_line, ax=ax)
    colorbar.set_label("reference speed [m/s]")
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.0, label="boundary left"),
        Line2D([0], [0], color="dimgray", linewidth=1.0, label="boundary right"),
        Line2D([0], [0], color=plt.cm.viridis(0.8), linewidth=3.0, label="reference"),
        Line2D([0], [0], color=plt.cm.viridis(0.5), linewidth=2.0, linestyle="--", label="kinematic rollout"),
        Line2D([0], [0], marker="o", color="tab:green", linestyle="None", markersize=7, label="start"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(reference_df["t_s"], reference_df["v_ref_mps"], color="tab:blue")
    axes[0].set_ylabel("v [m/s]")
    axes[0].set_title("Speed Profile")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(reference_df["t_s"], reference_df["curvature_1pm"], color="tab:red")
    axes[1].set_ylabel("curvature [1/m]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(reference_df["t_s"], reference_df["delta_ref_rad"], color="tab:purple")
    axes[2].set_ylabel("delta [rad]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(reference_df["t_s"], reference_df["a_ref_mps2"], color="tab:green", label="a_ref")
    axes[3].plot(reference_df["t_s"], rollout_df["xy_err_m"], color="tab:orange", linestyle="--", label="xy error")
    axes[3].set_ylabel("a / error")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_global_waypoints(reference_df: pd.DataFrame, output_path: Path, source_ds: float, target_ds: float) -> int:
    """Export the reference trajectory in the stack's global_waypoints.csv format.

    Columns: x_m, y_m, w_tr_right_m, w_tr_left_m, psi_rad, kappa_radpm, vx_mps
    (curvature_1pm -> kappa_radpm, v_ref_mps -> vx_mps). Resamples from the dense
    source_ds grid to target_ds by striding, matching the ~0.25 m spacing the
    waypoint_publisher / Pure Pursuit expect.
    """
    stride = max(1, int(round(target_ds / source_ds)))
    sub = reference_df.iloc[::stride]
    out = pd.DataFrame(
        {
            "x_m": sub["x_m"].to_numpy(),
            "y_m": sub["y_m"].to_numpy(),
            "w_tr_right_m": sub["w_tr_right_m"].to_numpy(),
            "w_tr_left_m": sub["w_tr_left_m"].to_numpy(),
            "psi_rad": sub["psi_rad"].to_numpy(),
            "kappa_radpm": sub["curvature_1pm"].to_numpy(),
            "vx_mps": sub["v_ref_mps"].to_numpy(),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, float_format="%.6f")
    return len(out)


def build_summary(track: ClosedTrackSpline, cfg: KinematicConfig, reference_df: pd.DataFrame, rollout_df: pd.DataFrame) -> dict[str, float]:
    total_time = float(reference_df["dt_s"].sum())
    return {
        "track_length_m": float(track.length),
        "n_samples": int(len(reference_df)),
        "ds_m": float(cfg.ds),
        "wheelbase_m": float(cfg.wheelbase),
        "v_max_mps": float(reference_df["v_ref_mps"].max()),
        "v_mean_mps": float(reference_df["v_ref_mps"].mean()),
        "lap_time_s": total_time,
        "max_abs_curvature_1pm": float(np.abs(reference_df["curvature_1pm"]).max()),
        "max_abs_delta_rad": float(np.abs(reference_df["delta_ref_rad"]).max()),
        "rollout_rmse_xy_m": float(np.sqrt(np.mean(rollout_df["xy_err_m"].to_numpy() ** 2))),
        "rollout_rmse_psi_rad": float(np.sqrt(np.mean(rollout_df["psi_err_rad"].to_numpy() ** 2))),
        "rollout_rmse_v_mps": float(np.sqrt(np.mean(rollout_df["v_err_mps"].to_numpy() ** 2))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a kinematic reference trajectory from centerline and boundaries.")
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--boundary-left", type=Path, default=DEFAULT_BOUNDARY_LEFT)
    parser.add_argument("--boundary-right", type=Path, default=DEFAULT_BOUNDARY_RIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--global-waypoints-out", type=Path, default=_MAP_DIR / "global_waypoints_trackopti.csv",
                        help="Where to write the stack-format global_waypoints CSV.")
    parser.add_argument("--waypoints-ds", type=float, default=0.25, help="Point spacing for the exported global_waypoints [m].")
    parser.add_argument("--ds", type=float, default=0.05, help="Spatial sampling step along the track [m].")
    parser.add_argument("--wheelbase", type=float, default=0.33, help="Kinematic bicycle wheelbase [m].")
    parser.add_argument("--v-max", type=float, default=4.0, help="Maximum reference speed [m/s].")
    parser.add_argument("--a-lat-max", type=float, default=3.5, help="Lateral acceleration limit [m/s^2].")
    parser.add_argument("--a-lon-max", type=float, default=2.5, help="Forward acceleration limit [m/s^2].")
    parser.add_argument("--a-brake-max", type=float, default=4.0, help="Braking limit [m/s^2].")
    parser.add_argument("--min-speed", type=float, default=0.5, help="Minimum reference speed [m/s].")
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
    track = ClosedTrackSpline(args.centerline)
    reference_df = generate_reference_trajectory(track, cfg)
    rollout_df = rollout_kinematic(reference_df, cfg.wheelbase)
    save_outputs(reference_df, rollout_df, args.boundary_left, args.boundary_right, args.output_dir)
    n_wpnts = save_global_waypoints(reference_df, args.global_waypoints_out, cfg.ds, args.waypoints_ds)

    summary = build_summary(track, cfg, reference_df, rollout_df)
    payload = {
        "config": asdict(cfg),
        "summary": summary,
    }

    summary_path = args.output_dir / "trajectory_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print("[Kinematic Trajectory]")
    print(json.dumps(payload, indent=2))
    print(f"trajectory_csv={args.output_dir / 'kinematic_reference_trajectory.csv'}")
    print(f"global_waypoints_csv={args.global_waypoints_out}  ({n_wpnts} pts @ {args.waypoints_ds} m)")
    print(f"map_png={args.output_dir / 'trajectory_map.png'}")
    print(f"profile_png={args.output_dir / 'trajectory_profile.png'}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
