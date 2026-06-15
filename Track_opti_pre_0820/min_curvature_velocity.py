from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.signal import savgol_filter


# =========================
# User Settings
# =========================
CENTERLINE_CSV    = Path("outputs/kinematic_track_trajectory_trackclass/track_assets/centerline_source.csv")
BOUNDARY_LEFT_CSV = Path("outputs/kinematic_track_trajectory_trackclass/track_assets/boundary_left_source.csv")
BOUNDARY_RIGHT_CSV= Path("outputs/kinematic_track_trajectory_trackclass/track_assets/boundary_right_source.csv")
OUTPUT_DIR        = Path("outputs/min_curvature_velocity")

NS             = 300   # discretization points
V_MAX          = 6.0   # max speed [m/s]
V_MIN          = 0.5   # min speed [m/s]
A_LAT_MAX      = 3.5   # lateral accel limit [m/s²]
A_LON_MAX      = 2.5   # longitudinal accel limit [m/s²]
A_BRAKE_MAX    = 4.0   # braking limit [m/s²]
TRACK_WIDTH_SCALE = 0.3  # safety margin (0–1)

SMOOTH_WINDOW    = 51
SMOOTH_POLYORDER = 3


@dataclass
class OptConfig:
    ns: int              = NS
    v_max: float         = V_MAX
    v_min: float         = V_MIN
    a_lat_max: float     = A_LAT_MAX
    a_lon_max: float     = A_LON_MAX
    a_brake_max: float   = A_BRAKE_MAX
    track_width_scale: float = TRACK_WIDTH_SCALE
    smooth_window: int   = SMOOTH_WINDOW
    smooth_polyorder: int = SMOOTH_POLYORDER


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def smooth_periodic(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    n = len(values)
    w = min(window if window % 2 == 1 else window + 1, n if n % 2 == 1 else n - 1)
    w = max(w, 5)
    pad = w // 2
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    smoothed = savgol_filter(ext, window_length=w, polyorder=min(polyorder, w - 2))
    return smoothed[pad:-pad]


def load_and_smooth_centerline(centerline_csv: Path, window: int, polyorder: int) -> pd.DataFrame:
    df = pd.read_csv(centerline_csv)
    xy = df[["x_m", "y_m"]].to_numpy(dtype=float)
    xy[:, 0] = smooth_periodic(xy[:, 0], window, polyorder)
    xy[:, 1] = smooth_periodic(xy[:, 1], window, polyorder)
    df = df.copy()
    df["x_m"] = xy[:, 0]
    df["y_m"] = xy[:, 1]
    return df


class ClosedTrackSpline:
    """Periodic cubic spline for a closed track centerline."""

    def __init__(self, center_df: pd.DataFrame):
        xy  = center_df[["x_m", "y_m"]].to_numpy(dtype=float)
        w_r = center_df["w_tr_right_m"].to_numpy(dtype=float)
        w_l = center_df["w_tr_left_m"].to_numpy(dtype=float)

        if np.linalg.norm(xy[0] - xy[-1]) > 1e-9:
            xy  = np.vstack([xy, xy[0]])
            w_r = np.append(w_r, w_r[0])
            w_l = np.append(w_l, w_l[0])

        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        self.s_nodes = np.concatenate([[0.0], np.cumsum(ds)])
        self.length  = float(self.s_nodes[-1])

        self.cx  = CubicSpline(self.s_nodes, xy[:, 0], bc_type="periodic")
        self.cy  = CubicSpline(self.s_nodes, xy[:, 1], bc_type="periodic")
        self.cwr = CubicSpline(self.s_nodes, w_r,      bc_type="periodic")
        self.cwl = CubicSpline(self.s_nodes, w_l,      bc_type="periodic")

    def sample(self, s: np.ndarray) -> dict:
        s_w  = s % self.length
        dx   = self.cx(s_w, 1);  dy  = self.cy(s_w, 1)
        ddx  = self.cx(s_w, 2);  ddy = self.cy(s_w, 2)
        psi  = np.unwrap(np.arctan2(dy, dx))
        denom = np.maximum((dx**2 + dy**2)**1.5, 1e-9)
        kappa = (dx * ddy - dy * ddx) / denom
        return {
            "x": self.cx(s_w),  "y": self.cy(s_w),
            "psi": psi,         "kappa": kappa,
            "w_r": self.cwr(s_w), "w_l": self.cwl(s_w),
            # left-pointing unit normal
            "nx": -np.sin(psi), "ny":  np.cos(psi),
        }


def solve_min_curvature(
    kappa_c: np.ndarray,
    w_right: np.ndarray,
    w_left: np.ndarray,
    ds: float,
    scale: float,
) -> np.ndarray:
    """QP: minimize Σ κ_path² w.r.t. lateral offset d (periodic).

    κ_path_i ≈ κ_c_i + (d_{i+1} - 2d_i + d_{i-1}) / ds²
    Gradient: ∂J/∂d_j = 2/ds² * (κ_{j-1} - 2κ_j + κ_{j+1})
    """
    N = len(kappa_c)

    def kappa_path(d: np.ndarray) -> np.ndarray:
        d_pp = (np.roll(d, -1) - 2.0 * d + np.roll(d, 1)) / ds**2
        return kappa_c + d_pp

    def obj(d: np.ndarray) -> float:
        return float(np.sum(kappa_path(d) ** 2))

    def grad(d: np.ndarray) -> np.ndarray:
        kp = kappa_path(d)
        return 2.0 / ds**2 * (np.roll(kp, 1) - 2.0 * kp + np.roll(kp, -1))

    bounds = [(-w_left[i] * scale, w_right[i] * scale) for i in range(N)]

    result = minimize(
        obj, np.zeros(N), jac=grad, method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-9},
    )
    return result.x


def build_optimal_path(center: dict, d_opt: np.ndarray) -> dict:
    """Convert lateral offset → global x, y, psi, kappa via periodic cubic spline."""
    x_opt = center["x"] + d_opt * center["nx"]
    y_opt = center["y"] + d_opt * center["ny"]

    # arc-length of optimal path
    seg   = np.sqrt(np.diff(x_opt, append=x_opt[0])**2 +
                    np.diff(y_opt, append=y_opt[0])**2)
    s_opt = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    total = float(np.sum(seg))

    # fit periodic spline for smooth psi and kappa
    s_cl  = np.append(s_opt, total)
    cx    = CubicSpline(s_cl, np.append(x_opt, x_opt[0]), bc_type="periodic")
    cy    = CubicSpline(s_cl, np.append(y_opt, y_opt[0]), bc_type="periodic")

    dx  = cx(s_opt, 1);  dy  = cy(s_opt, 1)
    ddx = cx(s_opt, 2);  ddy = cy(s_opt, 2)
    psi   = np.unwrap(np.arctan2(dy, dx))
    denom = np.maximum((dx**2 + dy**2)**1.5, 1e-9)
    kappa = (dx * ddy - dy * ddx) / denom

    return {"x": x_opt, "y": y_opt, "psi": psi, "kappa": kappa,
            "s": s_opt, "length": total}


def compute_velocity_profile(kappa: np.ndarray, s: np.ndarray, cfg: OptConfig) -> np.ndarray:
    """Forward/backward pass on lateral-accel-limited speed profile."""
    ds_arr = np.concatenate([np.diff(s), [s[-1] + (s[1] - s[0]) - s[-1] + s[0]]])
    ds_arr = np.abs(ds_arr)

    v_curv = np.sqrt(cfg.a_lat_max / np.maximum(np.abs(kappa), 1e-6))
    v = np.clip(v_curv, cfg.v_min, cfg.v_max)

    for _ in range(50):
        for i in range(len(v)):
            j = (i + 1) % len(v)
            v[j] = min(v[j], np.sqrt(max(v[i]**2 + 2.0 * cfg.a_lon_max * ds_arr[i], 0.0)))
        for i in range(len(v) - 1, -1, -1):
            j = (i - 1) % len(v)
            v[j] = min(v[j], np.sqrt(max(v[i]**2 + 2.0 * cfg.a_brake_max * ds_arr[i], 0.0)))

    return np.clip(v, cfg.v_min, cfg.v_max)


def run_optimization(center_df: pd.DataFrame, cfg: OptConfig) -> dict:
    track    = ClosedTrackSpline(center_df)
    s_uni    = np.linspace(0.0, track.length, cfg.ns, endpoint=False)
    ds       = float(s_uni[1] - s_uni[0])
    center   = track.sample(s_uni)

    kappa_c  = smooth_periodic(center["kappa"], cfg.smooth_window, cfg.smooth_polyorder)
    w_r, w_l = center["w_r"], center["w_l"]

    print("  Solving minimum curvature QP...")
    d_opt  = solve_min_curvature(kappa_c, w_r, w_l, ds, cfg.track_width_scale)
    path   = build_optimal_path(center, d_opt)

    kappa_smooth = smooth_periodic(path["kappa"], cfg.smooth_window, cfg.smooth_polyorder)
    print("  Computing velocity profile...")
    v = compute_velocity_profile(kappa_smooth, path["s"], cfg)

    ds_path = np.concatenate([np.diff(path["s"]), [path["length"] - path["s"][-1]]])
    v_avg   = 0.5 * (v + np.roll(v, -1))
    dt      = np.where(v_avg > 1e-6, np.abs(ds_path) / v_avg, 0.0)
    t       = np.concatenate([[0.0], np.cumsum(dt[:-1])])
    a       = np.gradient(v, np.maximum(t, 1e-9))

    w_right_path = np.interp(path["s"] % track.length, s_uni, w_r)
    w_left_path  = np.interp(path["s"] % track.length, s_uni, w_l)

    return {
        "x": path["x"], "y": path["y"],
        "psi": path["psi"], "kappa": kappa_smooth,
        "v": v, "a": a, "t": t,
        "w_right": w_right_path, "w_left": w_left_path,
        "s": path["s"], "d_opt": d_opt,
        "lap_time": float(np.sum(dt)),
        "track_length": path["length"],
    }


def save_outputs(
    result: dict,
    boundary_left_csv: Path,
    boundary_right_csv: Path,
    centerline_csv: Path,
    output_dir: Path,
) -> Path:
    left_df  = pd.read_csv(boundary_left_csv)
    right_df = pd.read_csv(boundary_right_csv)

    # global_waypoint.csv (same format as global_waypoints1.csv)
    global_waypoint_df = pd.DataFrame({
        "x_m":          result["x"],
        "y_m":          result["y"],
        "w_tr_right_m": result["w_right"],
        "w_tr_left_m":  result["w_left"],
        "psi_rad":      result["psi"],
        "kappa_radpm":  result["kappa"],
        "vx_mps":       result["v"],
    })
    waypoint_path = output_dir / "global_waypoint.csv"
    global_waypoint_df.to_csv(waypoint_path, index=False)

    # detailed CSV
    detail_df = pd.DataFrame({
        "s_m":          result["s"],
        "x_m":          result["x"],
        "y_m":          result["y"],
        "psi_rad":      result["psi"],
        "kappa_radpm":  result["kappa"],
        "vx_mps":       result["v"],
        "a_mps2":       result["a"],
        "t_s":          result["t"],
        "w_tr_right_m": result["w_right"],
        "w_tr_left_m":  result["w_left"],
        "d_opt_m":      result["d_opt"],
    })
    detail_df.to_csv(output_dir / "min_curv_velocity_detail.csv", index=False)

    # trajectory plot
    x, y, v = result["x"], result["y"], result["v"]
    pts  = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("equal")
    ax.plot(left_df["x_m"],  left_df["y_m"],  "k",         linewidth=1.0, label="inner")
    ax.plot(right_df["x_m"], right_df["y_m"],  color="dimgray", linewidth=1.0, label="outer")
    norm = plt.Normalize(float(v.min()), float(v.max()))
    lc   = LineCollection(segs, cmap="viridis", norm=norm)
    lc.set_array(v[:-1]); lc.set_linewidth(5)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="speed [m/s]")
    ax.set_title("Min-Curvature + Velocity Optimized Path")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(output_dir / "min_curv_traj.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # profile plot
    s = result["s"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(s, v,               color="tab:blue");  axes[0].set_ylabel("v [m/s]")
    axes[0].set_title("Min-Curvature Velocity Profile"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(s, result["kappa"], color="tab:red");   axes[1].set_ylabel("kappa [1/m]")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(s, result["a"],     color="tab:green"); axes[2].set_ylabel("a [m/s²]")
    axes[2].set_xlabel("s [m]");                         axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "min_curv_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return waypoint_path


def main() -> None:
    cfg = OptConfig(
        ns=NS, v_max=V_MAX, v_min=V_MIN,
        a_lat_max=A_LAT_MAX, a_lon_max=A_LON_MAX, a_brake_max=A_BRAKE_MAX,
        track_width_scale=TRACK_WIDTH_SCALE,
        smooth_window=SMOOTH_WINDOW, smooth_polyorder=SMOOTH_POLYORDER,
    )

    ensure_dir(OUTPUT_DIR)
    center_df = load_and_smooth_centerline(CENTERLINE_CSV, cfg.smooth_window, cfg.smooth_polyorder)

    print("[Min-Curvature + Velocity Optimization] running...")
    result = run_optimization(center_df, cfg)

    waypoint_path = save_outputs(
        result, BOUNDARY_LEFT_CSV, BOUNDARY_RIGHT_CSV, CENTERLINE_CSV, OUTPUT_DIR
    )

    summary = {
        "track_length_m":  result["track_length"],
        "lap_time_s":      result["lap_time"],
        "v_min_mps":       float(result["v"].min()),
        "v_max_mps":       float(result["v"].max()),
        "kappa_max_radpm": float(np.abs(result["kappa"]).max()),
        "d_opt_min_m":     float(result["d_opt"].min()),
        "d_opt_max_m":     float(result["d_opt"].max()),
    }
    payload = {
        "config": asdict(cfg),
        "summary": summary,
        "saved_files": {
            "global_waypoint_csv": str(waypoint_path),
            "detail_csv":  str(OUTPUT_DIR / "min_curv_velocity_detail.csv"),
            "traj_plot":   str(OUTPUT_DIR / "min_curv_traj.png"),
            "profile_plot":str(OUTPUT_DIR / "min_curv_profile.png"),
        },
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print("[Min-Curvature + Velocity Optimization]")
    print(json.dumps(payload, indent=2))
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
