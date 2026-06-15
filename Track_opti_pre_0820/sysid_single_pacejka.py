from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


DEFAULT_DRIVING_FILES = [
    Path("/home/hmcl/data needed/driving_data_20260602_170124.csv"),
    Path("/home/hmcl/data needed/driving_data_20260602_181926.csv"),
    Path("/home/hmcl/data needed/driving_data_20260602_182106.csv"),
]
# Map CSVs live in <repo>/stack_master/maps/final/ ; resolve relative to this file.
_MAP_DIR = Path(__file__).resolve().parent.parent / "stack_master" / "maps" / "final"
DEFAULT_CENTERLINE = _MAP_DIR / "centerline.csv"
DEFAULT_BOUNDARY_LEFT = _MAP_DIR / "boundary_left.csv"
DEFAULT_BOUNDARY_RIGHT = _MAP_DIR / "boundary_right.csv"
DEFAULT_OUTPUT_DIR = Path("outputs/single_pacejka_sysid")


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def choose_odd_window(n_points: int, target: int = 21) -> int:
    if n_points < 5:
        return 0
    window = min(target, n_points if n_points % 2 == 1 else n_points - 1)
    if window < 5:
        return 0
    return window


def smooth_signal(values: np.ndarray, target_window: int = 21, polyorder: int = 3) -> np.ndarray:
    window = choose_odd_window(len(values), target_window)
    if window == 0:
        return values.copy()
    poly = min(polyorder, window - 2)
    return savgol_filter(values, window_length=window, polyorder=poly, mode="interp")


def finite_difference(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        dt = max(time[-1] - time[0], 1e-6)
        return np.gradient(values) / dt
    return np.gradient(values, time, edge_order=2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class VehicleParams:
    B: float = 1.1
    C: float = 1.3
    mu: float = 0.8
    Iz: float = 0.024
    m: float = 3.0
    lf: float = 0.14
    lr: float = 0.14
    g: float = 9.81
    min_vx_for_slip: float = 0.3

    @property
    def Df(self) -> float:
        return self.mu * self.m * self.g * self.lr / (self.lf + self.lr)

    @property
    def Dr(self) -> float:
        return self.mu * self.m * self.g * self.lf / (self.lf + self.lr)

    def to_vector(self) -> np.ndarray:
        return np.array([self.B, self.C, self.mu, self.Iz], dtype=float)

    def with_vector(self, values: np.ndarray) -> "VehicleParams":
        return VehicleParams(
            B=float(values[0]),
            C=float(values[1]),
            mu=float(values[2]),
            Iz=float(values[3]),
            m=self.m,
            lf=self.lf,
            lr=self.lr,
            g=self.g,
            min_vx_for_slip=self.min_vx_for_slip,
        )


class ClosedTrackSpline:
    def __init__(self, centerline_csv: Path) -> None:
        center_df = pd.read_csv(centerline_csv)
        xy = center_df[["x_m", "y_m"]].to_numpy(dtype=float)
        if np.linalg.norm(xy[0] - xy[-1]) > 1e-9:
            xy = np.vstack([xy, xy[0]])

        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        self.s_nodes = np.concatenate(([0.0], np.cumsum(ds)))
        self.length = float(self.s_nodes[-1])
        self.centerline_xy = xy
        self.x_spline = CubicSpline(self.s_nodes, xy[:, 0], bc_type="periodic")
        self.y_spline = CubicSpline(self.s_nodes, xy[:, 1], bc_type="periodic")

    def wrap_s(self, s: np.ndarray | float) -> np.ndarray | float:
        return np.mod(s, self.length)

    def center_pose(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_wrapped = self.wrap_s(s)
        x = self.x_spline(s_wrapped)
        y = self.y_spline(s_wrapped)
        dx = self.x_spline(s_wrapped, 1)
        dy = self.y_spline(s_wrapped, 1)
        psi = np.arctan2(dy, dx)
        return x, y, psi

    def curvature(self, s: np.ndarray) -> np.ndarray:
        s_wrapped = self.wrap_s(s)
        dx = self.x_spline(s_wrapped, 1)
        dy = self.y_spline(s_wrapped, 1)
        ddx = self.x_spline(s_wrapped, 2)
        ddy = self.y_spline(s_wrapped, 2)
        denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
        return (dx * ddy - dy * ddx) / denom

    def frenet_to_global(
        self,
        s: np.ndarray,
        ey: np.ndarray,
        epsi: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_c, y_c, psi_c = self.center_pose(s)
        x = x_c - ey * np.sin(psi_c)
        y = y_c + ey * np.cos(psi_c)
        psi = psi_c if epsi is None else wrap_to_pi(psi_c + epsi)
        return x, y, psi


@dataclass
class PreparedSegment:
    source_file: str
    segment_id: int
    time: np.ndarray
    dt: np.ndarray
    s: np.ndarray
    ey: np.ndarray
    epsi: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    wz: np.ndarray
    curv: np.ndarray
    a: np.ndarray
    delta: np.ndarray
    valid_mask: np.ndarray

    @property
    def name(self) -> str:
        return f"{Path(self.source_file).stem}_seg{self.segment_id:02d}"


@dataclass
class SegmentResult:
    segment: PreparedSegment
    predicted: np.ndarray
    metrics: dict[str, float]


def rhs_single_track(x: np.ndarray, u: np.ndarray, curv: float, params: VehicleParams) -> np.ndarray:
    s, ey, epsi, vx, vy, wz = x
    ax_cmd, delta = u

    vx_safe = vx
    if abs(vx_safe) < params.min_vx_for_slip:
        vx_safe = np.sign(vx_safe) * params.min_vx_for_slip if vx_safe != 0.0 else params.min_vx_for_slip

    alpha_f = -np.arctan2(
        (vy + params.lf * wz) * np.cos(delta) - vx_safe * np.sin(delta),
        vx_safe * np.cos(delta) + (vy + params.lf * wz) * np.sin(delta),
    )
    alpha_r = -np.arctan2(vy - params.lr * wz, vx_safe)

    fyf = params.Df * np.sin(params.C * np.arctan(params.B * alpha_f))
    fyr = params.Dr * np.sin(params.C * np.arctan(params.B * alpha_r))

    denom = 1.0 - ey * curv
    if abs(denom) < 1e-3:
        denom = np.sign(denom) * 1e-3 if denom != 0.0 else 1e-3

    ds = (vx * np.cos(epsi) - vy * np.sin(epsi)) / denom
    dey = vx * np.sin(epsi) + vy * np.cos(epsi)
    depsi = wz - curv * ds

    ax = ax_cmd - curv * vx - fyf * np.sin(delta) / params.m
    ay = (fyf * np.cos(delta) + fyr) / params.m
    dvx = ax + wz * vy
    dvy = ay - wz * vx
    dwz = (params.lf * fyf * np.cos(delta) - params.lr * fyr) / params.Iz

    return np.array([ds, dey, depsi, dvx, dvy, dwz], dtype=float)


def rk4_step(x: np.ndarray, u: np.ndarray, curv: float, dt: float, params: VehicleParams) -> np.ndarray:
    k1 = rhs_single_track(x, u, curv, params)
    k2 = rhs_single_track(x + 0.5 * dt * k1, u, curv, params)
    k3 = rhs_single_track(x + 0.5 * dt * k2, u, curv, params)
    k4 = rhs_single_track(x + dt * k3, u, curv, params)
    x_next = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    x_next[2] = wrap_to_pi(x_next[2])
    return x_next


def fill_nan_angles(raw_angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(raw_angle))
    valid = np.isfinite(raw_angle)
    if not np.any(valid):
        return np.zeros_like(raw_angle), valid

    unwrapped_valid = np.unwrap(raw_angle[valid])
    filled = np.interp(idx, idx[valid], unwrapped_valid)
    return filled, valid


def estimate_epsi(
    time: np.ndarray,
    s: np.ndarray,
    ey: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    wz: np.ndarray,
    curv: np.ndarray,
    min_speed: float,
) -> np.ndarray:
    ds_dt = finite_difference(s, time)
    dey_dt = finite_difference(ey, time)
    track_vx = ds_dt * (1.0 - ey * curv)
    track_vy = dey_dt

    speed = np.hypot(vx, vy)
    track_speed = np.hypot(track_vx, track_vy)
    valid = (speed > min_speed) & (track_speed > 0.05)

    geom = np.full_like(vx, np.nan, dtype=float)
    numer = vx * track_vy - vy * track_vx
    denom = vx * track_vx + vy * track_vy
    geom[valid] = np.arctan2(numer[valid], denom[valid])

    geom_filled, geom_valid = fill_nan_angles(geom)
    int_epsi = np.zeros_like(geom_filled)
    int_epsi[0] = geom_filled[0]
    epsi_dot = wz - curv * ds_dt
    for i in range(len(time) - 1):
        dt = time[i + 1] - time[i]
        int_epsi[i + 1] = int_epsi[i] + 0.5 * (epsi_dot[i] + epsi_dot[i + 1]) * dt

    combined = int_epsi.copy()
    if np.any(geom_valid):
        offset = np.median(geom_filled[geom_valid] - int_epsi[geom_valid])
        combined = combined + offset
        combined = 0.65 * combined + 0.35 * geom_filled

    return smooth_signal(combined, target_window=31)


def unique_monotonic_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    keep[1:] = np.diff(df["time"].to_numpy(dtype=float)) > 1e-9
    return df.loc[keep].reset_index(drop=True)


def resample_segment(
    time: np.ndarray,
    signal: np.ndarray,
    target_time: np.ndarray,
    smooth_window: int,
) -> np.ndarray:
    smoothed = smooth_signal(signal, target_window=smooth_window)
    return np.interp(target_time, time, smoothed)


def prepare_segments(
    csv_path: Path,
    resample_dt: float,
    min_speed: float,
    split_jump_s: float = 5.0,
    split_jump_ey: float = 1.5,
    max_time_gap: float = 0.1,
    min_segment_points: int = 25,
) -> list[PreparedSegment]:
    df = unique_monotonic_time(pd.read_csv(csv_path))
    s = df["s"].to_numpy(dtype=float)
    ey = df["ey"].to_numpy(dtype=float)
    time = df["time"].to_numpy(dtype=float)

    cut_idx = np.where(
        (np.diff(time) > max_time_gap)
        | (np.abs(np.diff(s)) > split_jump_s)
        | (np.abs(np.diff(ey)) > split_jump_ey)
    )[0]

    starts = np.concatenate(([0], cut_idx + 1))
    ends = np.concatenate((cut_idx + 1, [len(df)]))

    segments: list[PreparedSegment] = []
    segment_counter = 0
    for start, end in zip(starts, ends, strict=False):
        raw = df.iloc[start:end].copy()
        if len(raw) < min_segment_points:
            continue

        time_raw = raw["time"].to_numpy(dtype=float)
        duration = time_raw[-1] - time_raw[0]
        if duration < 1.0:
            continue

        target_time = np.arange(time_raw[0], time_raw[-1] + 0.5 * resample_dt, resample_dt)
        if len(target_time) < min_segment_points:
            continue

        native_dt = np.median(np.diff(time_raw))
        smooth_window = max(11, int(round(0.45 / max(native_dt, 1e-3))) | 1)
        smooth_window = min(smooth_window, 51)

        s_rs = resample_segment(time_raw, raw["s"].to_numpy(dtype=float), target_time, smooth_window)
        ey_rs = resample_segment(time_raw, raw["ey"].to_numpy(dtype=float), target_time, smooth_window)
        vx_rs = resample_segment(time_raw, raw["vx"].to_numpy(dtype=float), target_time, smooth_window)
        wz_rs = resample_segment(time_raw, raw["w"].to_numpy(dtype=float), target_time, smooth_window)
        curv_rs = resample_segment(time_raw, raw["curv"].to_numpy(dtype=float), target_time, smooth_window)
        a_rs = resample_segment(time_raw, raw["a"].to_numpy(dtype=float), target_time, smooth_window)
        delta_rs = resample_segment(time_raw, raw["delta"].to_numpy(dtype=float), target_time, smooth_window)

        if "vy" in raw:
            vy_rs = resample_segment(time_raw, raw["vy"].to_numpy(dtype=float), target_time, smooth_window)
        else:
            vy_rs = np.zeros_like(vx_rs)

        vy_rs = np.where(np.abs(vy_rs) < 1e-6, 0.0, vy_rs)
        epsi_rs = estimate_epsi(target_time, s_rs, ey_rs, vx_rs, vy_rs, wz_rs, curv_rs, min_speed=min_speed)

        dt = np.diff(target_time, append=target_time[-1] + resample_dt)
        valid_mask = (
            np.isfinite(s_rs)
            & np.isfinite(ey_rs)
            & np.isfinite(epsi_rs)
            & np.isfinite(vx_rs)
            & np.isfinite(wz_rs)
            & np.isfinite(curv_rs)
            & (np.abs(vx_rs) > min_speed)
            & (np.abs(1.0 - ey_rs * curv_rs) > 0.15)
            & (np.abs(delta_rs) < 0.75)
        )

        segments.append(
            PreparedSegment(
                source_file=str(csv_path),
                segment_id=segment_counter,
                time=target_time,
                dt=dt,
                s=s_rs,
                ey=ey_rs,
                epsi=epsi_rs,
                vx=vx_rs,
                vy=vy_rs,
                wz=wz_rs,
                curv=curv_rs,
                a=a_rs,
                delta=delta_rs,
                valid_mask=valid_mask,
            )
        )
        segment_counter += 1

    return segments


def identification_residuals(theta: np.ndarray, segments: list[PreparedSegment], template: VehicleParams) -> np.ndarray:
    params = template.with_vector(theta)
    residuals: list[float] = []

    for seg in segments:
        fit_idx = np.flatnonzero(seg.valid_mask[:-1] & seg.valid_mask[1:])
        for i in fit_idx:
            xk = np.array([seg.s[i], seg.ey[i], seg.epsi[i], seg.vx[i], seg.vy[i], seg.wz[i]], dtype=float)
            uk = np.array([seg.a[i], seg.delta[i]], dtype=float)
            x_next_pred = rk4_step(xk, uk, seg.curv[i], seg.dt[i], params)

            s_err = (x_next_pred[0] - seg.s[i + 1]) / 0.30
            ey_err = (x_next_pred[1] - seg.ey[i + 1]) / 0.20
            epsi_err = wrap_to_pi(x_next_pred[2] - seg.epsi[i + 1]) / 0.08
            vx_err = (x_next_pred[3] - seg.vx[i + 1]) / 0.35
            wz_err = (x_next_pred[5] - seg.wz[i + 1]) / 0.45
            vy_penalty = x_next_pred[4] / 0.30

            residuals.extend([s_err, ey_err, epsi_err, vx_err, wz_err, 0.35 * vy_penalty])

    prior = template.to_vector()
    prior_scale = np.array([0.5, 0.4, 0.25, 0.01], dtype=float)
    residuals.extend(((theta - prior) / prior_scale * 0.05).tolist())

    return np.asarray(residuals, dtype=float)


def fit_pacejka_params(segments: list[PreparedSegment], initial_params: VehicleParams) -> tuple[VehicleParams, object]:
    lb = np.array([0.5, 0.8, 0.3, 0.008], dtype=float)
    ub = np.array([3.0, 2.0, 1.5, 0.06], dtype=float)

    result = least_squares(
        identification_residuals,
        x0=initial_params.to_vector(),
        bounds=(lb, ub),
        args=(segments, initial_params),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=80,
        verbose=2,
    )
    return initial_params.with_vector(result.x), result


def rollout_segment(segment: PreparedSegment, params: VehicleParams) -> np.ndarray:
    predicted = np.zeros((len(segment.time), 6), dtype=float)
    predicted[0] = np.array(
        [segment.s[0], segment.ey[0], segment.epsi[0], segment.vx[0], segment.vy[0], segment.wz[0]],
        dtype=float,
    )

    for i in range(len(segment.time) - 1):
        u = np.array([segment.a[i], segment.delta[i]], dtype=float)
        predicted[i + 1] = rk4_step(predicted[i], u, segment.curv[i], segment.dt[i], params)

    return predicted


def compute_metrics(segment: PreparedSegment, predicted: np.ndarray, track: ClosedTrackSpline) -> dict[str, float]:
    x_meas, y_meas, _ = track.frenet_to_global(segment.s, segment.ey, segment.epsi)
    x_pred, y_pred, _ = track.frenet_to_global(predicted[:, 0], predicted[:, 1], predicted[:, 2])

    metrics = {
        "rmse_s_m": float(np.sqrt(np.mean((predicted[:, 0] - segment.s) ** 2))),
        "rmse_ey_m": float(np.sqrt(np.mean((predicted[:, 1] - segment.ey) ** 2))),
        "rmse_epsi_rad": float(np.sqrt(np.mean(wrap_to_pi(predicted[:, 2] - segment.epsi) ** 2))),
        "rmse_vx_mps": float(np.sqrt(np.mean((predicted[:, 3] - segment.vx) ** 2))),
        "rmse_wz_radps": float(np.sqrt(np.mean((predicted[:, 5] - segment.wz) ** 2))),
        "rmse_xy_m": float(np.sqrt(np.mean((x_pred - x_meas) ** 2 + (y_pred - y_meas) ** 2))),
        "duration_s": float(segment.time[-1] - segment.time[0]),
    }
    return metrics


def save_rollout_csv(
    output_csv: Path,
    segment_results: list[SegmentResult],
    track: ClosedTrackSpline,
) -> None:
    rows: list[pd.DataFrame] = []
    for result in segment_results:
        seg = result.segment
        pred = result.predicted
        x_meas, y_meas, psi_meas = track.frenet_to_global(seg.s, seg.ey, seg.epsi)
        x_pred, y_pred, psi_pred = track.frenet_to_global(pred[:, 0], pred[:, 1], pred[:, 2])

        rows.append(
            pd.DataFrame(
                {
                    "segment": seg.segment_id,
                    "time": seg.time,
                    "s_meas": seg.s,
                    "ey_meas": seg.ey,
                    "epsi_meas": seg.epsi,
                    "vx_meas": seg.vx,
                    "vy_meas": seg.vy,
                    "wz_meas": seg.wz,
                    "curv": seg.curv,
                    "a": seg.a,
                    "delta": seg.delta,
                    "x_meas": x_meas,
                    "y_meas": y_meas,
                    "psi_meas": psi_meas,
                    "s_pred": pred[:, 0],
                    "ey_pred": pred[:, 1],
                    "epsi_pred": pred[:, 2],
                    "vx_pred": pred[:, 3],
                    "vy_pred": pred[:, 4],
                    "wz_pred": pred[:, 5],
                    "x_pred": x_pred,
                    "y_pred": y_pred,
                    "psi_pred": psi_pred,
                }
            )
        )

    pd.concat(rows, ignore_index=True).to_csv(output_csv, index=False)


def plot_state_comparison(output_path: Path, segment_results: list[SegmentResult]) -> None:
    reference = max(segment_results, key=lambda item: item.metrics["duration_s"])
    ref_seg = reference.segment
    ref_pred = reference.predicted
    time_rel = ref_seg.time - ref_seg.time[0]
    measured = np.column_stack([ref_seg.s, ref_seg.ey, ref_seg.epsi, ref_seg.vx, ref_seg.vy, ref_seg.wz])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    axes = axes.ravel()
    labels = [
        ("s [m]", 0, "s"),
        ("ey [m]", 1, "ey"),
        ("epsi [rad]", 2, "epsi"),
        ("vx [m/s]", 3, "vx"),
        ("wz [rad/s]", 5, "wz"),
    ]

    for ax_idx, (ylabel, state_idx, title) in enumerate(labels):
        ax = axes[ax_idx]
        ax.plot(time_rel, measured[:, state_idx], linewidth=1.4, label="measured" if ax_idx == 0 else None)
        ax.plot(
            time_rel,
            ref_pred[:, state_idx],
            linewidth=1.4,
            linestyle="--",
            label="rollout" if ax_idx == 0 else None,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    mean_metrics = {
        "rmse_s_m": float(np.mean([res.metrics["rmse_s_m"] for res in segment_results])),
        "rmse_ey_m": float(np.mean([res.metrics["rmse_ey_m"] for res in segment_results])),
        "rmse_epsi_rad": float(np.mean([res.metrics["rmse_epsi_rad"] for res in segment_results])),
        "rmse_vx_mps": float(np.mean([res.metrics["rmse_vx_mps"] for res in segment_results])),
        "rmse_wz_radps": float(np.mean([res.metrics["rmse_wz_radps"] for res in segment_results])),
        "rmse_xy_m": float(np.mean([res.metrics["rmse_xy_m"] for res in segment_results])),
    }
    metric_ax = axes[-1]
    text_lines = [
        f"reference: {reference.segment.name}",
        f"segments: {len(segment_results)}",
        f"duration: {reference.metrics['duration_s']:.2f} s",
        "",
    ]
    text_lines.extend(f"{key}: {value:.4f}" for key, value in mean_metrics.items())
    metric_ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        family="monospace",
    )

    for ax in axes[:-1]:
        ax.set_xlabel("time [s]")
    axes[0].legend(loc="upper right")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_track_map(
    output_path: Path,
    track: ClosedTrackSpline,
    boundary_left_csv: Path,
    boundary_right_csv: Path,
    grouped_results: dict[str, list[SegmentResult]],
) -> None:
    left_df = pd.read_csv(boundary_left_csv)
    right_df = pd.read_csv(boundary_right_csv)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(left_df["x_m"], left_df["y_m"], color="black", linewidth=1.0, label="boundary left")
    ax.plot(right_df["x_m"], right_df["y_m"], color="dimgray", linewidth=1.0, label="boundary right")
    ax.plot(
        track.centerline_xy[:, 0],
        track.centerline_xy[:, 1],
        color="tab:blue",
        linewidth=1.0,
        linestyle=":",
        label="centerline",
    )

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(grouped_results), 1)))
    for color, (source, results) in zip(colors, grouped_results.items(), strict=False):
        measured_labeled = False
        pred_labeled = False
        for result in results:
            seg = result.segment
            pred = result.predicted
            x_meas, y_meas, _ = track.frenet_to_global(seg.s, seg.ey, seg.epsi)
            x_pred, y_pred, _ = track.frenet_to_global(pred[:, 0], pred[:, 1], pred[:, 2])
            ax.plot(
                x_meas,
                y_meas,
                color=color,
                linewidth=1.5,
                alpha=0.85,
                label=f"{Path(source).stem} measured" if not measured_labeled else None,
            )
            ax.plot(
                x_pred,
                y_pred,
                color=color,
                linewidth=1.4,
                linestyle="--",
                alpha=0.95,
                label=f"{Path(source).stem} rollout" if not pred_labeled else None,
            )
            measured_labeled = True
            pred_labeled = True

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Single-track Pacejka SysID Trajectory Comparison")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(grouped_results: dict[str, list[SegmentResult]]) -> dict[str, object]:
    per_file: dict[str, dict[str, float]] = {}
    overall_xy: list[float] = []
    overall_ey: list[float] = []
    overall_vx: list[float] = []
    overall_wz: list[float] = []

    for source, results in grouped_results.items():
        file_xy = [res.metrics["rmse_xy_m"] for res in results]
        file_ey = [res.metrics["rmse_ey_m"] for res in results]
        file_vx = [res.metrics["rmse_vx_mps"] for res in results]
        file_wz = [res.metrics["rmse_wz_radps"] for res in results]
        per_file[source] = {
            "n_segments": int(len(results)),
            "mean_rmse_xy_m": float(np.mean(file_xy)),
            "mean_rmse_ey_m": float(np.mean(file_ey)),
            "mean_rmse_vx_mps": float(np.mean(file_vx)),
            "mean_rmse_wz_radps": float(np.mean(file_wz)),
        }
        overall_xy.extend(file_xy)
        overall_ey.extend(file_ey)
        overall_vx.extend(file_vx)
        overall_wz.extend(file_wz)

    return {
        "per_file": per_file,
        "overall": {
            "mean_rmse_xy_m": float(np.mean(overall_xy)) if overall_xy else np.nan,
            "mean_rmse_ey_m": float(np.mean(overall_ey)) if overall_ey else np.nan,
            "mean_rmse_vx_mps": float(np.mean(overall_vx)) if overall_vx else np.nan,
            "mean_rmse_wz_radps": float(np.mean(overall_wz)) if overall_wz else np.nan,
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-track Pacejka sysid + trajectory visualization")
    parser.add_argument(
        "--driving-files",
        nargs="+",
        type=Path,
        default=DEFAULT_DRIVING_FILES,
        help="Driving CSV files with time, s, ey, vx, vy, w, curv, a, delta columns.",
    )
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--boundary-left", type=Path, default=DEFAULT_BOUNDARY_LEFT)
    parser.add_argument("--boundary-right", type=Path, default=DEFAULT_BOUNDARY_RIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resample-dt", type=float, default=0.05, help="Uniform time step for preprocessing.")
    parser.add_argument("--min-speed", type=float, default=0.5, help="Minimum |vx| used in identification.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    ensure_dir(args.output_dir)

    track = ClosedTrackSpline(args.centerline)
    all_segments: list[PreparedSegment] = []
    grouped_segments: dict[str, list[PreparedSegment]] = {}

    for csv_path in args.driving_files:
        segments = prepare_segments(csv_path, resample_dt=args.resample_dt, min_speed=args.min_speed)
        if not segments:
            print(f"[WARN] no usable segments in {csv_path}")
            continue
        grouped_segments[str(csv_path)] = segments
        all_segments.extend(segments)
        print(f"[INFO] {csv_path.name}: prepared {len(segments)} segments")

    if not all_segments:
        raise RuntimeError("No usable driving segments found. Check the CSV paths and thresholds.")

    initial_params = VehicleParams()
    identified_params, optimizer_result = fit_pacejka_params(all_segments, initial_params)

    print("\n[SysID Result]")
    print(json.dumps(asdict(identified_params), indent=2))
    print(f"optimizer_cost={optimizer_result.cost:.6f}")
    print(f"optimizer_success={optimizer_result.success}")
    print(f"optimizer_message={optimizer_result.message}")

    grouped_results: dict[str, list[SegmentResult]] = {}
    for source, segments in grouped_segments.items():
        file_results: list[SegmentResult] = []
        for segment in segments:
            predicted = rollout_segment(segment, identified_params)
            metrics = compute_metrics(segment, predicted, track)
            file_results.append(SegmentResult(segment=segment, predicted=predicted, metrics=metrics))
        grouped_results[source] = file_results

        state_plot_path = args.output_dir / f"{Path(source).stem}_states.png"
        csv_output_path = args.output_dir / f"{Path(source).stem}_rollout.csv"
        plot_state_comparison(state_plot_path, file_results)
        save_rollout_csv(csv_output_path, file_results, track)

    plot_track_map(
        args.output_dir / "trajectory_map.png",
        track,
        args.boundary_left,
        args.boundary_right,
        grouped_results,
    )

    summary = summarize_metrics(grouped_results)
    summary_payload = {
        "identified_params": asdict(identified_params),
        "optimizer": {
            "cost": float(optimizer_result.cost),
            "success": bool(optimizer_result.success),
            "message": str(optimizer_result.message),
            "nfev": int(optimizer_result.nfev),
        },
        "summary": summary,
        "resample_dt": args.resample_dt,
        "min_speed": args.min_speed,
        "track_length_m": track.length,
        "n_segments": len(all_segments),
    }

    summary_path = args.output_dir / "sysid_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    print("\n[Artifacts]")
    print(f"- summary: {summary_path}")
    print(f"- map plot: {args.output_dir / 'trajectory_map.png'}")
    for source in grouped_results:
        stem = Path(source).stem
        print(f"- {stem}: {args.output_dir / f'{stem}_states.png'}")
        print(f"- {stem}: {args.output_dir / f'{stem}_rollout.csv'}")


if __name__ == "__main__":
    main()
