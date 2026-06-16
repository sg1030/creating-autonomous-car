#!/usr/bin/env python3
"""
Interactive editor for global_waypoints.csv — edit BOTH speed and path position.

Panels
  - Left  (Track Map)      : drag the racing line directly to reshape the PATH.
                             The line is moved laterally (perpendicular to the
                             original heading); arc-length order is preserved so
                             the path can never tangle or self-intersect.
  - Top-right (Velocity)   : velocity profile vs arc length.
  - Bottom-right (Curvature): live |kappa| of the edited path with the vehicle
                             min-radius limit drawn, so you can see when a corner
                             becomes physically un-drivable.

Controls
  On the MAP (path):
    Left-click   add / grab a control point and drag it sideways
    Right-click  remove nearest control point (min 2 kept)
  On the VELOCITY panel:
    Left-click   add / drag a control point      Right-click  remove
  Keys:
    S  save edited CSV  →  global_waypoints_edited.csv  (psi & kappa recomputed)
    R  reset everything to the original line
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(_HERE, '..', 'maps', 'final', 'global_waypoints.csv')
OUTPUT_PATH = os.path.join(_HERE, '..', 'maps', 'final', 'global_waypoints_edited.csv')

# Vehicle steering limit → minimum path radius (for the curvature warning line).
WHEELBASE_M = 0.33
MAX_STEER_RAD = 0.40
KAPPA_LIMIT = np.tan(MAX_STEER_RAD) / WHEELBASE_M   # = 1/R_min ≈ 1.28 [1/m]

CtrlPt = Tuple[float, float]   # (arc_length, value)


# ── helpers ───────────────────────────────────────────────────────────────────

def arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(d)])


def spline_through(xs: np.ndarray, ys: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """Cubic spline through sorted (xs, ys), evaluated at s_query (falls back to linear)."""
    if len(xs) < 2:
        return np.full_like(s_query, ys[0] if len(ys) else 0.0, dtype=float)
    if len(xs) == 2:
        return np.interp(s_query, xs, ys).astype(float)
    try:
        cs = CubicSpline(xs, ys, bc_type='not-a-knot')
        return cs(s_query).astype(float)
    except Exception:
        return np.interp(s_query, xs, ys).astype(float)


def geom_from_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Heading psi and signed curvature kappa via centered differences (closed loop)."""
    dx = (np.roll(x, -1) - np.roll(x, 1)) * 0.5
    dy = (np.roll(y, -1) - np.roll(y, 1)) * 0.5
    ddx = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
    ddy = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)
    psi = np.arctan2(dy, dx)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    kappa = (dx * ddy - dy * ddx) / denom
    return psi, kappa


# ── velocity panel editor (1-D profile vs arc length) ──────────────────────────

class ProfileEditor:
    """Velocity editor that HUGS the loaded profile.

    The curve is `base + correction`, where base is the profile loaded from the
    CSV and the correction is a spline through the control points' offsets from
    base. With only the two endpoint controls (offset 0) the curve reproduces
    the loaded profile exactly — so the editor opens on the existing optimized
    velocity instead of a flat reset. Dragging a control point sets the absolute
    value there; the curve still follows the loaded profile everywhere else.
    """

    DRAG_FRAC = 0.04

    def __init__(self, s, init_vals, ax, *, color, title, ylabel, ylim, original=None):
        self.s, self.ax = s, ax
        self.init_vals = init_vals.copy()
        self.base = init_vals.copy()          # loaded profile the curve hugs
        self.color, self.title, self.ylabel, self.ylim = color, title, ylabel, ylim
        self.original = original
        self._ctrl: List[CtrlPt] = [(float(s[0]),  float(init_vals[0])),
                                    (float(s[-1]), float(init_vals[-1]))]
        self._drag: Optional[int] = None
        self._cached = init_vals.copy()

    def compute(self):
        c = sorted(self._ctrl, key=lambda p: p[0])
        cs = np.array([p[0] for p in c])
        cv = np.array([p[1] for p in c])
        # control offsets from the loaded profile → spline → add back to base
        corr = spline_through(cs, cv - np.interp(cs, self.s, self.base), self.s)
        self._cached = np.clip(self.base + corr, self.ylim[0], self.ylim[1])
        return self._cached

    @property
    def current(self):
        return self._cached

    def draw(self):
        vals = self.compute()
        ax = self.ax
        ax.cla()
        ax.set_title(self.title, color=self.color, fontweight='bold', fontsize=10)
        ax.set_xlabel('Arc Length [m]', fontsize=8)
        ax.set_ylabel(self.ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(float(self.s[0]), float(self.s[-1]))
        ax.set_ylim(*self.ylim)
        if self.original is not None:
            ax.plot(self.s, self.original, color='lightgray', lw=1.2, zorder=1, label='original')
        ax.plot(self.s, vals, color=self.color, lw=2.0, zorder=2, label='spline')
        c = sorted(self._ctrl, key=lambda p: p[0])
        ax.scatter([p[0] for p in c], [p[1] for p in c], c='crimson', s=90,
                   zorder=5, edgecolors='white', linewidths=1.4)
        ax.legend(fontsize=7, loc='upper right')
        return vals

    def reset(self):
        self._ctrl = [(float(self.s[0]), float(self.init_vals[0])),
                      (float(self.s[-1]), float(self.init_vals[-1]))]
        self._drag = None

    def _nearest(self, x, y):
        s_rng = float(self.s[-1] - self.s[0]) or 1.0
        y_rng = float(self.ylim[1] - self.ylim[0]) or 1.0
        d = [np.hypot((p[0] - x) / s_rng, (p[1] - y) / y_rng) for p in self._ctrl]
        i = int(np.argmin(d))
        return i if d[i] < self.DRAG_FRAC else None

    def on_press(self, x, y, button):
        if button == 1:
            idx = self._nearest(x, y)
            if idx is not None:
                self._drag = idx
                return False
            si = int(np.argmin(np.abs(self.s - x)))
            self._ctrl.append((float(self.s[si]), float(np.clip(y, *self.ylim))))
            self._ctrl.sort(key=lambda p: p[0])
            return True
        if button == 3:
            idx = self._nearest(x, y)
            if idx is not None and len(self._ctrl) > 2:
                self._ctrl.pop(idx)
                return True
        return False

    def on_drag(self, x, y):
        if self._drag is None:
            return False
        si = int(np.argmin(np.abs(self.s - x)))
        self._ctrl[self._drag] = (float(self.s[si]), float(np.clip(y, *self.ylim)))
        return True

    def on_release(self):
        if self._drag is None:
            return False
        self._ctrl.sort(key=lambda p: p[0])
        self._drag = None
        return True


# ── main application ───────────────────────────────────────────────────────────

class WaypointEditor:

    MAP_GRAB_M = 0.35   # cursor→control distance to grab on the map [m]

    def __init__(self, csv_path, output_path):
        self.csv_path, self.output_path = csv_path, output_path
        df = pd.read_csv(csv_path)
        self.df_orig = df
        self.x0 = df['x_m'].to_numpy(float)
        self.y0 = df['y_m'].to_numpy(float)
        self.psi = df['psi_rad'].to_numpy(float)
        self.wr = df['w_tr_right_m'].to_numpy(float)
        self.wl = df['w_tr_left_m'].to_numpy(float)
        self.s = arc_length(self.x0, self.y0)
        self.n = len(df)
        # left-pointing normal (+ey = left of heading)
        self.nx = -np.sin(self.psi)
        self.ny = np.cos(self.psi)

        # lateral-offset control points, stored as (waypoint_index, ey)
        self.ey_ctrl: List[Tuple[int, float]] = [(0, 0.0), (self.n - 1, 0.0)]
        self._map_drag: Optional[int] = None

        # ── figure ──────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(18, 10))
        try:
            self.fig.canvas.manager.set_window_title('Waypoint Spline Editor — path + speed')
        except Exception:
            pass
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.40, wspace=0.28,
                               left=0.06, right=0.97, top=0.92, bottom=0.07)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_vel = self.fig.add_subplot(gs[0, 1])
        self.ax_curv = self.fig.add_subplot(gs[1, 1])

        v_orig = df['vx_mps'].to_numpy(float)
        self.vel_ed = ProfileEditor(
            self.s, v_orig, self.ax_vel,
            color='royalblue', title='Velocity  (drag points)',
            ylabel='vx_mps [m/s]', ylim=(0.0, max(float(v_orig.max()) * 1.2, 5.0)),
            original=v_orig,
        )

        # Velocity colourbar for the map — created ONCE (range is fixed), never
        # inside the redraw loop, otherwise a new bar is stacked every refresh.
        self._vnorm = Normalize(vmin=0.0, vmax=self.vel_ed.ylim[1])
        _sm = ScalarMappable(norm=self._vnorm, cmap='RdYlGn')
        _sm.set_array([])
        self.fig.colorbar(_sm, ax=self.ax_map, label='vx [m/s]', shrink=0.65, pad=0.02)

        c = self.fig.canvas
        c.mpl_connect('button_press_event',   self._on_press)
        c.mpl_connect('button_release_event', self._on_release)
        c.mpl_connect('motion_notify_event',  self._on_motion)
        c.mpl_connect('key_press_event',      self._on_key)

        self._refresh()
        plt.show()

    # ── geometry from current edits ─────────────────────────────────────────
    def _ey_profile(self) -> np.ndarray:
        c = sorted(self.ey_ctrl, key=lambda p: p[0])
        s_ctrl = np.array([self.s[i] for i, _ in c])
        ey_ctrl = np.array([e for _, e in c])
        ey = spline_through(s_ctrl, ey_ctrl, self.s)
        # keep inside the track (90% of each wall distance)
        return np.clip(ey, -(self.wr * 0.9), self.wl * 0.9)

    def _edited_xy(self, ey: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.x0 + self.nx * ey, self.y0 + self.ny * ey

    def _ctrl_map_pos(self, idx: int, ey: float):
        return self.x0[idx] + self.nx[idx] * ey, self.y0[idx] + self.ny[idx] * ey

    # ── drawing ─────────────────────────────────────────────────────────────
    def _refresh(self):
        vel = self.vel_ed.draw()
        ey = self._ey_profile()
        xe, ye = self._edited_xy(ey)
        _, kappa = geom_from_xy(xe, ye)
        self._draw_map(vel, ey, xe, ye)
        self._draw_curv(kappa)
        self._hint()
        self.fig.canvas.draw_idle()

    def _draw_map(self, vel, ey, xe, ye):
        ax = self.ax_map
        ax.cla()
        ax.set_title('Track Map — drag the line to move the PATH (colour = velocity)', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('y [m]', fontsize=8)

        # walls (from original line + widths along the normal)
        ax.plot(self.x0 - self.nx * self.wr, self.y0 - self.ny * self.wr,
                'k-', lw=0.9, alpha=0.35)
        ax.plot(self.x0 + self.nx * self.wl, self.y0 + self.ny * self.wl,
                'k-', lw=0.9, alpha=0.35, label='boundaries')
        # original line (faint)
        ax.plot(self.x0, self.y0, color='lightgray', lw=1.2, alpha=0.8, zorder=2, label='original')

        ax.scatter(xe, ye, c=vel, cmap='RdYlGn', s=14, norm=self._vnorm, zorder=3)

        # ey control points on the map
        for idx, e in self.ey_ctrl:
            cx, cy = self._ctrl_map_pos(idx, e)
            ax.scatter([cx], [cy], c='crimson', s=120, zorder=6,
                       edgecolors='white', linewidths=1.5)
        ax.plot(xe[0], ye[0], 'b^', ms=10, zorder=6, label='start')
        ax.legend(fontsize=7, loc='upper right')

    def _draw_curv(self, kappa):
        ax = self.ax_curv
        ax.cla()
        ax.set_title('Curvature |kappa|  (edited path)', fontsize=10)
        ax.set_xlabel('Arc Length [m]', fontsize=8)
        ax.set_ylabel('|kappa| [1/m]', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(float(self.s[0]), float(self.s[-1]))
        ax.plot(self.s, np.abs(kappa), color='purple', lw=1.5, zorder=2)
        ax.axhline(KAPPA_LIMIT, color='red', lw=1.2, ls='--',
                   label=f'vehicle limit (R={1.0 / KAPPA_LIMIT:.2f} m)')
        over = np.abs(kappa) > KAPPA_LIMIT
        if over.any():
            ax.scatter(self.s[over], np.abs(kappa)[over], c='red', s=10, zorder=3)
        ax.legend(fontsize=7, loc='upper right')

    def _hint(self, extra=''):
        hint = ('MAP: drag line = move path  ·  '
                'Left-click add/drag  Right-click remove  ·  '
                'VELOCITY panel editable  ·  S save  R reset')
        self.fig.suptitle(f'{extra}   {hint}' if extra else hint,
                          fontsize=8.5, color='darkgreen' if extra else '#444')

    # ── event routing ───────────────────────────────────────────────────────
    def _on_press(self, event):
        if event.xdata is None:
            return
        if event.inaxes == self.ax_vel:
            if self.vel_ed.on_press(event.xdata, event.ydata, event.button):
                self._refresh()
        elif event.inaxes == self.ax_map:
            if self._map_press(event.xdata, event.ydata, event.button):
                self._refresh()

    def _on_motion(self, event):
        if event.xdata is None:
            return
        if event.inaxes == self.ax_vel:
            if self.vel_ed.on_drag(event.xdata, event.ydata):
                self._refresh()
        elif event.inaxes == self.ax_map and self._map_drag is not None:
            self._map_set_ey(self._map_drag, event.xdata, event.ydata)
            self._refresh()

    def _on_release(self, event):
        moved = self.vel_ed.on_release()
        if self._map_drag is not None:
            self.ey_ctrl.sort(key=lambda p: p[0])
            self._map_drag = None
            moved = True
        if moved:
            self._refresh()

    def _on_key(self, event):
        if event.key == 's':
            self._save()
        elif event.key == 'r':
            self._reset()

    # ── map (path) editing ──────────────────────────────────────────────────
    def _nearest_ctrl(self, cx, cy) -> Optional[int]:
        best, best_d = None, self.MAP_GRAB_M
        for k, (idx, e) in enumerate(self.ey_ctrl):
            px, py = self._ctrl_map_pos(idx, e)
            d = np.hypot(cx - px, cy - py)
            if d < best_d:
                best, best_d = k, d
        return best

    def _ey_at(self, idx, cx, cy) -> float:
        """Signed lateral offset of cursor from the original point at idx."""
        e = (cx - self.x0[idx]) * self.nx[idx] + (cy - self.y0[idx]) * self.ny[idx]
        return float(np.clip(e, -(self.wr[idx] * 0.9), self.wl[idx] * 0.9))

    def _map_set_ey(self, k, cx, cy):
        idx, _ = self.ey_ctrl[k]
        self.ey_ctrl[k] = (idx, self._ey_at(idx, cx, cy))

    def _map_press(self, cx, cy, button) -> bool:
        if button == 1:
            k = self._nearest_ctrl(cx, cy)
            if k is not None:
                self._map_drag = k
                return False
            # add a control point at the nearest waypoint (by current line position)
            ey = self._ey_profile()
            xe, ye = self._edited_xy(ey)
            idx = int(np.argmin(np.hypot(xe - cx, ye - cy)))
            if any(i == idx for i, _ in self.ey_ctrl):
                return False
            self.ey_ctrl.append((idx, self._ey_at(idx, cx, cy)))
            self.ey_ctrl.sort(key=lambda p: p[0])
            return True
        if button == 3:
            k = self._nearest_ctrl(cx, cy)
            if k is not None and len(self.ey_ctrl) > 2:
                self.ey_ctrl.pop(k)
                return True
        return False

    # ── save / reset ────────────────────────────────────────────────────────
    def _save(self):
        ey = self._ey_profile()
        xe, ye = self._edited_xy(ey)
        psi, kappa = geom_from_xy(xe, ye)

        df = self.df_orig.copy()
        df['x_m'] = xe
        df['y_m'] = ye
        df['psi_rad'] = psi
        df['kappa_radpm'] = kappa
        df['vx_mps'] = self.vel_ed.current
        # clearance to walls shifts with the line: +ey (left) → more right room, less left
        if 'w_tr_right_m' in df:
            df['w_tr_right_m'] = np.maximum(self.wr + ey, 0.0)
        if 'w_tr_left_m' in df:
            df['w_tr_left_m'] = np.maximum(self.wl - ey, 0.0)

        df.to_csv(self.output_path, index=False, float_format='%.6f')
        over = int(np.sum(np.abs(kappa) > KAPPA_LIMIT))
        warn = f' (⚠ {over} pts over vehicle curvature limit)' if over else ''
        print(f'[saved] {self.output_path}{warn}')
        self._hint(f'✓ saved → {os.path.basename(self.output_path)}{warn}')
        self.fig.canvas.draw_idle()

    def _reset(self):
        self.vel_ed.reset()
        self.ey_ctrl = [(0, 0.0), (self.n - 1, 0.0)]
        self._map_drag = None
        print('[reset] restored original line + speed')
        self._refresh()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    csv = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(CSV_PATH)
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.abspath(OUTPUT_PATH)
    if not os.path.isfile(csv):
        sys.exit(f'[error] CSV not found: {csv}')
    print(f'[load]  {csv}')
    print(f'[save]  {out}')
    WaypointEditor(csv, out)
