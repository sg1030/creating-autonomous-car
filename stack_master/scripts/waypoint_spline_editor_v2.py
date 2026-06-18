#!/usr/bin/env python3
"""
Interactive editor for global_waypoints.csv — edit BOTH speed and path position.

v2 — LOCAL EDITING
  The original editor opened with only the two endpoint control points, so the
  cubic spline had to bridge huge gaps whenever you added one dot in the middle.
  That made every edit GLOBAL: the line overshot / wiggled far away from where
  you actually clicked ("noise").

  This version PRE-SEEDS a dense grid of control dots that sit exactly on the
  existing racing line and on the loaded velocity profile. The editor therefore
  opens identical to the input, but because neighbouring dots are already pinned
  close by, dragging any single dot only deforms that LOCAL portion — you tweak a
  corner or a small speed stretch "a little bit" without disturbing the rest.

Panels
  - Left  (Track Map)      : drag the racing line dots to reshape the PATH
                             laterally (perpendicular to the original heading);
                             arc-length order is preserved so the path can never
                             tangle or self-intersect.
  - Top-right (Velocity)   : velocity profile vs arc length (dotted grid).
  - Bottom-right (Curvature): live |kappa| of the edited path with the vehicle
                             min-radius limit drawn.

Controls
  On the MAP (path):
    Left-click   grab the nearest dot and drag it sideways (or add one)
    Right-click  remove nearest dot (min 2 kept)
  On the VELOCITY panel:
    Left-click   grab/drag the nearest dot (or add one)   Right-click  remove
  Keys:
    S  save edited CSV  →  global_waypoints_edited.csv  (psi & kappa recomputed)
    R  reset everything to the original line + re-seed the dot grid
    [  /  ]  fewer / more PATH dots (coarser / finer local control)
    -  /  =  fewer / more VELOCITY dots
    M  cycle VELOCITY shape mode: HUG → FREE spline → POLY
    ,  /  .  POLY degree down / up  (0 flat · 1 linear · 2 quad · 3 cubic …)

Velocity shape modes
  HUG    : curve = loaded profile + spline correction (default; follows the
           previous optimized velocity's tendency — the original behaviour).
  FREE   : curve = a cubic spline through your dots ONLY, independent of the
           loaded profile (still passes through every dot).
  POLY   : curve = a single least-squares polynomial of the chosen degree fitted
           to your dots, independent of the loaded profile. Degree 0 → flat,
           1 → straight line, 2 → parabola, 3 → cubic, etc. Because POLY is a
           global fit, use FEWER velocity dots ('-') for each dot to influence
           the shape more strongly.
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

# Pre-seeded control-dot spacing along the track [m]. Smaller → finer, more local
# edits; larger → smoother, broader edits. Adjustable live with the keys below.
SEED_SPACING_PATH_M = 1.0
SEED_SPACING_VEL_M  = 1.5
SEED_SPACING_MIN_M  = 0.3
SEED_SPACING_MAX_M  = 8.0

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


def seed_indices(s: np.ndarray, spacing_m: float) -> List[int]:
    """Indices of waypoints spaced ~`spacing_m` apart along arc length.

    Endpoints (0 and n-1) are always included so the seeded grid spans the whole
    track. Returns a sorted, de-duplicated list.
    """
    n = len(s)
    total = float(s[-1] - s[0])
    if total <= 0 or spacing_m <= 0:
        return [0, n - 1]
    targets = np.arange(s[0], s[-1] + spacing_m * 0.5, spacing_m)
    idx = sorted({int(np.argmin(np.abs(s - t))) for t in targets} | {0, n - 1})
    return idx


# ── velocity panel editor (1-D profile vs arc length) ──────────────────────────

class ProfileEditor:
    """Velocity editor that HUGS the loaded profile, with a pre-seeded dot grid.

    The curve is `base + correction`, where base is the profile loaded from the
    CSV and the correction is a spline through the control points' offsets from
    base. Control dots are pre-seeded ON the loaded profile (offset 0), so the
    editor opens reproducing the optimized velocity exactly. Because the seeded
    neighbours are pinned close together, dragging one dot only changes the
    velocity in that LOCAL stretch.
    """

    DRAG_FRAC = 0.04

    def __init__(self, s, init_vals, ax, *, color, title, ylabel, ylim,
                 original=None, seed_spacing=SEED_SPACING_VEL_M):
        self.s, self.ax = s, ax
        self.init_vals = init_vals.copy()
        self.base = init_vals.copy()          # loaded profile the curve hugs
        self.color, self.title, self.ylabel, self.ylim = color, title, ylabel, ylim
        self.original = original
        self.seed_spacing = seed_spacing
        self.mode = 'hug'        # 'hug' | 'free' | 'poly'
        self.poly_deg = 1        # used only in 'poly' mode
        self._ctrl: List[CtrlPt] = []
        self._drag: Optional[int] = None
        self._cached = init_vals.copy()
        self.reseed(seed_spacing)

    # control dots placed exactly on the loaded profile (offset 0 everywhere)
    def reseed(self, spacing_m: float):
        self.seed_spacing = float(np.clip(spacing_m, SEED_SPACING_MIN_M, SEED_SPACING_MAX_M))
        idx = seed_indices(self.s, self.seed_spacing)
        self._ctrl = [(float(self.s[i]), float(self.base[i])) for i in idx]
        self._drag = None

    def eff_degree(self) -> int:
        """Polynomial degree actually usable given the number of dots."""
        return int(np.clip(self.poly_deg, 0, max(len(self._ctrl) - 1, 0)))

    def compute(self):
        c = sorted(self._ctrl, key=lambda p: p[0])
        cs = np.array([p[0] for p in c])
        cv = np.array([p[1] for p in c])
        if self.mode == 'poly':
            # global least-squares polynomial through the dots — independent of
            # the loaded profile. deg 0 = flat, 1 = linear, 2 = quadratic, …
            deg = self.eff_degree()
            try:
                curve = np.polyval(np.polyfit(cs, cv, deg), self.s)
            except Exception:
                curve = spline_through(cs, cv, self.s)
        elif self.mode == 'free':
            # cubic spline through the dots only — independent of the loaded
            # profile, but still passes through every dot.
            curve = spline_through(cs, cv, self.s)
        else:  # 'hug' — follow the loaded profile's tendency (original behaviour)
            corr = spline_through(cs, cv - np.interp(cs, self.s, self.base), self.s)
            curve = self.base + corr
        self._cached = np.clip(curve, self.ylim[0], self.ylim[1])
        return self._cached

    def mode_label(self) -> str:
        if self.mode == 'poly':
            return f'POLY deg {self.eff_degree()}'
        return 'FREE spline' if self.mode == 'free' else 'HUG prev'

    def cycle_mode(self) -> str:
        order = ['hug', 'free', 'poly']
        self.mode = order[(order.index(self.mode) + 1) % len(order)]
        return self.mode

    def change_degree(self, delta: int) -> int:
        self.poly_deg = int(np.clip(self.poly_deg + delta, 0, 8))
        self.mode = 'poly'        # changing degree implies you want POLY
        return self.poly_deg

    @property
    def current(self):
        return self._cached

    def draw(self):
        vals = self.compute()
        ax = self.ax
        ax.cla()
        ax.set_title(f'{self.title}  [{self.mode_label()}]',
                     color=self.color, fontweight='bold', fontsize=10)
        ax.set_xlabel('Arc Length [m]', fontsize=8)
        ax.set_ylabel(self.ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(float(self.s[0]), float(self.s[-1]))
        ax.set_ylim(*self.ylim)
        if self.original is not None:
            ax.plot(self.s, self.original, color='lightgray', lw=1.2, zorder=1, label='original')
        ax.plot(self.s, vals, color=self.color, lw=2.0, zorder=2, label='spline')
        c = sorted(self._ctrl, key=lambda p: p[0])
        ax.scatter([p[0] for p in c], [p[1] for p in c], c='crimson', s=42,
                   zorder=5, edgecolors='white', linewidths=1.0)
        ax.legend(fontsize=7, loc='upper right')
        return vals

    def reset(self):
        self.reseed(self.seed_spacing)

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

    def __init__(self, csv_path, output_path,
                 seed_spacing_path=SEED_SPACING_PATH_M,
                 seed_spacing_vel=SEED_SPACING_VEL_M):
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
        self.seed_spacing_path = seed_spacing_path
        self.ey_ctrl: List[Tuple[int, float]] = []
        self._reseed_path(seed_spacing_path)
        self._map_drag: Optional[int] = None

        # ── figure ──────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(18, 10))
        try:
            self.fig.canvas.manager.set_window_title('Waypoint Spline Editor v2 — local path + speed')
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
            color='royalblue', title='Velocity  (drag dots — local)',
            ylabel='vx_mps [m/s]', ylim=(0.0, max(float(v_orig.max()) * 1.2, 5.0)),
            original=v_orig, seed_spacing=seed_spacing_vel,
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

    # ── seeding ──────────────────────────────────────────────────────────────
    def _reseed_path(self, spacing_m: float):
        """Pre-seed lateral control dots ON the original line (ey=0)."""
        self.seed_spacing_path = float(np.clip(spacing_m, SEED_SPACING_MIN_M, SEED_SPACING_MAX_M))
        idx = seed_indices(self.s, self.seed_spacing_path)
        self.ey_ctrl = [(i, 0.0) for i in idx]
        self._map_drag = None

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
        ax.set_title('Track Map — drag dots to move the PATH locally (colour = velocity)', fontsize=10)
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

        # ey control dots on the map — moved dots highlighted brighter/larger
        for idx, e in self.ey_ctrl:
            cx, cy = self._ctrl_map_pos(idx, e)
            moved = abs(e) > 1e-6
            ax.scatter([cx], [cy],
                       c='crimson' if moved else 'darkorange',
                       s=70 if moved else 40, zorder=6,
                       edgecolors='white', linewidths=1.2)
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
        hint = (f'MAP dots={len(self.ey_ctrl)} (spc {self.seed_spacing_path:.1f}m, [ ])  ·  '
                f'VEL dots={len(self.vel_ed._ctrl)} (spc {self.vel_ed.seed_spacing:.1f}m, - =)  ·  '
                f'VEL mode [{self.vel_ed.mode_label()}] (M cycle, , . deg)  ·  '
                'L-drag move  R-click remove  ·  S save  R reset')
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
        elif event.key == '[':
            self._reseed_path(self.seed_spacing_path * 1.5)
            print(f'[path] coarser dots → spacing {self.seed_spacing_path:.2f} m '
                  f'({len(self.ey_ctrl)} dots)')
            self._refresh()
        elif event.key == ']':
            self._reseed_path(self.seed_spacing_path / 1.5)
            print(f'[path] finer dots → spacing {self.seed_spacing_path:.2f} m '
                  f'({len(self.ey_ctrl)} dots)')
            self._refresh()
        elif event.key == '-':
            self.vel_ed.reseed(self.vel_ed.seed_spacing * 1.5)
            print(f'[vel] coarser dots → spacing {self.vel_ed.seed_spacing:.2f} m '
                  f'({len(self.vel_ed._ctrl)} dots)')
            self._refresh()
        elif event.key in ('=', '+'):
            self.vel_ed.reseed(self.vel_ed.seed_spacing / 1.5)
            print(f'[vel] finer dots → spacing {self.vel_ed.seed_spacing:.2f} m '
                  f'({len(self.vel_ed._ctrl)} dots)')
            self._refresh()
        elif event.key == 'm':
            self.vel_ed.cycle_mode()
            print(f'[vel] shape mode → {self.vel_ed.mode_label()}')
            self._refresh()
        elif event.key == '.':
            self.vel_ed.change_degree(+1)
            print(f'[vel] {self.vel_ed.mode_label()}')
            self._refresh()
        elif event.key == ',':
            self.vel_ed.change_degree(-1)
            print(f'[vel] {self.vel_ed.mode_label()}')
            self._refresh()

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
        self._reseed_path(self.seed_spacing_path)
        print('[reset] restored original line + speed, re-seeded dot grid')
        self._refresh()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    csv = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(CSV_PATH)
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.abspath(OUTPUT_PATH)
    if not os.path.isfile(csv):
        sys.exit(f'[error] CSV not found: {csv}')
    print(f'[load]  {csv}')
    print(f'[save]  {out}')
    print('[keys]  S save · R reset · [ ] path dots · - = velocity dots')
    print('[keys]  M velocity mode (HUG/FREE/POLY) · , . poly degree')
    WaypointEditor(csv, out)
