#!/usr/bin/env python3
"""
Interactive spline editor for global_waypoints.csv.

Two editable profiles:
  - vx_mps  : velocity at each waypoint
  - ey      : lateral offset from the original line (+ = left of heading)

Controls:
  V            switch to velocity editor
  E            switch to lateral-offset (ey) editor
  Left-click   add control point on the active profile panel
  Right-click  remove nearest control point (minimum 2 always kept)
  Drag         move a control point
  S            save edited CSV  →  global_waypoints_edited.csv
  R            reset everything to original
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(_HERE, '..', 'maps', 'final', 'global_waypoints.csv')
OUTPUT_PATH = os.path.join(_HERE, '..', 'maps', 'final', 'global_waypoints_edited.csv')

CtrlPt = Tuple[float, float]   # (arc_length, value)


# ── helpers ───────────────────────────────────────────────────────────────────

def arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(d)])


def spline_interp(ctrl: List[CtrlPt], s_query: np.ndarray) -> np.ndarray:
    """Cubic spline through sorted control points, evaluated at s_query."""
    ctrl = sorted(ctrl, key=lambda p: p[0])
    xs = np.array([p[0] for p in ctrl])
    ys = np.array([p[1] for p in ctrl])

    if len(xs) < 2:
        return np.full_like(s_query, ys[0] if len(ys) else 0.0, dtype=float)
    if len(xs) == 2:
        return np.interp(s_query, xs, ys).astype(float)
    try:
        cs = CubicSpline(xs, ys, bc_type='not-a-knot')
        return cs(s_query).astype(float)
    except Exception:
        return np.interp(s_query, xs, ys).astype(float)


# ── one-panel editor ──────────────────────────────────────────────────────────

class ProfileEditor:
    """Manages control points and spline for a single 1-D profile."""

    DRAG_FRAC = 0.035   # normalised distance threshold for grab

    def __init__(self,
                 s: np.ndarray,
                 init_vals: np.ndarray,
                 ax: plt.Axes,
                 *,
                 color: str,
                 title: str,
                 ylabel: str,
                 ylim: Tuple[float, float],
                 original: Optional[np.ndarray] = None):
        self.s         = s
        self.init_vals = init_vals.copy()
        self.ax        = ax
        self.color     = color
        self.title     = title
        self.ylabel    = ylabel
        self.ylim      = ylim
        self.original  = original
        self.active    = False

        self._ctrl: List[CtrlPt] = [(float(s[0]),  float(init_vals[0])),
                                     (float(s[-1]), float(init_vals[-1]))]
        self._drag_idx: Optional[int] = None
        self._cached   = init_vals.copy()

    # ── public ────────────────────────────────────────────────────────────────

    def compute(self) -> np.ndarray:
        self._cached = spline_interp(self._ctrl, self.s)
        return self._cached

    @property
    def current(self) -> np.ndarray:
        return self._cached

    def draw(self) -> np.ndarray:
        vals = self.compute()
        ax = self.ax
        ax.cla()

        suffix = '  ◀ ACTIVE' if self.active else ''
        ax.set_title(f'{self.title}{suffix}',
                     color=self.color if self.active else 'black',
                     fontweight='bold' if self.active else 'normal',
                     fontsize=10)
        ax.set_xlabel('Arc Length [m]', fontsize=8)
        ax.set_ylabel(self.ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(float(self.s[0]), float(self.s[-1]))
        ax.set_ylim(*self.ylim)

        if self.original is not None:
            ax.plot(self.s, self.original, color='lightgray', lw=1.2,
                    zorder=1, label='original')

        ax.axhline(0, color='#aaa', lw=0.6, zorder=1)
        ax.plot(self.s, vals, color=self.color, lw=2.0, zorder=2, label='spline')

        ctrl_sorted = sorted(self._ctrl, key=lambda p: p[0])
        cx = [p[0] for p in ctrl_sorted]
        cy = [p[1] for p in ctrl_sorted]
        dot_c = 'crimson' if self.active else '#888'
        ax.scatter(cx, cy, c=dot_c, s=110, zorder=5,
                   edgecolors='white', linewidths=1.5)

        ax.legend(fontsize=7, loc='upper right')
        return vals

    def reset(self, init_vals: Optional[np.ndarray] = None):
        v = init_vals if init_vals is not None else self.init_vals
        self._ctrl = [(float(self.s[0]), float(v[0])),
                      (float(self.s[-1]), float(v[-1]))]
        self._drag_idx = None

    # ── events (return True when redraw needed) ───────────────────────────────

    def on_press(self, x: float, y: float, button: int) -> bool:
        if button == 1:             # left – drag or add
            idx = self._nearest(x, y)
            if idx is not None:
                self._drag_idx = idx
                return False        # start drag, no immediate redraw
            # add new point snapped to nearest waypoint
            si = int(np.argmin(np.abs(self.s - x)))
            y_c = float(np.clip(y, self.ylim[0], self.ylim[1]))
            self._ctrl.append((float(self.s[si]), y_c))
            self._ctrl.sort(key=lambda p: p[0])
            return True
        if button == 3:             # right – remove
            idx = self._nearest(x, y)
            if idx is not None and len(self._ctrl) > 2:
                self._ctrl.pop(idx)
                return True
        return False

    def on_drag(self, x: float, y: float) -> bool:
        if self._drag_idx is None:
            return False
        si = int(np.argmin(np.abs(self.s - x)))
        y_c = float(np.clip(y, self.ylim[0], self.ylim[1]))
        self._ctrl[self._drag_idx] = (float(self.s[si]), y_c)
        return True

    def on_release(self) -> bool:
        if self._drag_idx is None:
            return False
        self._ctrl.sort(key=lambda p: p[0])
        self._drag_idx = None
        return True

    # ── private ───────────────────────────────────────────────────────────────

    def _nearest(self, x: float, y: float) -> Optional[int]:
        if not self._ctrl:
            return None
        s_rng = float(self.s[-1] - self.s[0]) or 1.0
        y_rng = float(self.ylim[1] - self.ylim[0]) or 1.0
        dists = [np.hypot((p[0] - x) / s_rng, (p[1] - y) / y_rng)
                 for p in self._ctrl]
        i = int(np.argmin(dists))
        return i if dists[i] < self.DRAG_FRAC else None


# ── main application ──────────────────────────────────────────────────────────

class WaypointEditor:

    def __init__(self, csv_path: str, output_path: str):
        self.csv_path    = csv_path
        self.output_path = output_path

        df = pd.read_csv(csv_path)
        self.df_orig = df
        self.s = arc_length(df['x_m'].values, df['y_m'].values)

        # ── figure ────────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(18, 10))
        try:
            self.fig.canvas.manager.set_window_title('Waypoint Spline Editor')
        except Exception:
            pass
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                                hspace=0.45, wspace=0.30,
                                left=0.06, right=0.97, top=0.92, bottom=0.06)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_vel = self.fig.add_subplot(gs[0, 1])
        self.ax_ey  = self.fig.add_subplot(gs[1, 1])

        # ── profile editors ───────────────────────────────────────────────────
        v_orig = df['vx_mps'].values
        v_top  = max(float(v_orig.max()) * 1.2, 10.0)
        self.vel_ed = ProfileEditor(
            self.s, v_orig, self.ax_vel,
            color='royalblue',
            title='Velocity  [press V]',
            ylabel='vx_mps [m/s]',
            ylim=(0.0, v_top),
            original=v_orig,
        )

        half_w = max(float(df['w_tr_right_m'].max()),
                     float(df['w_tr_left_m'].max())) * 0.95
        self.ey_ed = ProfileEditor(
            self.s, np.zeros(len(df)), self.ax_ey,
            color='forestgreen',
            title='Lateral Offset ey  [press E]',
            ylabel='ey [m]  (+ = left of heading)',
            ylim=(-half_w, half_w),
            original=np.zeros(len(df)),
        )

        self.vel_ed.active = True
        self._active: ProfileEditor = self.vel_ed

        # ── events ────────────────────────────────────────────────────────────
        c = self.fig.canvas
        c.mpl_connect('button_press_event',   self._on_press)
        c.mpl_connect('button_release_event', self._on_release)
        c.mpl_connect('motion_notify_event',  self._on_motion)
        c.mpl_connect('key_press_event',      self._on_key)

        self._refresh()
        plt.show()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _refresh(self):
        vel = self.vel_ed.draw()
        ey  = self.ey_ed.draw()
        self._draw_map(vel, ey)
        self._set_hint()
        self.fig.canvas.draw_idle()

    def _draw_map(self, vel: np.ndarray, ey: np.ndarray):
        ax = self.ax_map
        ax.cla()
        ax.set_title('Track Map  (colour = velocity)', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('y [m]', fontsize=8)

        df  = self.df_orig
        x0  = df['x_m'].values
        y0  = df['y_m'].values
        psi = df['psi_rad'].values
        wr  = df['w_tr_right_m'].values
        wl  = df['w_tr_left_m'].values

        # Heading  →  (cos ψ, sin ψ)
        # Right boundary: rotate 90° CW  →  ( sin ψ, −cos ψ)
        # Left  boundary: rotate 90° CCW → (−sin ψ,  cos ψ)
        ax.plot(x0 + wr * np.sin(psi), y0 - wr * np.cos(psi),
                'k-', lw=0.9, alpha=0.35)
        ax.plot(x0 - wl * np.sin(psi), y0 + wl * np.cos(psi),
                'k-', lw=0.9, alpha=0.35, label='boundaries')

        # Trajectory shifted by ey (positive ey → left → (−sin ψ, cos ψ))
        x_new = x0 - ey * np.sin(psi)
        y_new = y0 + ey * np.cos(psi)

        sc = ax.scatter(x_new, y_new, c=vel, cmap='RdYlGn',
                         s=14, vmin=0, vmax=float(self.vel_ed.ylim[1]),
                         zorder=3)
        self.fig.colorbar(sc, ax=ax, label='vx [m/s]', shrink=0.65, pad=0.02)
        ax.plot(x_new[0], y_new[0], 'b^', ms=10, zorder=6, label='start')
        ax.legend(fontsize=7, loc='upper right')

    def _set_hint(self, extra: str = ''):
        hint = ('V velocity | E lateral-ey  ·  '
                'Left-click add  Right-click remove  Drag move  ·  '
                'S save  R reset')
        msg  = f'{extra}   {hint}' if extra else hint
        col  = 'darkgreen' if extra else '#444'
        self.fig.suptitle(msg, fontsize=8.5, color=col)

    # ── event handlers ────────────────────────────────────────────────────────

    def _ed_for_ax(self, ax) -> Optional[ProfileEditor]:
        if ax == self.ax_vel:
            return self.vel_ed
        if ax == self.ax_ey:
            return self.ey_ed
        return None

    def _on_press(self, event):
        ed = self._ed_for_ax(event.inaxes)
        if ed is None or event.xdata is None:
            return
        if ed is not self._active:
            self._active.active = False
            ed.active = True
            self._active = ed
        if ed.on_press(event.xdata, event.ydata, event.button):
            self._refresh()

    def _on_motion(self, event):
        if event.inaxes != self._active.ax or event.xdata is None:
            return
        if self._active.on_drag(event.xdata, event.ydata):
            self._refresh()

    def _on_release(self, event):
        if self._active.on_release():
            self._refresh()

    def _on_key(self, event):
        k = event.key
        if k == 'v':
            self._switch(self.vel_ed)
        elif k == 'e':
            self._switch(self.ey_ed)
        elif k == 's':
            self._save()
        elif k == 'r':
            self._reset()

    def _switch(self, ed: ProfileEditor):
        self._active.active = False
        ed.active = True
        self._active = ed
        self._refresh()

    # ── save / reset ──────────────────────────────────────────────────────────

    def _save(self):
        vel = self.vel_ed.current
        ey  = self.ey_ed.current

        df  = self.df_orig.copy()
        psi = df['psi_rad'].values
        df['x_m']    = df['x_m'].values - ey * np.sin(psi)
        df['y_m']    = df['y_m'].values + ey * np.cos(psi)
        df['vx_mps'] = vel

        df.to_csv(self.output_path, index=False, float_format='%.6f')
        msg = f'✓ saved → {os.path.basename(self.output_path)}'
        print(f'[saved] {self.output_path}')
        self._set_hint(msg)
        self.fig.canvas.draw_idle()

    def _reset(self):
        self.vel_ed.reset()
        self.ey_ed.reset(np.zeros(len(self.df_orig)))
        self._set_hint()
        self._refresh()
        print('[reset] restored original values')


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    csv  = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(CSV_PATH)
    out  = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.abspath(OUTPUT_PATH)

    if not os.path.isfile(csv):
        sys.exit(f'[error] CSV not found: {csv}')

    print(f'[load]  {csv}')
    print(f'[save]  {out}')
    WaypointEditor(csv, out)
