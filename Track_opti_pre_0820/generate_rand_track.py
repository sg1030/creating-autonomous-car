import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import cv2
import os
import shapely.geometry as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from utils import *
# PARAMETERS
TRACK_WIDTH = 2.5           # [m]
TRACK_RADIUS = 5 # 평균 반지름 (실내 사이즈)
CONTROL_PTS = 20          # control point 개수 (복잡도 조절)
RADIUS_NOISE = 1.5    # 모양 다양성
N_POINTS = 300              # spline point 수
OUTPUT_DIR = './'

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_MAPS = 20
def create_track(track_radius):
    SCALE = 10.0
    TRACK_DETAIL_STEP = 0.1
    TRACK_TURN_RATE = 0.3 #np.radians(30)

    start_alpha = 0.

    # Create checkpoints
    checkpoints = []
    for c in range(CONTROL_PTS):
        alpha = 2*np.pi*c/CONTROL_PTS + np.random.uniform(0, 2*np.pi*1/CONTROL_PTS)
        rad = np.random.uniform(track_radius/5, track_radius)
        if c==0:
            alpha = 0
            rad = RADIUS_NOISE*track_radius
        if c==CONTROL_PTS-1:
            alpha = 2*np.pi*c/CONTROL_PTS
            start_alpha = 2*np.pi*(-0.5)/CONTROL_PTS
            rad = RADIUS_NOISE*track_radius
        checkpoints.append( (alpha, rad*np.cos(alpha), rad*np.sin(alpha)) )
    road = []

    # Go from one checkpoint to another to create track
    x, y, beta = 2*track_radius, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = np.arctan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2*np.pi
        while True:
            failed = True
            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break
            if not failed:
                break
            alpha -= 2*np.pi
            continue
        r1x = np.cos(beta)
        r1y = np.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x
        dest_dy = dest_y - y
        proj = r1x*dest_dx + r1y*dest_dy
        while beta - alpha >  1.5*np.pi:
            beta -= 2*np.pi
        while beta - alpha < -1.5*np.pi:
            beta += 2*np.pi
        prev_beta = beta
        proj *= SCALE
        if proj >  0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001*proj))
        x += p1x*TRACK_DETAIL_STEP
        y += p1y*TRACK_DETAIL_STEP
        track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze==0:
            break

    # Find closed loop
    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i==0:
            return False
        pass_through_start = track[i][0] > start_alpha and track[i-1][0] <= start_alpha
        if pass_through_start and i2==-1:
            i2 = i
        elif pass_through_start and i1==-1:
            i1 = i
            break
    print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
    assert i1!=-1
    assert i2!=-1

    track = track[i1:i2-1]
    first_beta = track[0][1]
    first_perp_x = np.cos(first_beta)
    first_perp_y = np.sin(first_beta)

    # Length of perpendicular jump to put together head and tail
    well_glued_together = np.sqrt(
        np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
        np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
    if well_glued_together > TRACK_DETAIL_STEP:
        return False

    # post processing, converting to numpy, finding exterior and interior walls
    track_xy = [(x, y) for (a1, b1, x, y) in track]
    track_xy = np.asarray(track_xy)
    track_poly = shp.Polygon(track_xy)

    track_xy_offset_in = track_poly.buffer(TRACK_WIDTH/2)
    track_xy_offset_out = track_poly.buffer(-TRACK_WIDTH/2)

    # ✅ Check that result is a valid Polygon
    if not isinstance(track_xy_offset_in, shp.Polygon) or not isinstance(track_xy_offset_out, shp.Polygon):
        print("❌ Buffer result is not a Polygon")
        return False

    track_xy_offset_in_np = np.array(track_xy_offset_in.exterior.coords)
    track_xy_offset_out_np = np.array(track_xy_offset_out.exterior.coords)

    return track_xy, track_xy_offset_in_np, track_xy_offset_out_np

def generate_nonclustered_angles(n, min_angle=np.pi/15):
    angles = []
    attempts = 0
    while len(angles) < n and attempts < 1000:
        candidate = np.random.uniform(0, 2*np.pi)
        if all(abs((candidate - a + np.pi) % (2*np.pi) - np.pi) > min_angle for a in angles):
            angles.append(candidate)
        attempts += 1
    return np.sort(np.array(angles))

def generate_random_closed_track():

    angles = generate_nonclustered_angles(CONTROL_PTS)
    radii = TRACK_RADIUS + np.random.uniform(-RADIUS_NOISE, RADIUS_NOISE, CONTROL_PTS)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # 부드러운 폐곡선 spline 생성
    tck, _ = splprep([x, y], s=2, per=True)
    u_fine = np.linspace(0, 1, N_POINTS)
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack([x_fine, y_fine]).T

def resample_centerline(centerline):
    # 누적 거리 계산
    diffs = np.diff(centerline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumulative = np.insert(np.cumsum(dists), 0, 0)
    total_length = cumulative[-1]

    # 등간격 포인트
    u = np.linspace(0, total_length, N_POINTS)
    x_interp = np.interp(u, cumulative, centerline[:, 0])
    y_interp = np.interp(u, cumulative, centerline[:, 1])
    return np.vstack([x_interp, y_interp]).T

def compute_offset_curves(centerline, width):
    # tangent 방향 계산
    diffs = np.roll(centerline, -1, axis=0) - centerline
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6  # 나눗셈 방지
    tangents = diffs / norms
    normals = np.stack([-tangents[:,1], tangents[:,0]], axis=1)
    normals[-1] = normals[0]  # 연속성 보장

    inner = centerline - width/2 * normals
    outer = centerline + width/2 * normals

    inner = smooth_curve(inner, s=1.5)
    outer = smooth_curve(outer, s=3)
    return outer, inner

def smooth_curve(curve, s=0.5):
    tck, _ = splprep([curve[:, 0], curve[:, 1]], s=s, per=True)
    u_fine = np.linspace(0, 1, len(curve))
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack([x_fine, y_fine]).T

if __name__ == "__main__":
    i = 0
    while(1):
        # center = generate_random_closed_track()
        # center = resample_centerline(center[:-1], N_POINTS)
        # outer, inner = compute_offset_curves(center, TRACK_WIDTH)
        track_radius = np.random.randint(5,8) 
        result = create_track(track_radius)
        if result is False:
            print(f"[{i}] Invalid track, retrying...")
            continue
        center, inner, outer = result
        
        map_dir = f"./MAP{i}"
        create_directory_structure(map_dir)
        save_track_txt(center, inner, outer, map_dir)
        plot_track(center, inner, outer, os.path.join(map_dir, "track.png"))
        i = i + 1
        if i > 1:
            break
