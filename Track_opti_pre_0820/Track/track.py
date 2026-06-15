import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from Track.track_utils import *


class Track():
    def __init__(self, track_path, track_file='centerline', traj_type=None):
		
        self.track_path = track_path
        if traj_type is None:
            center = np.loadtxt(track_path+'centerline.txt', delimiter=",", dtype = float)
            # self.center = interpolate(center, 0.5)
            self.center = center
            self.calc_track_info()

        else:
            opt_traj = np.loadtxt(track_path+track_file+'_frenet.txt', delimiter=",", dtype = float) #Frenet coordinate
            self.opt = np.loadtxt(track_path+track_file+'.txt', delimiter=",", dtype = float) #(x-y) coordinate
            self.center = np.loadtxt(track_path+'centerline_.txt', delimiter=",", dtype = float)
            self.calc_track_info()
            self.curv_center = self.center[:,-1]
            
            if len(opt_traj) != len(self.s_center):
                self.opt, opt_traj = self.interpolate(opt_traj)
            
            opt_traj_ = opt_traj
            self.s, self.ey, self.epsi, self.v = opt_traj_[:,0], opt_traj_[:,1], opt_traj_[:,2], opt_traj_[:,3]

            if opt_traj.shape[1] > 6: # Dynamic Model
                self.vy, self.psidot = opt_traj_[:,4], opt_traj_[:,5]
                self.sdot = opt_traj_[:,6]

            self.s = np.linspace(0, self.track_length*2, len(opt_traj)*2)
            self.opt_traj = opt_traj
            
            
        self.inner = np.loadtxt(track_path+'innerwall.txt', delimiter=",", dtype = float)
        self.outer = np.loadtxt(track_path+'outerwall.txt', delimiter=",", dtype = float)

        self.track_width_scale = 0.8
        self.track_width = 2
        
    def interpolate2(self, traj):
        opt_fren = np.zeros((len(self.s_center), traj.shape[1])) 
        opt_xy = np.zeros((len(self.s_center), traj.shape[1])) 
        for i in range(len(self.s_center)):
            near_idx_array = np.where(traj[:,0] >= self.s_center[i])
            if len(near_idx_array[0]) >= 1:
                near_idx = near_idx_array[0][0]

                if traj[near_idx,0] == self.s_center[i]:
                    opt_fren[i] = traj[near_idx]
                    opt_xy[i,3:] = opt_fren[i, 3:]
                else:
                    s1, s2 = traj[near_idx-1,0], traj[near_idx,0]
                    t = (self.s_center[i] - s1) / (s2 - s1)
                    opt_fren[i, 0] = self.s_center[i]

                    for j in range(1, traj.shape[1]):
                        state1, state2 = traj[near_idx-1,j], traj[near_idx,j]
                        opt_fren[i,j] = state1 + t * (state2 - state1)
                        
                        if j >= 3:
                            opt_xy[i,j] = opt_fren[i, j]
            else:
                print(f"Fail at {i}")
                opt_fren[i] = traj[0]
                opt_xy[i,3:] = opt_fren[0, 3:]

            px, py, psi = self.local_to_global(np.array([opt_fren[i, 0], opt_fren[i, 1], opt_fren[i, 2]]))
            opt_xy[i,:3] = [px, py, psi]
    
        return opt_xy, opt_fren

    def interpolate(self, traj):
        s = traj[:, 0]  # trajectory progress (assumed sorted and monotonic)
        s_uniform = self.s_center
        n_samples = len(s_uniform)

        interpolated_traj = np.zeros((n_samples, traj.shape[1]))

        interpolated_traj[0] = traj[0]
        
        for i in range(traj.shape[1]):
            f = interp1d(s[:], traj[:, i], kind='linear', fill_value="extrapolate")
            interpolated_traj[1:, i] = f(s_uniform[1:])

        # local_to_global 적용
        opt_xy = np.zeros((n_samples, traj.shape[1]))
        opt_xy[0, 0],opt_xy[0, 1],opt_xy[0, 2]= self.local_to_global([interpolated_traj[0,0],interpolated_traj[0,1],interpolated_traj[0,2]])
        opt_xy[0, 3:] = interpolated_traj[0, 3:]
        for i in range(1, n_samples):
            px, py, psi = self.local_to_global(interpolated_traj[i, :3])
            opt_xy[i, :3] = [px, py, psi]
            opt_xy[i, 3:] = interpolated_traj[i, 3:]

        return opt_xy, interpolated_traj

    def linspace_s(self, N=100):
        ''' linspace s and caculate the (x,y) and curvature along the linspaced s'''
        s = np.linspace(0,self.track_length-self.d_dist, N)

        # Compute the curvature for s
        traj = np.zeros((len(s),3))
        for i in range(0,len(s)):
            traj[i,0],traj[i,1],_ = self.local_to_global([s[i],0,0])

        self.center = traj
        self.calc_track_info()
        traj[:,-1] = self.curv_center

        # plt.figure()
        # t = np.linspace(0,len(traj),len(traj))
        # plt.plot(t, self.curv_center)

        np.savetxt(f'{self.track_path}/centerline_.txt', self.center, delimiter=",")

    def calc_track_info(self): #Compute s, psi, track_length, curvature
        dx = np.gradient(self.center[:,0])
        dy = np.gradient(self.center[:,1])
        ds = np.sqrt(dx**2 + dy**2)
        self.d_dist = ds[0]
        self.track_length = np.sum(ds)

        self.s_center = np.insert(np.cumsum(ds), 0, 0)[:-1]
        self.ey_center = np.zeros((len(self.s_center),))
        self.psi_center = np.arctan2(dy,dx)
        self.curv_center = self.calc_track_curv(self.center)

        
    def calc_track_curv(self, traj):
        # Apply Gaussian filter to smooth the trajectory data
        N = len(traj) 
        x = gaussian_filter1d(traj[:,0], sigma=2, mode='wrap')
        y = gaussian_filter1d(traj[:,1], sigma=2, mode='wrap')
        traj = np.column_stack([x, y])

        # dx = np.gradient(traj[:,0])
        # dy = np.gradient(traj[:,1])
        # d2x = np.gradient(dx)
        # d2y = np.gradient(dy)
        # curv = (dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5

        # curv = curv[:N]
        # idx = np.where(abs(curv) < 0.01)
        # curv[idx[0]] =  1e-5

        offset = 5

        idx_p = [(i+offset) % N for i in range(N)]
        idx_m = [(i-offset) % N for i in range(N)]
        
        dx  = (traj[idx_p,0] - traj[idx_m,0]) / (2*offset)
        dy  = (traj[idx_p,1] - traj[idx_m,1]) / (2*offset)

        d2x = (traj[idx_p,0] - 2*traj[:,0] + traj[idx_m,0]) / (offset**2)
        d2y = (traj[idx_p,1] - 2*traj[:,1] + traj[idx_m,1]) / (offset**2)

        curv = (dx * d2y - d2x * dy) / (dx**2 + dy**2)**1.5
        return curv
    
    def local_to_global(self, cl_coord):

        s = cl_coord[0]
        while s < 0: s += self.track_length
        while s >= self.track_length: s -= self.track_length

        e_y = cl_coord[1]
        e_psi = cl_coord[2]


        if s <= self.s_center[0]:
            idx_s = 0
        else:
            if len(np.where(s >= self.s_center)[0]) > 0:
                idx_s = np.where(s >= self.s_center)[0][-1]
            else:
                idx_s = -1


        x_s = self.center[idx_s,0]
        y_s = self.center[idx_s,1]
        psi_s = self.psi_center[idx_s]
        curv = self.curv_center[idx_s]
        d = s - self.s_center[idx_s]

        if curv == 0.0:
            curv = 1e-5
            
        r = 1 / curv
        dir = np.sign(r)

        # Find coordinates for center of curved segment
        x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
        y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)

        # Angle spanned up to current location along segment
        span_ang = d / np.abs(r)

        # Angle of the tangent vector at the current location
        psi_d = wrap_angle(psi_s + dir * span_ang)

        ang_norm = wrap_angle(psi_s + dir * np.pi / 2)
        ang = -np.sign(ang_norm) * (np.pi - np.abs(ang_norm))

        x = x_c + (np.abs(r) - dir * e_y) * np.cos(ang + dir * span_ang)
        y = y_c + (np.abs(r) - dir * e_y) * np.sin(ang + dir * span_ang)
        psi = wrap_angle(psi_d + e_psi)
        
        return (x, y, psi)

    def global_to_local(self, xy_coord):
        x = xy_coord[0]
        y = xy_coord[1]
        psi = xy_coord[2]

        pos_cur = np.array([x, y])
        cl_coord = None

        for i in range(len(self.center) - 1):
            x_s = self.center[i, 0]
            y_s = self.center[i, 1]
            psi_s = self.psi_center[i]
            curve_s = self.curv_center[i]

            x_f = self.center[i + 1, 0]
            y_f = self.center[i + 1, 1]
            psi_f = self.psi_center[i + 1]
            curve_f = self.curv_center[i + 1]

            l = self.s[i + 1] - self.s[i]

            pos_s = np.array([x_s, y_s])
            pos_f = np.array([x_f, y_f])

            # Check if at any of the segment start or end points
            if la.norm(pos_s - pos_cur) == 0:
                # At start of segment
                s = self.s[i]
                ey = 0
                epsi = np.unwrap([psi_s, psi])[1] - psi_s
                cl_coord = (s, ey, epsi)
                break

            if la.norm(pos_f - pos_cur) == 0:
                # At end of segment
                s = self.s[i + 1]
                ey = 0
                epsi = np.unwrap([psi_f, psi])[1] - psi_f
                cl_coord = (s, ey, epsi)
                break

            if curve_f == 0:
                # Check if on straight segment
                if (
                    np.abs(compute_angle(pos_s, pos_cur, pos_f)) <= np.pi / 2
                    and np.abs(compute_angle(pos_f, pos_cur, pos_s)) <= np.pi / 2
                ):
                    v = pos_cur - pos_s
                    ang = compute_angle(pos_s, pos_f, pos_cur)
                    ey = la.norm(v) * np.sin(ang)

                    if np.abs(ey) <= self.track_width / 2:
                        d = la.norm(v) * np.cos(ang)
                        s = self.s[i] + d
                        epsi = np.unwrap([psi_s, psi])[1] - psi_s
                        cl_coord = (s, ey, epsi)
                        break
                    else:
                        continue
                else:
                    continue
            else:
                # Check if on curved segment
                r = 1 / curve_f
                dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
                y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)
                curve_center = np.array([x_c, y_c])

                span_ang = l / r
                cur_ang = compute_angle(curve_center, pos_s, pos_cur)

                # Updated comparison to handle potential ambiguity with arrays
                span_sign = np.sign(span_ang)
                cur_sign = np.sign(cur_ang)

                # if (
                #     span_sign == cur_sign and
                #     np.abs(span_ang) >= np.abs(cur_ang)
                # ):
                #     v = pos_cur - curve_center
                #     ey = -np.sign(dir) * (la.norm(v) - np.abs(r))

                #     if np.abs(ey) <= self.track_width / 2:
                #         d = np.abs(cur_ang) * np.abs(r)
                #         s = self.s[i] + d
                #         epsi = np.unwrap([psi_s + cur_ang, psi])[1] - (psi_s + cur_ang)
                #         cl_coord = (s, ey, epsi)
                #         break
                #     else:
                #         continue
                # else:
                #     continue

        return cl_coord

    def global_to_frenet(self, global_coords):
            """
            Convert global coordinates (x, y, psi) to Frenet coordinates (s, ey, epsi) based on the centerline.

            Parameters:
                centerline (numpy.ndarray): Array of shape (N, 2) representing the centerline (x, y).
                global_coords (tuple): Global coordinates as (x, y, psi).

            Returns:
                tuple: Frenet coordinates (s, ey, epsi).
            """
            x, y, psi = global_coords
            self.center=self.center[:,:2]
            # Calculate distances to all points on the centerline
            distances = la.norm(self.center[:] - np.array([x, y]), axis=1)

            # Find the index of the closest point on the centerline
            closest_idx = np.argmin(distances)

            # Get the closest point and the next point on the centerline
            closest_point = self.center[closest_idx]
            next_idx = (closest_idx + 1) % len(self.center)
            next_point = self.center[next_idx]
            # Calculate the tangent vector at the closest point
            tangent_vector = next_point - closest_point
            tangent_vector /= la.norm(tangent_vector)  # Normalize the vector

            # Calculate s (arc length along the centerline)
            s = np.sum(la.norm(np.diff(self.center[:closest_idx + 1], axis=0), axis=1))

            # Calculate the vector from the closest point to the global point
            vector_to_point = np.array([x, y]) - closest_point

            # Compute ey (lateral distance) using the cross product
            ey = np.cross(tangent_vector, vector_to_point)

            # Compute epsi (heading error)
            tangent_angle = np.arctan2(tangent_vector[1], tangent_vector[0])
            epsi = wrap_angle(psi - tangent_angle)

            return s, ey, epsi

    def wrap_angle(self,angle):
            """Wrap angle to the range [-pi, pi]."""
            return (angle + np.pi) % (2 * np.pi) - np.pi
