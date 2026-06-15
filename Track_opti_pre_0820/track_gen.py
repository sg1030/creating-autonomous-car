import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import yaml
import os
import time
from scipy.interpolate import CubicSpline

class TrackSegment:
    def __init__(self, length, width, segment_type='straight', curvature=0):
        self.length = length  # 구간 길이
        self.width = width    # 트랙 폭
        self.type = segment_type  # 'straight' 또는 'curve'
        self.curvature = curvature  # 곡률 (curve인 경우)

class Track:
    def __init__(self):
        self.segments = []
        self.centerline = []
        self.boundaries = {'left': [], 'right': []}
        self.is_closed = False
        
    def add_segment(self, segment):
        self.segments.append(segment)
        self._update_track()
        
    def _update_track(self):
        # 센터라인과 경계선 계산
        x, y = 0, 0
        angle = 0
        
        # 리스트 초기화
        self.centerline = [(x, y)]
        self.boundaries['left'] = []
        self.boundaries['right'] = []
        
        # 모든 세그먼트에 대해 새로 계산
        for segment in self.segments:
            if segment.type == 'straight':
                new_x = x + segment.length * np.cos(angle)
                new_y = y + segment.length * np.sin(angle)
                
                # 직선 구간의 경계선 계산
                left_x = x + segment.width/2 * np.sin(angle)
                left_y = y - segment.width/2 * np.cos(angle)
                right_x = x - segment.width/2 * np.sin(angle)
                right_y = y + segment.width/2 * np.cos(angle)
                
                # 끝점도 추가
                left_end_x = new_x + segment.width/2 * np.sin(angle)
                left_end_y = new_y - segment.width/2 * np.cos(angle)
                right_end_x = new_x - segment.width/2 * np.sin(angle)
                right_end_y = new_y + segment.width/2 * np.cos(angle)
                
                self.boundaries['left'].extend([(left_x, left_y), (left_end_x, left_end_y)])
                self.boundaries['right'].extend([(right_x, right_y), (right_end_x, right_end_y)])
                
            elif segment.type == 'curve':
                # 곡선 반경 계산
                radius = abs(1/segment.curvature)
                direction = np.sign(segment.curvature)
                
                # 곡선의 중심점 계산 (방향에 따라 다르게 계산)
                if direction > 0:  # 반시계 방향 (왼쪽 커브)
                    center_x = x - radius * np.sin(angle)
                    center_y = y + radius * np.cos(angle)
                    start_angle = angle - np.pi/2
                else:  # 시계 방향 (오른쪽 커브)
                    center_x = x + radius * np.sin(angle)
                    center_y = y - radius * np.cos(angle)
                    start_angle = angle + np.pi/2
                
                # 곡선의 각도 계산
                arc_length = segment.length
                arc_angle = arc_length / radius
                if direction > 0:
                    end_angle = start_angle + arc_angle
                else:
                    end_angle = start_angle - arc_angle
                
                # 곡선 구간의 포인트들 생성
                num_points = 50
                t = np.linspace(start_angle, end_angle, num_points)
                
                # 센터라인 계산
                curve_x = center_x + radius * np.cos(t)
                curve_y = center_y + radius * np.sin(t)
                
                # 끝점 및 끝 각도 계산
                new_x = curve_x[-1]
                new_y = curve_y[-1]
                angle = end_angle + (np.pi/2 if direction > 0 else -np.pi/2)
                
                # 경계선 계산
                inner_radius = radius - segment.width/2
                outer_radius = radius + segment.width/2
                
                inner_x = center_x + inner_radius * np.cos(t)
                inner_y = center_y + inner_radius * np.sin(t)
                outer_x = center_x + outer_radius * np.cos(t)
                outer_y = center_y + outer_radius * np.sin(t)
                
                # 방향에 따라 경계선 할당
                if direction > 0:  # 반시계 방향
                    self.boundaries['left'].extend(list(zip(outer_x, outer_y)))
                    self.boundaries['right'].extend(list(zip(inner_x, inner_y)))
                else:  # 시계 방향
                    self.boundaries['left'].extend(list(zip(inner_x, inner_y)))
                    self.boundaries['right'].extend(list(zip(outer_x, outer_y)))
                
                # 센터라인에 곡선의 모든 점 추가
                for cx, cy in zip(curve_x, curve_y):
                    self.centerline.append((cx, cy))
            
            x, y = new_x, new_y
    
    def visualize(self):
        plt.figure(figsize=(10, 10))
        
        # 센터라인 그리기
        center_x, center_y = zip(*self.centerline)
        plt.plot(center_x, center_y, 'b-', label='Centerline')
        
        # 경계선 그리기
        left_x, left_y = zip(*self.boundaries['left'])
        right_x, right_y = zip(*self.boundaries['right'])
        plt.plot(left_x, left_y, 'r-', label='Left boundary')
        plt.plot(right_x, right_y, 'g-', label='Right boundary')
        
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save(self, filename):
        # PNG로 저장 (트랙만)
        plt.figure(figsize=(10, 10))
        
        # 센터라인 그리기 (검정 점선)
        center_x, center_y = zip(*self.centerline)
        plt.plot(center_x, center_y, '--k', label='Centerline')
        
        # 경계선 그리기 (검정 실선)
        left_x, left_y = zip(*self.boundaries['left'])
        right_x, right_y = zip(*self.boundaries['right'])
        plt.plot(left_x, left_y, 'k', label='Track Bounds')
        plt.plot(right_x, right_y, 'k')
        
        plt.axis('equal')
        plt.grid(False)
        plt.gca().set_position([0, 0, 1, 1])
        plt.axis('off')
        
        plt.savefig(f"{filename}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 센터라인 TXT 저장 (콤마 구분자로 변경)
        with open(f"{filename}_centerline.txt", 'w') as f:
            for x, y in self.centerline:
                f.write(f"{x:.6f},{y:.6f}\n")
        
        # 안쪽 경계선 TXT 저장 (콤마 구분자로 변경)
        with open(f"{filename}_inner.txt", 'w') as f:
            for x, y in self.boundaries['right']:  # 오른쪽이 안쪽 경계선
                f.write(f"{x:.6f},{y:.6f}\n")
        
        # 바깥쪽 경계선 TXT 저장 (콤마 구분자로 변경)
        with open(f"{filename}_outer.txt", 'w') as f:
            for x, y in self.boundaries['left']:  # 왼쪽이 바깥쪽 경계선
                f.write(f"{x:.6f},{y:.6f}\n")

    def close_track(self):
        if len(self.centerline) < 4:
            return False

        # centerline 포인트들을 numpy 배열로 변환
        points = np.array(self.centerline)
        start = points[0]
        end = points[-1]
        
        if np.linalg.norm(end - start) < 1e-6:
            return True
        
        # 마지막 몇 개의 포인트와 처음 몇 개의 포인트를 사용하여 스플라인 생성
        n_points = 5  # 양쪽에서 사용할 포인트 수
        
        # 연결을 위한 포인트 준비
        connect_points = np.vstack([
            points[-n_points:],  # 마지막 n개의 포인트
            points[:n_points]    # 처음 n개의 포인트
        ])
        
        # 매개변수 생성 (포인트 간의 거리 기반)
        t = np.zeros(len(connect_points))
        for i in range(1, len(connect_points)):
            t[i] = t[i-1] + np.linalg.norm(connect_points[i] - connect_points[i-1])
        
        # 스플라인 피팅
        cs = CubicSpline(t, connect_points, bc_type='natural')
        
        # 보간된 포인트 생성
        num_points = 50
        t_new = np.linspace(t[n_points-1], t[n_points], num_points)
        interpolated = cs(t_new)
        
        # 트랙 폭 계산 (현재 세그먼트의 폭 사용)
        width = self.segments[-1].width
        
        # 보간된 포인트들의 방향 벡터 계산
        tangents = np.gradient(interpolated, axis=0)
        norms = np.sqrt(np.sum(tangents**2, axis=1))
        tangents = tangents / norms[:, np.newaxis]
        
        # 진행 방향 확인 (마지막 세그먼트의 방향 유지)
        last_dir = points[-1] - points[-2]
        first_tangent = tangents[0]
        if np.dot(last_dir, first_tangent) < 0:
            tangents = -tangents
        
        # 법선 벡터 계산 (진행 방향 기준 왼쪽이 양수)
        normals = np.array([-tangents[:,1], tangents[:,0]]).T
        
        # 기존 경계선의 방향 확인
        last_left = np.array(self.boundaries['left'][-1])
        last_right = np.array(self.boundaries['right'][-1])
        last_center = np.array(self.centerline[-1])
        
        # 현재 경계선 방향 확인
        current_normal = (last_left - last_center) / np.linalg.norm(last_left - last_center)
        if np.dot(current_normal, normals[0]) < 0:
            normals = -normals
        
        # 경계선 계산
        left_boundary = interpolated + (width/2) * normals
        right_boundary = interpolated - (width/2) * normals
        
        # 기존 포인트 리스트에서 마지막 포인트를 제거하고 보간된 포인트 추가
        self.centerline = self.centerline[:-1]
        self.boundaries['left'] = self.boundaries['left'][:-1]
        self.boundaries['right'] = self.boundaries['right'][:-1]
        
        # 새로운 포인트들 추가
        for i in range(len(interpolated)):
            self.centerline.append(tuple(interpolated[i]))
            self.boundaries['left'].append(tuple(left_boundary[i]))
            self.boundaries['right'].append(tuple(right_boundary[i]))
        
        self.is_closed = True
        return True

class TrackGenerator:
    def __init__(self):
        self.track = Track()
        
        # GUI 설정
        self.fig = plt.figure(figsize=(16, 9))
        self.setup_plot()
        self.setup_controls()
        
        # 현재 세그먼트 파라미터
        self.current_params = {
            'length': 10,
            'width': 2,
            'curvature': 0.1,
            'type': 'straight'
        }

    def setup_plot(self):
        self.ax_main = self.fig.add_axes([0.1, 0.3, 0.8, 0.6])
        self.ax_main.set_title('Track Generator')
        self.ax_main.grid(True)
        self.ax_main.axis('equal')
        self.update_plot()

    def setup_controls(self):
        # 슬라이더 설정
        slider_props = {
            'length': (10, 1, 300),
            'width': (3, 1, 20),
            'curvature': (0.1, -0.5, 0.5),  # 음수 곡률 추가 (-0.5 ~ 0.5)
        }
        
        self.sliders = {}
        y_pos = 0.15
        for param, (val, min_val, max_val) in slider_props.items():
            ax = self.fig.add_axes([0.1, y_pos, 0.65, 0.02])
            self.sliders[param] = Slider(
                ax, param, min_val, max_val,
                valinit=val,
                valstep=0.01 if param == 'curvature' else 1
            )
            self.sliders[param].on_changed(self.update_params)
            y_pos -= 0.03

        # 버튼 추가
        button_width = 0.12  # 버튼 폭을 좀 줄임
        button_height = 0.04
        x_start = 0.1
        button_spacing = 0.02  # 버튼 사이 간격
        
        # Add Straight Segment 버튼
        self.ax_button_straight = self.fig.add_axes([x_start, 0.05, button_width, button_height])
        self.button_straight = Button(self.ax_button_straight, 'Add Straight')
        self.button_straight.on_clicked(self.add_straight_segment)
        
        # Add Left Curve 버튼
        self.ax_button_left = self.fig.add_axes([x_start + (button_width + button_spacing), 0.05, button_width, button_height])
        self.button_left = Button(self.ax_button_left, 'Add Left Curve')
        self.button_left.on_clicked(lambda x: self.add_curve_segment(x, direction=1))
        
        # Add Right Curve 버튼
        self.ax_button_right = self.fig.add_axes([x_start + 2 * (button_width + button_spacing), 0.05, button_width, button_height])
        self.button_right = Button(self.ax_button_right, 'Add Right Curve')
        self.button_right.on_clicked(lambda x: self.add_curve_segment(x, direction=-1))
        
        # Close Track 버튼
        self.ax_button_close = self.fig.add_axes([x_start + 3 * (button_width + button_spacing), 0.05, button_width, button_height])
        self.button_close = Button(self.ax_button_close, 'Close Track')
        self.button_close.on_clicked(self.close_track)
        
        # Save Track 버튼
        self.ax_button_save = self.fig.add_axes([x_start + 4 * (button_width + button_spacing), 0.05, button_width, button_height])
        self.button_save = Button(self.ax_button_save, 'Save Track')
        self.button_save.on_clicked(self.save_track)
        
        # Undo 버튼
        self.ax_button_undo = self.fig.add_axes([x_start + 5 * (button_width + button_spacing), 0.05, button_width, button_height])
        self.button_undo = Button(self.ax_button_undo, 'Undo')
        self.button_undo.on_clicked(self.undo_last_segment)

    def update_params(self, val):
        for param in self.sliders:
            self.current_params[param] = self.sliders[param].val
        self.update_plot()

    def add_straight_segment(self, event):
        segment = TrackSegment(
            length=self.current_params['length'],
            width=self.current_params['width']
        )
        self.track.add_segment(segment)
        self.update_plot()

    def add_curve_segment(self, event, direction=1):
        curvature = abs(self.current_params['curvature']) * direction
        segment = TrackSegment(
            length=self.current_params['length'],
            width=self.current_params['width'],
            segment_type='curve',
            curvature=curvature
        )
        self.track.add_segment(segment)
        self.update_plot()

    def undo_last_segment(self, event):
        if self.track.segments:
            self.track.segments.pop()
            self.track._update_track()
            self.update_plot()

    def update_plot(self):
        self.ax_main.clear()
        if self.track.centerline:
            center_x, center_y = zip(*self.track.centerline)
            self.ax_main.plot(center_x, center_y, 'b-', label='Centerline')
            
            if self.track.boundaries['left'] and self.track.boundaries['right']:
                left_x, left_y = zip(*self.track.boundaries['left'])
                right_x, right_y = zip(*self.track.boundaries['right'])
                self.ax_main.plot(left_x, left_y, 'r-', label='Left boundary')
                self.ax_main.plot(right_x, right_y, 'g-', label='Right boundary')
        
        self.ax_main.axis('equal')
        self.ax_main.legend()
        self.ax_main.grid(True)
        plt.draw()

    def save_track(self, event):
        if not self.track.centerline:
            print("트랙이 비어있습니다!")
            return
            
        if not self.track.is_closed:
            response = input("트랙이 닫히지 않았습니다. 저장하시겠습니까? (y/n): ")
            if response.lower() != 'y':
                return
        
        # 저장 디렉토리 생성
        save_dir = './Traj'
        os.makedirs(save_dir, exist_ok=True)
        
        # 센터라인 TXT 저장 (콤마 구분자로 변경)
        with open(f"{save_dir}/centerline.txt", 'w') as f:
            for x, y in self.track.centerline:
                f.write(f"{x:.6f},{y:.6f}\n")
        
        # 안쪽 경계선 TXT 저장 (콤마 구분자로 변경)
        with open(f"{save_dir}/innerwall.txt", 'w') as f:
            for x, y in self.track.boundaries['right']:  # 오른쪽이 안쪽 경계선
                f.write(f"{x:.6f},{y:.6f}\n")
        
        # 바깥쪽 경계선 TXT 저장 (콤마 구분자로 변경)
        with open(f"{save_dir}/outerwall.txt", 'w') as f:
            for x, y in self.track.boundaries['left']:  # 왼쪽이 바깥쪽 경계선
                f.write(f"{x:.6f},{y:.6f}\n")
        
        # PNG로 저장 (트랙만)
        plt.figure(figsize=(10, 10))
        
        # 센터라인 그리기 (검정 점선)
        center_x, center_y = zip(*self.track.centerline)
        plt.plot(center_x, center_y, '--k')
        
        # 경계선 그리기 (검정 실선)
        left_x, left_y = zip(*self.track.boundaries['left'])
        right_x, right_y = zip(*self.track.boundaries['right'])
        plt.plot(left_x, left_y, 'k')
        plt.plot(right_x, right_y, 'k')
        
        plt.axis('equal')
        plt.grid(False)
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        
        plt.savefig(f"{save_dir}/map_gen.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"트랙이 저장되었습니다: {save_dir}/map_gen")

    def close_track(self, event):
        if self.track.segments:
            if self.track.close_track():
                self.update_plot()
                print("트랙이 성공적으로 닫혔습니다.")
            else:
                print("트랙을 닫기 위한 충분한 세그먼트가 없습니다.")

def main():
    generator = TrackGenerator()
    plt.show()

if __name__ == "__main__":
    main()
