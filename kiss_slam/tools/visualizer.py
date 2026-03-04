import importlib
import os
from abc import ABC
from functools import partial
from typing import Callable, List
import sys
from collections import deque
import numpy as np

import atexit
from datetime import datetime

try:
    import cv2  # ưu tiên dùng OpenCV nếu có
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
try:
    import imageio as iio  # v2 API
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

SHOW_ODOM = False
FLIP_Z = True

def _flip_z_points(pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    out = pts.copy()
    out[:, 2] *= -1.0
    return out

def _flip_z_pose(T: np.ndarray) -> np.ndarray:
    out = T.copy()
    out[2, 3] *= -1.0
    return out

# -------- Colors / sizes --------
YELLOW = np.array([1, 0.706, 0])
GREY   = np.array([0.5, 0.5, 0.5])
RED    = np.array([128, 0, 0]) / 255.0
BLACK  = np.array([0, 0, 0]) / 255.0
BLUE   = np.array([0.4, 0.5, 0.9])
GREEN  = np.array([0.4, 0.9, 0.5])
DARK_BLUE = np.array([0.0, 0.0, 0.75])
ORANGE    = np.array([1.0, 0.55, 0.0])

SPHERE_SIZE_KEYPOSES = 0.5
SPHERE_SIZE_ODOMETRY = 0.2
TRAJ_Z_OFFSET = -11.0
THEMES = {
    "TH1": {
        "trajectory": np.array([0.4, 0.5, 0.9]),
        "keyframe":  np.array([1.0, 0.8, 0.0]),
        "odom":      np.array([0.4, 0.5, 0.9]),
        "localmap":  np.array([0.7, 0.7, 0.7]), # Xám cho Global Map
        "closure":   np.array([1.0, 0.0, 0.0])
    },
    "TH2": {
        "trajectory": np.array([1.0, 0.6, 0.0]),
        "keyframe":   np.array([1.0, 0.8, 0.0]),
        "odom":       np.array([1.0, 0.6, 0.0]),
        "localmap":   np.array([0.7, 0.7, 0.7]),
        "closure":    np.array([1.0, 0.0, 0.0])
    }
}

def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t

class StubVisualizer(ABC):
    def __init__(self): pass
    def update(self, slam): pass

class RegistrationVisualizer(StubVisualizer):
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            print('open3d is not installed on your system, run "pip install open3d"')
            exit(1)
        self.theme = THEMES["TH1"]

        self.block_vis = True
        self.play_crun = True
        self.reset_bounding_box = True
        # self.follow_cam = True
        self.follow_cam = False

        # --- DUAL MAP SYSTEM (LOCAL & GLOBAL) ---
        self.local_map = self.o3d.geometry.PointCloud()
        self.global_map = self.o3d.geometry.PointCloud() # Thêm Global Map
        self.global_points = np.empty((0, 3))            # Mảng lưu trữ vĩnh viễn
        self.last_kf_idx = 0                             # Bộ đếm Keyframe
        
        self.closures = []
        self.key_poses = []
        self.key_frames = []
        self.global_frames = []
        self.odom_frames = []
        self.edges = []
        self.current_node = None
        self.current_marker = None
        
        self.z_plane = None
        self.tube_radius =  0.205

        self.current_marker_pose = None
        self.odom_pts = deque(maxlen=2)
        
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()
        
        self.flip_view_z = False
        self._S_flipZ = np.eye(4); self._S_flipZ[2, 2] = -1.0

        self.vis.get_render_option().mesh_show_back_face = True
        self.odom_tubes = []
        
        # ------- Video recording -------
        self.record = os.getenv("KISSLAM_RECORD", "0") == "1"
        default_name = f"slam_vis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.video_path = os.getenv("KISSLAM_VIDEO", default_name)
        self.fps = int(os.getenv("KISSLAM_FPS", "20"))

        self._video_writer = None   
        self._vw_backend = None     
        self._frame_size = None     
        
        self.odom_mega  = self.o3d.geometry.TriangleMesh()  
        self.merge_every = 50              
        self._tube_count_since_merge = 0
        self.z_offset_delta = 9.6  
        self.z_step = 0.20         
        
        self.frame_idx = 0  
        self.max_faces   = 200_000
        self.hard_reset_every = 10000    
        self._merge_round = 0
        self._segment_total = 0
        atexit.register(self._finalize_video)  

    def _compact_history(self):
        tris = len(self.odom_mega.triangles)
        if tris == 0: return
        target = min(self.max_faces, max(50_000, tris // 2))
        try:
            new_mesh = self.odom_mega.simplify_quadric_decimation(target_number_of_triangles=target)
            new_mesh.compute_vertex_normals()
            if hasattr(self, "_odom_mega_added") and self._odom_mega_added:
                self.vis.remove_geometry(self.odom_mega, reset_bounding_box=False)
            self.odom_mega = new_mesh
            self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
            self._odom_mega_added = True
            try:
                self.odom_mega.remove_duplicated_vertices()
                self.odom_mega.remove_duplicated_triangles()
                self.odom_mega.remove_degenerate_triangles()
                self.odom_mega.remove_unreferenced_vertices()
                self.odom_mega.compute_vertex_normals()
            except Exception:
                pass
        except Exception as e:
            pass

    def _merge_tubes_if_needed(self):
        if self._tube_count_since_merge >= self.merge_every and len(self.odom_tubes) > 0:
            tmp = self.o3d.geometry.TriangleMesh()
            for t in self.odom_tubes:
                tmp += t
                self.vis.remove_geometry(t, reset_bounding_box=False)
            self.odom_tubes.clear()
            self._tube_count_since_merge = 0

            try:
                tmp.remove_duplicated_vertices()
                tmp.remove_duplicated_triangles()
                tmp.remove_degenerate_triangles()
                tmp.remove_unreferenced_vertices()
            except Exception:
                pass

            self.odom_mega += tmp
            tris = len(self.odom_mega.triangles)
            self._merge_round += 1
            if tris > self.max_faces or (self._merge_round % 5 == 0):
                target = min(self.max_faces, max(50_000, tris // 2))
                try:
                    new_mesh = self.odom_mega.simplify_quadric_decimation(target_number_of_triangles=target)
                    new_mesh.compute_vertex_normals()
                    if hasattr(self, "_odom_mega_added") and self._odom_mega_added:
                        self.vis.remove_geometry(self.odom_mega, reset_bounding_box=False)
                    self.odom_mega = new_mesh
                    if not hasattr(self, "_odom_mega_added"):
                        self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
                        self._odom_mega_added = True
                    else:
                        self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
                except Exception:
                    pass

            if not hasattr(self, "_odom_mega_added"):
                self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
                self._odom_mega_added = True
            else:
                self.vis.update_geometry(self.odom_mega)

    def _update_odom_tubes(self, current_pose, z_anchor, radius=0.5):
        pt = current_pose[:3, 3].copy(); pt[2] = z_anchor

        if len(self.odom_pts) == 0 or np.linalg.norm(pt - self.odom_pts[-1]) > 1e-9:
            self.odom_pts.append(pt)
            if len(self.odom_pts) == 2:
                p0, p1 = self.odom_pts[0], self.odom_pts[1]
                tube = self._cylinder_between(p0, p1, radius=radius, color=self.theme["trajectory"])
                if tube is not None:
                    self.odom_tubes.append(tube)
                    self.vis.add_geometry(tube, reset_bounding_box=False)
                    self._tube_count_since_merge += 1
                    self._merge_tubes_if_needed()
                    self._segment_total += 1
                    if (self._segment_total % self.hard_reset_every) == 0:
                        self._compact_history()

    def update(self, slam):
        self._update_geometries(slam)
        self.vis.poll_events()
        self.vis.update_renderer()
        if self.record:
            self._capture_and_record()
        self.frame_idx += 1

    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        
        # Bật cả Local và Global Map lên màn hình
        self.vis.add_geometry(self.local_map)
        self.vis.add_geometry(self.global_map)
        
        self._set_black_background(self.vis)
        ro = self.vis.get_render_option()
        ro.point_size = 1
        ro.line_width = 8

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["N"], self._next_frame)
        self._register_key_callback(["C"], self._center_viewpoint)
        self._register_key_callback(["B"], self._set_black_background)
        self._register_key_callback(["W"], self._set_white_background)
        self._register_key_callback(["F"], self._toggle_follow)
        self._register_key_callback(["R"], self._toggle_record) 
        self._register_key_callback(["X"], self._clear_path)      
        self._register_key_callback(["J"], self._thicker_tube)  
        self._register_key_callback(["K"], self._thinner_tube)  
        self._register_key_callback(["["], self._lower_traj)   
        self._register_key_callback(["]"], self._raise_traj)   
        self._register_key_callback(["0"], self._reset_traj)   
        self._register_key_callback(["Z"], self._toggle_view_flip_z)

    def _toggle_view_flip_z(self, vis):
        self.flip_view_z = not self.flip_view_z
        self._apply_view_flip_z() 

    def _apply_view_flip_z(self):
        ctr = self.vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = params.extrinsic @ self._S_flipZ
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    def _clear_path(self, vis):
        for t in self.odom_tubes:
            self.vis.remove_geometry(t, reset_bounding_box=False)
        self.odom_tubes.clear()
        self.odom_pts.clear()
        self._tube_count_since_merge = 0
        if hasattr(self, "_odom_mega_added") and self._odom_mega_added:
            self.vis.remove_geometry(self.odom_mega, reset_bounding_box=False)
            self._odom_mega_added = False
        self.odom_mega = self.o3d.geometry.TriangleMesh()
        
        # Xóa luôn Global Map khi ấn X
        self.global_points = np.empty((0, 3))
        self.global_map.points = self.o3d.utility.Vector3dVector(self.global_points)
        self.vis.update_geometry(self.global_map)

    def _thicker_tube(self, vis):
        if not hasattr(self, "tube_radius"): self.tube_radius = 0.205
        self.tube_radius = min(self.tube_radius * 1.25, 1.0)

    def _thinner_tube(self, vis):
        self.tube_radius /= 1.25
        self.tube_radius = max(self.tube_radius, 0.01)

    def _nudge_traj_height(self, delta):
        if abs(delta) < 1e-9: return
        for i in range(len(self.odom_pts)): self.odom_pts[i][2] += delta
        for t in self.odom_tubes:
            t.translate((0, 0, delta), relative=True)
            self.vis.update_geometry(t)
        if hasattr(self, "_odom_mega_added") and self._odom_mega_added:
            self.odom_mega.translate((0, 0, delta), relative=True)
            self.vis.update_geometry(self.odom_mega)
        if self.z_plane is not None: self.z_plane += delta
        self.z_offset_delta += delta

    def _raise_traj(self, vis): self._nudge_traj_height(+self.z_step)
    def _lower_traj(self, vis): self._nudge_traj_height(-self.z_step)
    def _reset_traj(self, vis):
        self._nudge_traj_height(-self.z_offset_delta)
        self.z_plane = None

    def _set_black_background(self, vis): vis.get_render_option().background_color = [0.0, 0.0, 0.0]
    def _set_white_background(self, vis): vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        if self.record:
            self.vis.update_renderer()
            last_frame = self._capture_rgb_frame()
            if last_frame is not None:
                if self._video_writer is None: self._ensure_writer(last_frame)
                if self._video_writer is not None:
                    self._write_frame(last_frame)
                    self._finalize_video()
        vis.destroy_window()
        sys.exit(0)   

    def _next_frame(self, vis): self.block_vis = not self.block_vis
    def _start_stop(self, vis): self.play_crun = not self.play_crun
    def _center_viewpoint(self, vis): vis.reset_view_point(True)
    def _toggle_follow(self, vis): self.follow_cam = not self.follow_cam
    def _toggle_record(self, vis):
        self.record = not self.record
        if not self.record: self._finalize_video() 

    def _capture_rgb_frame(self):
        img = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        if img is None or img.size == 0: return None
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        if img_u8.ndim == 2: img_u8 = np.repeat(img_u8[..., None], 3, axis=2)
        return img_u8 

    def _ensure_writer(self, frame_rgb):
        if self._video_writer is not None: return
        h, w, _ = frame_rgb.shape
        self._frame_size = (w, h)
        if _HAS_CV2:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(self.video_path, fourcc, self.fps, self._frame_size)
            if vw is not None and vw.isOpened():
                self._video_writer = vw
                self._vw_backend = "cv2"
                return
        if _HAS_IMAGEIO:
            self._video_writer = iio.get_writer(self.video_path, fps=self.fps, codec="libx264", macro_block_size=None)
            self._vw_backend = "imageio"
        else:
            self.record = False

    def _write_frame(self, frame_rgb):
        if self._video_writer is None: return
        if self._vw_backend == "cv2": self._video_writer.write(frame_rgb[..., ::-1])
        elif self._vw_backend == "imageio": self._video_writer.append_data(frame_rgb)

    def _capture_and_record(self):
        if not self.record: return
        frame = self._capture_rgb_frame()
        if frame is None: return
        if self._video_writer is None:
            self._ensure_writer(frame)
            if self._video_writer is None: return
                
        self._write_frame(frame)
        
        try:
            os.makedirs("renders", exist_ok=True)
            img_path = f"renders/frame_{self.frame_idx:05d}.png"
            if _HAS_IMAGEIO: iio.imwrite(img_path, frame)
            elif _HAS_CV2: cv2.imwrite(img_path, frame[..., ::-1])
                
            if self.frame_idx > 5:
                old_path = f"renders/frame_{self.frame_idx-5:05d}.png"
                if os.path.exists(old_path): os.remove(old_path)
        except Exception: pass

    def _finalize_video(self):
        if self._video_writer is None: return
        try:
            if self._vw_backend == "cv2": self._video_writer.release()
            elif self._vw_backend == "imageio": self._video_writer.close()
        except Exception: pass
        finally:
            self._video_writer = None
            self._vw_backend = None
            self._frame_size = None

    def _add_line(self, pose0, pose1, color, z_offset=0.0):
        p0 = pose0.copy(); p1 = pose1.copy()
        p0[2] += z_offset; p1[2] += z_offset
        lines = [[0, 1]]
        colors = [color for _ in lines]
        line_set = self.o3d.geometry.LineSet()
        line_set.points = self.o3d.utility.Vector3dVector([p0, p1])
        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        line_set.colors = self.o3d.utility.Vector3dVector(colors)
        return line_set

    def _cylinder_between(self, p0, p1, radius=0.1, color=None, resolution=12):
        axis = p1 - p0
        L = np.linalg.norm(axis)
        if L < 1e-9: return None
        z = axis / L
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z, up)) > 0.99: up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-9)
        y = np.cross(z, x)
        R = np.eye(4)
        R[:3, 0] = x; R[:3, 1] = y; R[:3, 2] = z; R[:3, 3] = p0

        cyl = self.o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=resolution, split=1)
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color(self.theme["trajectory"] if color is None else color)
        
        T = np.eye(4)
        T[:3, 3] = np.array([0, 0, L/2.0])
        cyl.transform(T)
        cyl.transform(R)
        return cyl

    def _add_frames(self, poses, size, color):
        frames = []
        for pose in poses: frames.append(self._add_frame(pose, size, color))
        return frames

    def _add_frame(self, pose, size, color):
        m = self.o3d.geometry.TriangleMesh.create_sphere(size)
        m.paint_uniform_color(color)
        m.compute_vertex_normals()
        m.transform(pose)
        return m

    def _follow_pose(self, pose, distance=10.0, height=5.0, zoom=0.35):
        ctr = self.vis.get_view_control()
        R = pose[:3, :3]
        t = pose[:3, 3]
        fwd = R @ np.array([0.05, 0.0, 0.0])
        up  = R @ np.array([0.0, 0.0, 1.0])
        back = -fwd

        cam_pos = t + back * distance + up * height
        lookat = t

        if self.flip_view_z:
            cam_pos = cam_pos.copy(); lookat = lookat.copy(); up = up.copy()
            cam_pos[2] *= -1.0; lookat[2]  *= -1.0; up[2] *= -1.0

        front = (lookat - cam_pos)
        n = np.linalg.norm(front)
        if n < 1e-9: return
        front /= n
        up_n = up / (np.linalg.norm(up) + 1e-9)

        ctr.set_lookat(lookat.tolist())
        ctr.set_front(front.tolist())
        ctr.set_up(up_n.tolist())
        ctr.set_zoom(zoom)
        
    def _update_geometries(self, slam):
        current_node = slam.local_map_graph.last_local_map
        local_map_in_global = transform_points(
            slam.voxel_grid.point_cloud(), current_node.keypose
        )

        if FLIP_Z:
            local_map_in_global = _flip_z_points(local_map_in_global)

        # 1. LOCAL MAP: Sơn MÀU ĐỎ và chỉ giữ mây điểm hiện tại
        self.local_map.points = self.o3d.utility.Vector3dVector(local_map_in_global)
        self.local_map.paint_uniform_color(np.array([1.0, 0.2, 0.2])) # Red
        self.vis.update_geometry(self.local_map)

        # 2. GLOBAL MAP: Tích lũy các điểm và nén định kỳ
        key_poses = slam.get_keyposes()
        if len(key_poses) > self.last_kf_idx:
            self.last_kf_idx = len(key_poses)
            
            if self.global_points.shape[0] == 0:
                self.global_points = local_map_in_global
            else:
                self.global_points = np.vstack((self.global_points, local_map_in_global))
            
            # CHỐNG SẬP RAM: Nén mảng global mỗi khi có 5 keyframes mới
            if self.last_kf_idx % 5 == 0:
                temp_pcd = self.o3d.geometry.PointCloud()
                temp_pcd.points = self.o3d.utility.Vector3dVector(self.global_points)
                # Kích thước voxel 0.5m: Giữ bản đồ rõ ràng nhưng siêu nhẹ
                temp_pcd = temp_pcd.voxel_down_sample(voxel_size=0.5)
                self.global_points = np.asarray(temp_pcd.points)

            self.global_map.points = self.o3d.utility.Vector3dVector(self.global_points)
            self.global_map.paint_uniform_color(self.theme["localmap"]) # Màu Xám
            self.vis.update_geometry(self.global_map)

        # --- TÍNH TOÁN Z-PLANE (Dành cho Quỹ Đạo) ---
        if local_map_in_global.shape[0] > 0:
            plane_model, inliers = self.local_map.segment_plane(
                distance_threshold=0.05, ransac_n=3, num_iterations=500
            )
            a, b, c, d = plane_model
            t_disp = _flip_z_pose(current_node.endpose)[:3, 3] if FLIP_Z else current_node.endpose[:3, 3]
            if abs(c) > 1e-6:
                z_ground = (-a * t_disp[0] - b * t_disp[1] - d) / c
                target = z_ground + TRAJ_Z_OFFSET + self.z_offset_delta
            else:
                target = float(t_disp[2]) + TRAJ_Z_OFFSET + self.z_offset_delta
        else:
            t_disp = _flip_z_pose(current_node.endpose)[:3, 3] if FLIP_Z else current_node.endpose[:3, 3]
            target = float(t_disp[2]) + TRAJ_Z_OFFSET + self.z_offset_delta

        alpha = 0.15
        self.z_plane = target if self.z_plane is None else (1 - alpha) * self.z_plane + alpha * target
        z_anchor = self.z_plane

        current_pose = _flip_z_pose(current_node.endpose) if FLIP_Z else current_node.endpose
        self._update_odom_tubes(current_pose, z_anchor, radius=self.tube_radius)

        if self.follow_cam:
            self._follow_pose(current_pose)

        # --- Vẽ Keyframes & Loop Closures ---
        if key_poses != self.key_poses:
            for frame in self.key_frames: self.vis.remove_geometry(frame, reset_bounding_box=False)
            self.key_frames = []

            kf_transformed = [(_flip_z_pose(k) if FLIP_Z else k) for k in key_poses]
            self.key_frames = self._add_frames(kf_transformed, SPHERE_SIZE_KEYPOSES, self.theme["keyframe"])
            for frame in self.key_frames: self.vis.add_geometry(frame, reset_bounding_box=False)
            self.key_poses = key_poses

            for edge in self.edges: self.vis.remove_geometry(edge, reset_bounding_box=False)
            self.edges = []

            for (idx0, idx1) in self.closures:
                p0 = self.key_frames[idx0].get_center()
                p1 = self.key_frames[idx1].get_center()
                self.edges.append(self._add_line(p0, p1, self.theme["closure"]))

            for e in self.edges: self.vis.add_geometry(e, reset_bounding_box=False)

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False
