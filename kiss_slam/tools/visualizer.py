# export KISSLAM_RECORD=1
# export KISSLAM_VIDEO="/mnt/sdc/output_kslam.mp4"
# export KISSLAM_FPS=20
# pip install "imageio[ffmpeg]"
# pip install "imageio[pyav]"
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
    # chỉ lật phần tịnh tiến z; giữ nguyên quay
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
DARK_BLUE = np.array([0.0, 0.0, 0.75])   # <-- xanh dương đậm cho trajectory
ORANGE    = np.array([1.0, 0.55, 0.0])   # Trajectory: cam

SPHERE_SIZE_KEYPOSES = 0.5
SPHERE_SIZE_ODOMETRY = 0.2
TRAJ_Z_OFFSET = -11.0   # <-- nâng đường trajectory lên 0.4m (tùy cảnh có thể 0.2–1.0)
THEMES = {
    "TH1": {
        # "trajectory": np.array([0.0, 0.0, 0.75]),   # Dark Blue
        "trajectory": np.array([0.4, 0.5, 0.9]),   #  Light Blue 
        "keyframe":  np.array([1.0, 0.8, 0.0]),    #  Yellow 
        # "odom":      np.array([0.4, 0.9, 0.5]),    # Green
        "odom":      np.array([0.4, 0.5, 0.9]),    # Dark Blue  
        "localmap":  np.array([0.7, 0.7, 0.7]),    # Grey
        "closure":   np.array([1.0, 0.0, 0.0])     # Red
    },
    "TH2": {
        # "trajectory": np.array([1.0, 0.55, 0.0]),  # Orange
        "trajectory": np.array([1.0, 0.6, 0.0]),   # Light Orange 
        "keyframe":   np.array([1.0, 0.8, 0.0]),   # Yellow
        # "odom":       np.array([0.4, 0.9, 0.5]),   # Green
        "odom":       np.array([1.0, 0.6, 0.0]),   # Orange   
        "localmap":   np.array([0.7, 0.7, 0.7]),   # Grey
        "closure":    np.array([1.0, 0.0, 0.0])    # Red
    }
}


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


class StubVisualizer(ABC):
    def __init__(self):
        pass


    def update(self, slam):
        pass


class RegistrationVisualizer(StubVisualizer):
    """
    Open3D visualizer with a 'follow camera' that chases the current LiDAR pose.
    Hotkeys:
      SPACE : play/pause
      N     : step one frame
      C     : center viewpoint (one-time)
      B/W   : black/white background
      F     : toggle follow camera
      ESC/Q : quit
    """
    # Public Interface ----------------------------------------------------------------------------
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            print('open3d is not installed on your system, run "pip install open3d"')
            exit(1)
        self.theme = THEMES["TH1"]   # mặc định TH1

        # GUI / control flags
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True
        self.follow_cam = True  # <-- follow camera enabled by default

        # Scene data
        self.local_map = self.o3d.geometry.PointCloud()
        self.closures = []
        self.key_poses = []
        self.key_frames = []
        self.global_frames = []
        self.odom_frames = []
        self.edges = []
        self.current_node = None
        self.current_marker = None   # marker vàng tại pose hiện tại
        
        self.z_plane = None  # mặt phẳng z ổn định cho ống
        self.tube_radius =  0.205 # hoặc 0.04–0.10 tùy scale

        # optim
        self.current_marker_pose = None
        self.odom_pts = deque(maxlen=2)
        # Open3D visualizer
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()
        # --- View flip Z (mirror qua Oxy) ---
        self.flip_view_z = False
        self._S_flipZ = np.eye(4); self._S_flipZ[2, 2] = -1.0

        # Khi phản xạ, hướng mặt tam giác có thể bị đảo -> bật vẽ mặt sau để tránh “mất hình”
        self.vis.get_render_option().mesh_show_back_face = True

        # --- ODOM polyline persistent ---
        # self.odom_traj_ls = self.o3d.geometry.LineSet()
        self.odom_tubes = []  # list TriangleMesh các ống lẻ
        # self.vis.add_geometry(self.odom_traj_ls, reset_bounding_box=False)
                # ------- Video recording -------
        # Bật/tắt auto-record bằng env (mặc định tắt): KISSLAM_RECORD=1
        self.record = os.getenv("KISSLAM_RECORD", "0") == "1"
        # Đường dẫn xuất video (mặc định: slam_vis_<timestamp>.mp4)
        default_name = f"slam_vis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.video_path = os.getenv("KISSLAM_VIDEO", default_name)
        # FPS đầu ra (mặc định 20)
        self.fps = int(os.getenv("KISSLAM_FPS", "20"))

        self._video_writer = None   # cv2.VideoWriter hoặc imageio writer
        self._vw_backend = None     # "cv2" hoặc "imageio"
        self._frame_size = None     # (w, h)
        # __init__
        self.odom_mega  = self.o3d.geometry.TriangleMesh()  # mesh gộp
        self.merge_every = 50              # gộp mỗi 50 đoạn
        self._tube_count_since_merge = 0
        self.z_offset_delta = 9.6  # tổng độ lệch z đang cộng thêm vào đường
        self.z_step = 0.20         # mỗi lần bấm phím sẽ nhấc/hạ bấy nhiêu (m)
        # --- Per-frame recording state ---
        self.frame_idx = 0  # đếm số frame SLAM đã render
        self.merge_every = 10
        self.max_faces   = 200_000
        self.hard_reset_every = 10000    # mỗi 20k đoạn thì “nén” 1 lần
        self._merge_round = 0
        self._segment_total = 0
        atexit.register(self._finalize_video)  # đóng file an toàn khi thoát

    # def _merge_tubes_if_needed(self):
    #     if self._tube_count_since_merge >= self.merge_every and len(self.odom_tubes) > 0:
    #         # Gộp tất cả ống lẻ vào odom_mega
    #         for t in self.odom_tubes:
    #             self.odom_mega += t
    #             self.vis.remove_geometry(t, reset_bounding_box=False)
    #         self.odom_tubes.clear()
    #         self._tube_count_since_merge = 0
    #         # Thêm/ cập nhật odom_mega vào scene
    #         if not hasattr(self, "_odom_mega_added"):
    #             self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
    #             self._odom_mega_added = True
    #         else:
    #             self.vis.update_geometry(self.odom_mega)
    def _compact_history(self):
        """Nén (decimate) odom_mega để số mặt không phình vô hạn, rồi SWAP mesh mới vào scene."""
        tris = len(self.odom_mega.triangles)
        if tris == 0:
            return
        target = min(self.max_faces, max(50_000, tris // 2))
        try:
            new_mesh = self.odom_mega.simplify_quadric_decimation(
                target_number_of_triangles=target
            )
            new_mesh.compute_vertex_normals()
            # SWAP: gỡ mesh cũ khỏi scene để giải phóng native memory, thêm mesh mới
            if hasattr(self, "_odom_mega_added") and self._odom_mega_added:
                self.vis.remove_geometry(self.odom_mega, reset_bounding_box=False)
            self.odom_mega = new_mesh
            self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
            self._odom_mega_added = True
            # (tuỳ chọn) dọn lưới lần nữa
            try:
                self.odom_mega.remove_duplicated_vertices()
                self.odom_mega.remove_duplicated_triangles()
                self.odom_mega.remove_degenerate_triangles()
                self.odom_mega.remove_unreferenced_vertices()
                self.odom_mega.compute_vertex_normals()
            except Exception:
                pass
        except Exception as e:
            print(f"[Compact] decimate failed: {e}")

    def _merge_tubes_if_needed(self):
        if self._tube_count_since_merge >= self.merge_every and len(self.odom_tubes) > 0:
            # 1) gộp các ống lẻ vào 1 mesh tạm
            tmp = self.o3d.geometry.TriangleMesh()
            for t in self.odom_tubes:
                tmp += t
                self.vis.remove_geometry(t, reset_bounding_box=False)
            self.odom_tubes.clear()
            self._tube_count_since_merge = 0

            # 2) dọn lưới tạm
            try:
                tmp.remove_duplicated_vertices()
                tmp.remove_duplicated_triangles()
                tmp.remove_degenerate_triangles()
                tmp.remove_unreferenced_vertices()
            except Exception:
                pass

            # 3) cộng vào odom_mega
            self.odom_mega += tmp

            # 4) nếu quá nhiều mặt -> decimate + SWAP object để giải phóng bộ nhớ cũ
            tris = len(self.odom_mega.triangles)
            self._merge_round += 1
            if tris > self.max_faces or (self._merge_round % 5 == 0):
                target = min(self.max_faces, max(50_000, tris // 2))
                try:
                    new_mesh = self.odom_mega.simplify_quadric_decimation(
                        target_number_of_triangles=target
                    )
                    new_mesh.compute_vertex_normals()
                    # SWAP: tháo mesh cũ khỏi scene, thay bằng mesh mới để GC thu hồi bộ nhớ cũ
                    if hasattr(self, "_odom_mega_added") and self._odom_mega_added:
                        self.vis.remove_geometry(self.odom_mega, reset_bounding_box=False)
                    self.odom_mega = new_mesh
                    if not hasattr(self, "_odom_mega_added"):
                        self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
                        self._odom_mega_added = True
                    else:
                        self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
                except Exception:
                    # nếu decimate không khả dụng, vẫn cập nhật geometry hiện có
                    pass

            # 5) cập nhật scene nếu chưa add
            if not hasattr(self, "_odom_mega_added"):
                self.vis.add_geometry(self.odom_mega, reset_bounding_box=False)
                self._odom_mega_added = True
            else:
                self.vis.update_geometry(self.odom_mega)

    # def _update_odom_tubes(self, current_pose, z_anchor, radius=0.5):
    #     pt = current_pose[:3, 3].copy()
    #     pt[2] = z_anchor
    #     if len(self.odom_pts) == 0 or np.linalg.norm(pt - self.odom_pts[-1]) > 1e-9:
    #         self.odom_pts.append(pt)
    #         if len(self.odom_pts) >= 2:
    #             p0, p1 = self.odom_pts[-2], self.odom_pts[-1]
    #             tube = self._cylinder_between(p0, p1, radius=radius, color=self.theme["trajectory"])
    #             if tube is not None:
    #                 self.odom_tubes.append(tube)
    #                 self.vis.add_geometry(tube, reset_bounding_box=False)
    #                 self._tube_count_since_merge += 1
    #                 self._merge_tubes_if_needed()
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
                # --- đếm tổng số đoạn và nén định kỳ để RES không leo ---
                    self._segment_total += 1
                    if (self._segment_total % self.hard_reset_every) == 0:
                        self._compact_history()
            # luôn giữ maxlen=2; khi thêm điểm thứ 3, điểm đầu tự bị bỏ

    # def update(self, slam):
    #     self._update_geometries(slam)
    #     while self.block_vis:
    #         self.vis.poll_events()
    #         self.vis.update_renderer()
    #         # Ghi frame sau mỗi lần render nếu đang record
    #         self._capture_and_record()
    #         if self.play_crun:
    #             break
    #     self._capture_and_record()
    #     self.block_vis = not self.block_vis
    def update(self, slam):
        """
        Per-frame recording: mỗi lần pipeline gọi update() là:
        - Cập nhật geometry theo frame SLAM hiện tại
        - Render đúng 1 lần
        - (Nếu record đang bật) chụp màn hình và ghi vào video đúng 1 frame
        """
        # 1) cập nhật scene theo dữ liệu SLAM mới
        self._update_geometries(slam)

        # 2) chạy một vòng render duy nhất cho frame này
        self.vis.poll_events()
        self.vis.update_renderer()

        # 3) ghi đúng 1 frame vào video (nếu bật record)
        if self.record:
            self._capture_and_record()

        # 4) tăng bộ đếm frame (hữu ích cho debug/log)
        self.frame_idx += 1

    # Private Interface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        self.vis.add_geometry(self.local_map)
        self._set_black_background(self.vis)
        ro = self.vis.get_render_option()
        ro.point_size = 1
        ro.line_width = 8
        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t  [ESC] to exit\n"
            "\t    [N] to step\n"
            "\t    [C] to center the viewpoint\n"
            "\t    [W] to toggle a white background\n"
            "\t    [B] to toggle a black background\n"
            "\t    [F] to toggle follow camera\n"
        )
    def _update_odom_polyline(self, current_pose, z_anchor):
        """Append current pose to a persistent LineSet so the odom path is continuous and on top."""

        # Lấy vị trí hiện tại
        pt = current_pose[:3, 3].copy()
        pt[2] = z_anchor  # neo Z lên trên đỉnh point cloud để không bị che

        # Nếu là điểm đầu hoặc khác điểm cũ -> thêm
        if (len(self.odom_pts) == 0) or (np.linalg.norm(pt - self.odom_pts[-1]) > 1e-9):
            self.odom_pts.append(pt)

            # Cập nhật LineSet: các đoạn (i -> i+1)
            if len(self.odom_pts) >= 2:
                lines = [[i, i + 1] for i in range(len(self.odom_pts) - 1)]
            else:
                lines = []

            self.odom_traj_ls.points = self.o3d.utility.Vector3dVector(self.odom_pts)
            self.odom_traj_ls.lines  = self.o3d.utility.Vector2iVector(lines)
            self.odom_traj_ls.colors = self.o3d.utility.Vector3dVector(
                [self.theme["trajectory"] for _ in lines]
            )

            self.vis.update_geometry(self.odom_traj_ls)

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
        # self._register_key_callback(["F"], self._toggle_follow)  # <-- new
        self._register_key_callback(["F"], self._toggle_follow)
        self._register_key_callback(["R"], self._toggle_record)  # <-- NEW: bật/tắt ghi hình

        # self._register_key_callback(["R"], self._toggle_record)
        self._register_key_callback(["X"], self._clear_path)      # reset đường
        self._register_key_callback(["J"], self._thicker_tube)  # dày hơn
        self._register_key_callback(["K"], self._thinner_tube)  # mỏng hơn
        # … các hotkey khác …
        self._register_key_callback(["["], self._lower_traj)   # hạ đường
        self._register_key_callback(["]"], self._raise_traj)   # nâng đường
        self._register_key_callback(["0"], self._reset_traj)   # reset về mặc định
        self._register_key_callback(["Z"], self._toggle_view_flip_z)
    def _toggle_view_flip_z(self, vis):
        """Bật/tắt lật khung nhìn theo Z bằng cách nhân extrinsic với S."""
        self.flip_view_z = not self.flip_view_z
        self._apply_view_flip_z()  # nhân S vào extrinsic hiện tại
        print(f"[ViewFlipZ] {'ON' if self.flip_view_z else 'OFF'}")

    def _apply_view_flip_z(self):
        """Nhân extrinsic camera với S (S^2 = I nên toggle 2 lần sẽ về trạng thái cũ)."""
        ctr = self.vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = params.extrinsic @ self._S_flipZ
        # allow_arbitrary=True để chấp nhận extrinsic bất kỳ
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
        print("[Odom] Cleared path.")
    def _thicker_tube(self, vis):
        if not hasattr(self, "tube_radius"):
            self.tube_radius = 0.205
        self.tube_radius = min(self.tube_radius * 1.25, 1.0)
        print(f"[Odom] tube radius = {self.tube_radius:.3f}")
    def _thinner_tube(self, vis):
        self.tube_radius /= 1.25
        self.tube_radius = max(self.tube_radius, 0.01)
        print(f"[Odom] tube radius = {self.tube_radius:.3f}")
    def _nudge_traj_height(self, delta):
        """Di chuyển toàn bộ đường ống theo trục Z ngay lập tức, và cập nhật trạng thái."""
        if abs(delta) < 1e-9:
            return
        # 1) dời toàn bộ điểm đã lưu
        for i in range(len(self.odom_pts)):
            self.odom_pts[i][2] += delta
        # 2) dời toàn bộ mesh ống lẻ
        for t in self.odom_tubes:
            t.translate((0, 0, delta), relative=True)
            self.vis.update_geometry(t)
        # 3) dời mesh gộp (nếu có)
        if hasattr(self, "_odom_mega_added") and self._odom_mega_added:
            self.odom_mega.translate((0, 0, delta), relative=True)
            self.vis.update_geometry(self.odom_mega)
        # 4) cập nhật mặt phẳng z hiện tại + tổng offset
        if self.z_plane is not None:
            self.z_plane += delta
        self.z_offset_delta += delta
        print(f"[Odom] traj height {'UP' if delta>0 else 'DOWN'} {abs(delta):.2f} m  |  total offset = {self.z_offset_delta:.2f} m")

    def _raise_traj(self, vis):
        self._nudge_traj_height(+self.z_step)

    def _lower_traj(self, vis):
        self._nudge_traj_height(-self.z_step)

    def _reset_traj(self, vis):
        # đưa về baseline: nhấc/hạ lại đúng -z_offset_delta
        self._nudge_traj_height(-self.z_offset_delta)
        # làm mềm lại z-plane ở frame tiếp theo
        self.z_plane = None
        print("[Odom] traj height reset to baseline")

    # ----- Background / window -----
    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]


    def _quit(self, vis):
    # Nếu đang record mà chưa khởi tạo writer, chụp một frame cuối để khởi tạo
        if self.record:
            self.vis.update_renderer()
            last_frame = self._capture_rgb_frame()
            if last_frame is not None:
                if self._video_writer is None:
                    self._ensure_writer(last_frame)
                if self._video_writer is not None:
                    self._write_frame(last_frame)
                    self._finalize_video()
                    print(f"Video saved to {self.video_path}")
                else:
                    print("Recording was ON but no writer backend available.")
            else:
                print("Recording was ON but no frame captured on exit.")
        else:
            print("No video recording was active.")

        print("Destroying Visualizer...")
        vis.destroy_window()
        sys.exit(0)   # <-- dùng sys.exit để flush & chạy atexit handlers
    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _center_viewpoint(self, vis):
        vis.reset_view_point(True)

    def _toggle_follow(self, vis):
        self.follow_cam = not self.follow_cam
        print(f"[FollowCam] {'ON' if self.follow_cam else 'OFF'}")
    def _toggle_record(self, vis):
        self.record = not self.record
        print(f"[Record] {'ON' if self.record else 'OFF'}")
        if not self.record:
            self._finalize_video()  # tắt thì đóng ngay

    def _capture_rgb_frame(self):
        """Chụp frame hiện tại từ Open3D (RGB uint8)."""
        # Lấy buffer float [0..1], rồi convert sang uint8
        img = np.asarray(self.vis.capture_screen_float_buffer(do_render=False))
        if img is None or img.size == 0:
            return None
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        # Open3D trả (H, W, 3). Đảm bảo đúng shape:
        if img_u8.ndim == 2:  # phòng hờ grayscale (hiếm)
            img_u8 = np.repeat(img_u8[..., None], 3, axis=2)
        return img_u8  # RGB

    def _ensure_writer(self, frame_rgb):
        """Khởi tạo writer nếu chưa có (dựa theo kích thước frame đầu tiên)."""
        if self._video_writer is not None:
            return

        h, w, _ = frame_rgb.shape
        self._frame_size = (w, h)

        if _HAS_CV2:
            # fourcc mp4v gần như tương thích rộng rãi
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(self.video_path, fourcc, self.fps, self._frame_size)
            if vw is None or not vw.isOpened():
                print("[Record] cv2.VideoWriter open failed, fallback to imageio...")
            else:
                self._video_writer = vw
                self._vw_backend = "cv2"
                print(f"[Record] Writing MP4 via OpenCV → {self.video_path} @ {self.fps} FPS")
                return
        if _HAS_IMAGEIO:
            # imageio v2 writer
            # codec 'libx264' + fps; thêm macro_block_size=None để tránh lỗi kích thước không chia hết 16
            self._video_writer = iio.get_writer(
                self.video_path,
                fps=self.fps,
                codec="libx264",
                macro_block_size=None
            )
            self._vw_backend = "imageio"
            print(f"[Record] Writing MP4 via imageio (libx264) → {self.video_path} @ {self.fps} FPS")

        else:
            print("[Record] No backend available (install opencv-python or imageio[ffmpeg]). Recording disabled.")
            self.record = False

    def _write_frame(self, frame_rgb):
        """Ghi một frame RGB vào video writer hiện tại."""
        if self._video_writer is None:
            return
        if self._vw_backend == "cv2":
            # cv2 yêu cầu BGR
            frame_bgr = frame_rgb[..., ::-1]
            self._video_writer.write(frame_bgr)

        elif self._vw_backend == "imageio":
            # imageio v2 dùng append_data, nhận RGB
            self._video_writer.append_data(frame_rgb)

    def _capture_and_record(self):
        """Nếu record đang bật, capture và ghi frame hiện tại."""
        if not self.record:
            return
        frame = self._capture_rgb_frame()
        if frame is None:
            return
        if self._video_writer is None:
            self._ensure_writer(frame)
            if self._video_writer is None:
                return
        self._write_frame(frame)

    def _finalize_video(self):
        """Đóng writer an toàn."""
        if self._video_writer is None:
            return
        try:
            if self._vw_backend == "cv2":
                self._video_writer.release()
            # elif self._vw_backend == "imageio":
            #     self._video_writer.close()
            elif self._vw_backend == "imageio":
                self._video_writer.close()
            print(f"[Record] Saved video → {self.video_path}")
        except Exception as e:
            print(f"[Record] finalize error: {e}")
        finally:
            self._video_writer = None
            self._vw_backend = None
            self._frame_size = None

    def _add_line(self, pose0, pose1, color, z_offset=0.0):
    # Nâng Z để tránh chìm trong point cloud
        p0 = pose0.copy()
        p1 = pose1.copy()
        p0[2] += z_offset
        p1[2] += z_offset

        lines = [[0, 1]]
        colors = [color for _ in lines]
        line_set = self.o3d.geometry.LineSet()
        line_set.points = self.o3d.utility.Vector3dVector([p0, p1])
        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        line_set.colors = self.o3d.utility.Vector3dVector(colors)
        return line_set

    def _cylinder_between(self, p0, p1, radius=0.1, color=None, resolution=12):
        # p0, p1: np.array([x,y,z])
        axis = p1 - p0
        L = np.linalg.norm(axis)
        if L < 1e-9:
            return None
        z = axis / L

        # dựng hệ trục: z = trục ống
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-9)
        y = np.cross(z, x)

        R = np.eye(4)
        R[:3, 0] = x; R[:3, 1] = y; R[:3, 2] = z; R[:3, 3] = p0

        cyl = self.o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=resolution, split=1)
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color(self.theme["trajectory"] if color is None else color)
        # Dời cylinder sao cho đáy ở p0 và trục trùng z của local frame
        # create_cylinder mặc định tâm ở gốc, dài theo +Z và đối xứng quanh tâm (cao từ -L/2..+L/2)
        # nên cần dịch +L/2 dọc trục z trước khi transform R:
        T = np.eye(4)
        T[:3, 3] = np.array([0, 0, L/2.0])
        cyl.transform(T)
        cyl.transform(R)
        return cyl

    def _add_frames(self, poses, size, color):
        frames = []
        for pose in poses:
            frames.append(self._add_frame(pose, size, color))
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

        # --- giữ hiệu ứng lật trong chế độ follow ---
        if self.flip_view_z:
            cam_pos = cam_pos.copy(); lookat = lookat.copy(); up = up.copy()
            cam_pos[2] *= -1.0
            lookat[2]  *= -1.0
            up[2]      *= -1.0

        front = (lookat - cam_pos)
        n = np.linalg.norm(front)
        if n < 1e-9:
            return
        front /= n
        up_n = up / (np.linalg.norm(up) + 1e-9)

        ctr.set_lookat(lookat.tolist())
        ctr.set_front(front.tolist())
        ctr.set_up(up_n.tolist())
        ctr.set_zoom(zoom)
        
    # ----- Scene update -----
    def _update_geometries(self, slam):
    # Local map points in global
        current_node = slam.local_map_graph.last_local_map
        local_map_in_global = transform_points(
            slam.voxel_grid.point_cloud(), current_node.keypose
        )

        # [NEW] Lật Z cho toàn bộ point cloud
        if FLIP_Z:
            local_map_in_global = _flip_z_points(local_map_in_global)

        self.local_map.points = self.o3d.utility.Vector3dVector(local_map_in_global)
        self.local_map.paint_uniform_color(self.theme["localmap"])
        self.vis.update_geometry(self.local_map)

        # ---- Tính z_plane (neo quỹ đạo) trên cloud đã lật ----
        if local_map_in_global.shape[0] > 0:
            # segment_plane chạy trên self.local_map (đã là cloud đã lật)
            plane_model, inliers = self.local_map.segment_plane(
                distance_threshold=0.05, ransac_n=3, num_iterations=500
            )
            a, b, c, d = plane_model

            # [CHANGED] dùng vị trí pose đã lật để tính z_ground
            t_disp = _flip_z_pose(current_node.endpose)[:3, 3] if FLIP_Z else current_node.endpose[:3, 3]
            if abs(c) > 1e-6:
                z_ground = (-a * t_disp[0] - b * t_disp[1] - d) / c
                target = z_ground + TRAJ_Z_OFFSET + self.z_offset_delta
            else:
                target = float(t_disp[2]) + TRAJ_Z_OFFSET + self.z_offset_delta
        else:
            # [CHANGED] fallback dùng pose đã lật
            t_disp = _flip_z_pose(current_node.endpose)[:3, 3] if FLIP_Z else current_node.endpose[:3, 3]
            target = float(t_disp[2]) + TRAJ_Z_OFFSET + self.z_offset_delta

        alpha = 0.15
        self.z_plane = target if self.z_plane is None else (1 - alpha) * self.z_plane + alpha * target
        z_anchor = self.z_plane

        # 3) Lấy pose hiện tại (đã lật phần tịnh tiến Z)
        current_pose = _flip_z_pose(current_node.endpose) if FLIP_Z else current_node.endpose

        # 4) Marker ở pose hiện tại
        # odom_frame = self._add_frame(current_pose, SPHERE_SIZE_ODOMETRY, self.theme["odom"])
        # self.odom_frames.append(odom_frame)
        # self.vis.add_geometry(odom_frame, reset_bounding_box=False)
        # 4) Marker duy nhất tại pose hiện tại (di chuyển thay vì add mới)
        # if self.current_marker is None:
        #     # tạo lần đầu
        #     self.current_marker = self.o3d.geometry.TriangleMesh.create_sphere(
        #         radius=SPHERE_SIZE_ODOMETRY, resolution=12
        #     )
        #     self.current_marker.paint_uniform_color(self.theme["odom"])
        #     self.current_marker.compute_vertex_normals()
        #     self.current_marker_pose = np.eye(4)
        #     self.vis.add_geometry(self.current_marker, reset_bounding_box=False)

        # # di chuyển marker từ pose cũ -> pose mới dùng biến đổi tương đối
        # T_new = current_pose
        # T_prev = self.current_marker_pose
        # # T_delta = inv(T_prev) @ T_new
        # T_delta = np.linalg.inv(T_prev) @ T_new
        # self.current_marker.transform(T_delta)
        # self.current_marker_pose = T_new
        # self.vis.update_geometry(self.current_marker)

        # 5) Quỹ đạo (ống)
        self._update_odom_tubes(current_pose, z_anchor, radius=self.tube_radius)

        # 6) Follow camera
        if self.follow_cam:
            self._follow_pose(current_pose)

        # 7) Keyframes: lật phần tịnh tiến z trước khi vẽ spheres/edges
        key_poses = slam.get_keyposes()
        if key_poses != self.key_poses:
            for frame in self.key_frames:
                self.vis.remove_geometry(frame, reset_bounding_box=False)
            self.key_frames = []

            kf_transformed = [(_flip_z_pose(k) if FLIP_Z else k) for k in key_poses]
            self.key_frames = self._add_frames(kf_transformed, SPHERE_SIZE_KEYPOSES, self.theme["keyframe"])
            for frame in self.key_frames:
                self.vis.add_geometry(frame, reset_bounding_box=False)
            self.key_poses = key_poses

            for edge in self.edges:
                self.vis.remove_geometry(edge, reset_bounding_box=False)
            self.edges = []

            for (idx0, idx1) in self.closures:
                p0 = self.key_frames[idx0].get_center()
                p1 = self.key_frames[idx1].get_center()
                self.edges.append(self._add_line(p0, p1, self.theme["closure"]))

            for e in self.edges:
                self.vis.add_geometry(e, reset_bounding_box=False)

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False
