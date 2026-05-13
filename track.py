"""
Face / head pose → direction labels (same math as original track.py).

Import `HeadTracker` into Panda3D (see test.py) or run `moving.py` to calibrate then load the full world.
Standalone: `python track.py`
"""

from __future__ import annotations

import math
import pathlib
import urllib.request

import cv2
import numpy as np

_FACE_LMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

X_THRESHOLD = 0.01
Y_THRESHOLD = 0.01
Z_ENTER_THRESHOLD = 0.04
EMA_ALPHA = 0.25
NEUTRAL_SIZE_TOLERANCE = 0.05
PLANAR_IDLE_MAX = 0.35


#makes sure that the face_landmarker.task file is downloaded and can be used, and downloads it if it doesn't exist
def _ensure_face_landmarker_model(path: pathlib.Path) -> pathlib.Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    print("Downloading face_landmarker.task (MediaPipe Tasks, one-time)…")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_FACE_LMARKER_MODEL_URL, timeout=120) as resp, open(path, "wb") as f:
        f.write(resp.read())
    return path


_MESH_EDGE_PAIRS: list[tuple[int, int]] | None = None

#Uses connection data from the face_landmarker.task file to get the pairs of landmarks in integers that make up the face mesh
def _pair_from_connection(conn) -> tuple[int, int] | None:
    """MediaPipe FACEMESH_* entries vary: tuples, NamedTuples ``(x,y)``, ``(first,second)``, frozensets, etc."""
    if isinstance(conn, (tuple, list)) and len(conn) >= 2:
        return int(conn[0]), int(conn[1])
    if isinstance(conn, frozenset):
        el = tuple(conn)
        if len(el) == 2:
            return int(el[0]), int(el[1])
        return None
    for fld_a, fld_b in (("first", "second"), ("x", "y")):
        if hasattr(conn, fld_a) and hasattr(conn, fld_b):
            try:
                return int(getattr(conn, fld_a)), int(getattr(conn, fld_b))
            except (TypeError, ValueError):
                pass
    try:
        t = tuple(conn)
        if len(t) >= 2:
            return int(t[0]), int(t[1])
    except (TypeError, ValueError):
        pass
    return None

#this makes and loads the face mesh grid
def _get_facemesh_tesselation_pairs() -> list[tuple[int, int]]:
    """Face mesh edges from MediaPipe topology (Legacy + Tasks overlays share indices)."""
    global _MESH_EDGE_PAIRS
    if _MESH_EDGE_PAIRS is not None:
        return _MESH_EDGE_PAIRS

    tess_sources: list = []
    import importlib

    for mod_path in (
        "mediapipe.solutions.face_mesh",
        "mediapipe.python.solutions.face_mesh",
        "mediapipe.solutions.face_mesh_connections",
    ):
        try:
            mod = importlib.import_module(mod_path)
            tess = getattr(mod, "FACEMESH_TESSELATION", None)
            if tess is not None:
                tess_sources.append(tess)
        except (ImportError, AttributeError, TypeError):
            continue

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    #adds the pairs of landmarks to the list, ensuring that each pair is only added once, and organizes all face connecdtions in a single list
    def add_pair(p: tuple[int, int] | None) -> None:
        if p is None:
            return
        a, b = p if p[0] < p[1] else (p[1], p[0])
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))

    for tess in tess_sources:
        for conn in tess:
            add_pair(_pair_from_connection(conn))

    _MESH_EDGE_PAIRS = pairs
    return _MESH_EDGE_PAIRS
class HeadTracker:
    

#This initializes the tracker settings, open webcam, and sets up the AI face detection system
    def __init__(
        self,
        camera_index: int = 0,
        *,
        verbose_direction_print: bool = False,
        window_title: str = "Face Tracking",
        draw_landmarks_legacy: bool = True,
    ):
        self.verbose_direction_print = verbose_direction_print
        self.window_title = window_title
        self.draw_landmarks_legacy = draw_landmarks_legacy

        self.direction = "Looking straight"
        self.running = True
        self.calibrated = False
        self.calibrate_request = False

        self.neutral_nose_x = 0.0
        self.neutral_nose_y = 0.0
        self.neutral_face_width = 0.0
        self.neutral_head_height = 0.0
        self.smooth_face_width = None
        self.smooth_face_height = None

        self._post_cal_show_frames = 0

        self.video = cv2.VideoCapture(camera_index)
        if not self.video.isOpened():
            print(
                f"Could not open camera index {camera_index}. "
                "Check System Settings → Privacy & Security → Camera, or try another index."
            )

        self._detect_mode = None
        self._landmarker_ts_ms = 0
        self._MpVisionImage = None
        self._mp_srgb_fmt = None
        self._drawing = None
        self._drawing_spec = None
        self._legacy_face_mesh = None

        try:
            import mediapipe.solutions.face_mesh as _fm  # type: ignore

            self._detect_mode = "legacy"
            self._detector = _fm.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75,
            )
            self._legacy_face_mesh = _fm
            import mediapipe.solutions.drawing_utils as _drawing  # type: ignore

            self._drawing = _drawing
            self._drawing_spec = _drawing.DrawingSpec(thickness=1, circle_radius=1)
        except (ImportError, AttributeError, ModuleNotFoundError):
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python_tasks
            from mediapipe.tasks.python.vision.core.image import Image as MpVisionImage
            from mediapipe.tasks.python.vision.core.image import ImageFormat as MpVisionFormat

            self._MpVisionImage = MpVisionImage
            self._mp_srgb_fmt = MpVisionFormat.SRGB
            model_path = _ensure_face_landmarker_model(
                pathlib.Path(__file__).resolve().parent / "face_landmarker.task"
            )
            self._detect_mode = "tasks"
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python_tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.75,
                min_face_presence_confidence=0.75,
            )
            self._detector = vision.FaceLandmarker.create_from_options(opts)

    
    #Gets the AI to find the coordinates of face landmarks from the video
    def _lm_from_frame(self, image_rgb):
        if self._detect_mode == "legacy":
            res = self._detector.process(image_rgb)
            if not res.multi_face_landmarks:
                return None, None
            fl = res.multi_face_landmarks[0]
            return fl.landmark, fl
        rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        packed = self._MpVisionImage(image_format=self._mp_srgb_fmt, data=rgb)
        self._landmarker_ts_ms += 33
        res = self._detector.detect_for_video(packed, self._landmarker_ts_ms)
        if not res.face_landmarks:
            return None, None
        return res.face_landmarks[0], None

    
    #draws the face mesh grid on the video
    def _draw_face_mesh_overlay(self, image_bgr, lm, face_for_draw) -> None:
        """Original track-style face mesh: Mediapipe ``draw_landmarks`` when legacy; OpenCV tessellation fallback for Tasks."""
        if lm is None:
            return

        legacy_ok = (
            self._detect_mode == "legacy"
            and self._drawing is not None
            and face_for_draw is not None
            and self.draw_landmarks_legacy
        )
        if legacy_ok:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self._drawing.draw_landmarks(
                image=rgb,
                landmark_list=face_for_draw,
                connections=self._legacy_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self._drawing_spec,
                connection_drawing_spec=self._drawing_spec,
            )
            np.copyto(image_bgr, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return

        pairs = _get_facemesh_tesselation_pairs()
        ih, iw = image_bgr.shape[:2]
        try:
            n = len(lm)
        except TypeError:
            n = lm.__len__()

        line_color = (0, 255, 255)
        thickness = max(1, min(iw, ih) // 400)

        if pairs:
            for a, b in pairs:
                if a >= n or b >= n:
                    continue
                try:
                    pa, pb = lm[a], lm[b]
                except (IndexError, TypeError):
                    continue
                pt1 = (int(pa.x * iw), int(pa.y * ih))
                pt2 = (int(pb.x * iw), int(pb.y * ih))
                cv2.line(image_bgr, pt1, pt2, line_color, thickness, lineType=cv2.LINE_AA)
        else:
            for i in range(n):
                try:
                    p = lm[i]
                    cv2.circle(
                        image_bgr,
                        (int(p.x * iw), int(p.y * ih)),
                        max(1, thickness),
                        line_color,
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                except (IndexError, TypeError):
                    break

    
    #Uses math to compare current head position to calibrated center
    def _compute_direction(self, lm) -> str:
        """Same rules as original track.py (depth first, then planar + idle band)."""
        nose = lm[1]
        nose_x, nose_y = nose.x, nose.y
        left_cheek, right_cheek = lm[234], lm[454]
        face_width = math.hypot(right_cheek.x - left_cheek.x, right_cheek.y - left_cheek.y)
        top_face, bottom_face = lm[10], lm[152]
        face_height = math.hypot(top_face.y - bottom_face.y, top_face.x - bottom_face.x)

        if self.smooth_face_width is None:
            self.smooth_face_width = face_width
        else:
            self.smooth_face_width = EMA_ALPHA * face_width + (1 - EMA_ALPHA) * self.smooth_face_width
        if self.smooth_face_height is None:
            self.smooth_face_height = face_height
        else:
            self.smooth_face_height = EMA_ALPHA * face_height + (1 - EMA_ALPHA) * self.smooth_face_height

        dx = self.neutral_nose_x - nose_x
        dy = nose_y - self.neutral_nose_y

        width_base = max(self.neutral_face_width, 1e-6)
        height_base = max(self.neutral_head_height, 1e-6)

        width_low = self.neutral_face_width - NEUTRAL_SIZE_TOLERANCE
        width_high = self.neutral_face_width + NEUTRAL_SIZE_TOLERANCE
        height_low = self.neutral_head_height - NEUTRAL_SIZE_TOLERANCE
        height_high = self.neutral_head_height + NEUTRAL_SIZE_TOLERANCE

        sw, sh = self.smooth_face_width, self.smooth_face_height
        if width_low <= sw <= width_high:
            width_reference = self.neutral_face_width
        elif sw > width_high:
            width_reference = width_high
        else:
            width_reference = width_low

        if height_low <= sh <= height_high:
            height_reference = self.neutral_head_height
        elif sh > height_high:
            height_reference = height_high
        else:
            height_reference = height_low

        dw_norm = (sw - width_reference) / width_base
        dh_norm = (sh - height_reference) / height_base
        z_score = 0.6 * dw_norm + 0.4 * dh_norm

        sx = abs(dx) / max(X_THRESHOLD, 1e-9)
        sy = abs(dy) / max(Y_THRESHOLD, 1e-9)

        if z_score > Z_ENTER_THRESHOLD:
            return "Moving forward"
        if z_score < -Z_ENTER_THRESHOLD:
            return "Moving back"

        x_act = abs(dx) > X_THRESHOLD
        y_act = abs(dy) > Y_THRESHOLD
        best_planar = max(sx, sy)

        if not x_act and not y_act and best_planar < PLANAR_IDLE_MAX:
            return "Looking straight"
        if x_act and y_act:
            if sx >= sy:
                return "Looking right" if dx > 0 else "Looking left"
            return "Looking down" if dy > 0 else "Looking up"
        if x_act:
            return "Looking right" if dx > 0 else "Looking left"
        if y_act:
            return "Looking down" if dy > 0 else "Looking up"
        return "Looking straight"

    #Uses math to compare current head position to calibrated center after calibration is requested
    def _maybe_calibrate(self, lm) -> None:
        if not self.calibrate_request:
            return
        nose = lm[1]
        nose_x, nose_y = nose.x, nose.y
        left_cheek, right_cheek = lm[234], lm[454]
        face_width = math.hypot(right_cheek.x - left_cheek.x, right_cheek.y - left_cheek.y)
        top_face, bottom_face = lm[10], lm[152]
        face_height = math.hypot(top_face.y - bottom_face.y, top_face.x - bottom_face.x)

        self.neutral_nose_x = nose_x
        self.neutral_nose_y = nose_y
        self.neutral_face_width = face_width
        self.neutral_head_height = face_height
        self.smooth_face_width = face_width
        self.smooth_face_height = face_height
        self.calibrated = True
        self.calibrate_request = False
        print("Calibrated")

    #runs window that allows the user to calibrate the head tracker by pressing C before the world is loaded
    def tick_calibration_only(self):
        if not self.running:
            return "quit"
        if not self.video.isOpened():
            cv2.waitKey(1)
            return None

        ret, image = self.video.read()
        if not ret:
            cv2.waitKey(1)
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            lm, face_for_draw = self._lm_from_frame(image_rgb)
        except Exception as e:
            print("Face detection error:", e)
            lm, face_for_draw = None, None

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            self.calibrate_request = True

        cv2.putText(
            image,
            "Calibrate (then world opens)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if lm is not None:
            cv2.putText(
                image,
                "Head detected — Press C HERE to calibrate",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            self._draw_face_mesh_overlay(image, lm, face_for_draw)
            self._maybe_calibrate(lm)
        else:
            cv2.putText(
                image,
                "Show your face to the camera",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                image,
                "(then press C in this window)",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

        if self.calibrated:
            self._post_cal_show_frames += 1
            cv2.putText(
                image,
                "Calibrated! Loading 3D world...",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            if self._post_cal_show_frames >= 45:
                cv2.destroyAllWindows()
                return "done"

        cv2.imshow("Face Tracking — calibration", image)
        if key == ord("q"):
            self.running = False
            cv2.destroyAllWindows()
            return "quit"
        return None

    def tick(self) -> bool:
        """One frame: capture, update self.direction, show OpenCV. False = stop (q). Main thread only."""
        if not self.running:
            return False
        if not self.video.isOpened():
            cv2.waitKey(1)
            return True

        ret, image = self.video.read()
        if not ret:
            cv2.waitKey(1)
            return True

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            lm, face_for_draw = self._lm_from_frame(image_rgb)
        except Exception as e:
            print("Head tracking error:", e)
            lm, face_for_draw = None, None

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            self.calibrate_request = True

        if lm is not None:
            cv2.putText(image, "Head detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self._draw_face_mesh_overlay(image, lm, face_for_draw)

            self._maybe_calibrate(lm)

            if self.calibrated:
                self.direction = self._compute_direction(lm)
                if self.verbose_direction_print:
                    print(self.direction)
                cv2.putText(
                    image,
                    self.direction,
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    image,
                    "Look straight and press C to calibrate",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
        else:
            cv2.putText(image, "No head detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(self.window_title, image)
        if key == ord("q"):
            self.running = False
            cv2.destroyAllWindows()
            return False
        return True

    def close(self):
        if self._detect_mode == "tasks" and hasattr(self._detector, "close"):
            self._detector.close()
        if self.video.isOpened():
            self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print(
        "OpenCV-only preview. For calibrate-then-full-world, run: python moving.py\n"
        "Or open the world directly (webcam during play): python test.py"
    )
    tracker = HeadTracker(verbose_direction_print=True)
    try:
        while tracker.tick():
            pass
    finally:
        tracker.close()
