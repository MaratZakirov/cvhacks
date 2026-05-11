import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Каноническая 3D модель головы
# -----------------------------
PTS_3D = np.array([
    [ 0.0, -0.4, -0.5],   # 0 nose
    [-0.5,  0.0,  0.0],   # 1 left_eye
    [ 0.5,  0.0,  0.0],   # 2 right_eye
    [-1.0,  0.0,  1.0],   # 3 left_ear
    [ 1.0,  0.0,  1.0],   # 4 right_ear
], dtype=np.float64)

AUX_NAMES = [
    "top_head",
    "chin",
    "forehead",
    "neck_base",
    "back_head",
]

PTS_3D_AUX = np.array([
    [0.0,  1.0,  2.0],   # top_head
    [0.0, -1.4,  0.1],   # chin
    [0.0,  1.0,  0.1],   # forehead
    [0.0, -1.4,  2.0],   # neck_base
    [0.0,  0.0,  2.3],   # back_head
], dtype=np.float64)

HEAD_KPT_IDX = [0, 1, 2, 3, 4]  # nose, l_eye, r_eye, l_ear, r_ear

# COCO colors BGR
BLUE = (255, 120, 0)
RED = (0, 0, 255)
GREEN = (0, 220, 0)
YELLOW = (0, 220, 220)
WHITE = (255, 255, 255)


def get_camera_matrix(w: int, h: int) -> np.ndarray:
    fx = fy = float(w)
    cx = w / 2.0
    cy = h / 2.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def draw_point(img, pt, color, label=None, r=5):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img, (x, y), r, color, -1, lineType=cv2.LINE_AA)
    if label is not None:
        cv2.putText(
            img, label, (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )


def project_points(points_3d, rvec, tvec, K, dist):
    pts_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    return pts_2d.reshape(-1, 2)


def head_pose_from_5pts(pts_2d_xy: np.ndarray, K: np.ndarray, dist: np.ndarray):
    pts_2d_cv = np.ascontiguousarray(pts_2d_xy, dtype=np.float64).reshape(-1, 1, 2)
    pts_3d_cv = np.ascontiguousarray(PTS_3D, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        pts_3d_cv,
        pts_2d_cv,
        K,
        dist,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    pts_3d_cam = PTS_3D @ R.T + t
    pts_3d_aux_cam = PTS_3D_AUX @ R.T + t

    pts_2d_aux = project_points(PTS_3D_AUX, rvec, tvec, K, dist)

    face_vec = pts_3d_aux_cam[4] - pts_3d_cam[0]  # back_head -> nose
    norm = np.linalg.norm(face_vec)
    if norm > 1e-9:
        face_vec = face_vec / norm

    face_angle_deg = np.degrees(np.arccos(np.clip(face_vec @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0)))

    return {
        "rvec": rvec,
        "tvec": tvec,
        "R": R,
        "pts_3d_cam": pts_3d_cam,
        "pts_3d_aux_cam": pts_3d_aux_cam,
        "pts_2d_aux": pts_2d_aux,
        "face_vec": face_vec,
        "face_angle_deg": face_angle_deg,
    }


def pick_best_person(result):
    if result.keypoints is None or result.keypoints.data is None:
        return None

    k = result.keypoints.data.cpu().numpy()  # (num_persons, 17, 3)
    if len(k) == 0:
        return None

    best_i = None
    best_score = -1.0

    for i in range(len(k)):
        head = k[i, HEAD_KPT_IDX, 2]
        score = float(np.mean(head))
        if score > best_score:
            best_score = score
            best_i = i

    return k[best_i]


def main():
    model = YOLO("yolo26n-pose.pt")

    preferred_device = "mps"   # для Mac mini M4
    inference_device = preferred_device

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть вебкамеру")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Управление: q/ESC - выход, m - переключить mps/cpu")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        K = get_camera_matrix(w, h)
        dist = np.zeros((4, 1), dtype=np.float64)

        try:
            results = model.predict(
                source=frame,
                device=inference_device,
                verbose=False,
                imgsz=640,
                conf=0.25
            )
        except Exception as e:
            if inference_device == "mps":
                print(f"MPS ошибка ({e}), переключаюсь на CPU")
                inference_device = "cpu"
                continue
            else:
                raise

        vis = frame.copy()
        result = results[0]
        person = pick_best_person(result)

        if person is not None:
            pts_2d = person[HEAD_KPT_IDX, :2]      # (5, 2)
            confs = person[HEAD_KPT_IDX, 2]        # (5,)

            # Рисуем исходные 5 точек головы
            base_names = ["nose", "l_eye", "r_eye", "l_ear", "r_ear"]
            for i, pt in enumerate(pts_2d):
                draw_point(vis, pt, BLUE, f"{base_names[i]}:{confs[i]:.2f}")

            pose = head_pose_from_5pts(pts_2d, K, dist)

            if pose is not None:
                aux2d = pose["pts_2d_aux"]
                aux3d = pose["pts_3d_aux_cam"]
                face_vec = pose["face_vec"]
                face_angle_deg = pose["face_angle_deg"]

                # Рисуем достроенные точки
                for i, pt in enumerate(aux2d):
                    draw_point(vis, pt, RED, AUX_NAMES[i], r=4)

                # Линия back_head -> nose
                nose_2d = pts_2d[0]
                back_2d = aux2d[4]
                cv2.line(
                    vis,
                    tuple(np.int32(back_2d)),
                    tuple(np.int32(nose_2d)),
                    GREEN,
                    2,
                    lineType=cv2.LINE_AA
                )

                # Текстовая панель
                panel_lines = [
                    f"device: {inference_device}",
                    f"angle to camera normal: {face_angle_deg:.1f} deg",
                    f"face_vec: [{face_vec[0]:+.2f}, {face_vec[1]:+.2f}, {face_vec[2]:+.2f}]",
                    f"top_head_cam: [{aux3d[0,0]:+.2f}, {aux3d[0,1]:+.2f}, {aux3d[0,2]:+.2f}]",
                    f"back_head_cam: [{aux3d[4,0]:+.2f}, {aux3d[4,1]:+.2f}, {aux3d[4,2]:+.2f}]",
                ]

                y0 = 28
                for i, txt in enumerate(panel_lines):
                    cv2.putText(
                        vis, txt, (20, y0 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2, cv2.LINE_AA
                    )
            else:
                cv2.putText(
                    vis, "solvePnP failed", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA
                )
        else:
            cv2.putText(
                vis, f"device: {inference_device} | no person", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA
            )

        cv2.imshow("Head pose + aux points", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("m"):
            inference_device = "cpu" if inference_device == "mps" else "mps"
            print("device =", inference_device)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
