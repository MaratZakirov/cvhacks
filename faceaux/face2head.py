import cv2
import numpy as np
from matplotlib import pyplot as plt

# Канонические 3D точки (модель)
pts_3d = np.array([
    [ 0.0, -0.4, -0.5],   # nose
    [-0.5,  0.0,  0.0],   # left_eye
    [ 0.5,  0.0,  0.0],   # right_eye
    [-1.0,  0.0,  1.0],   # left_ear
    [ 1.0,  0.0,  1.0]    # right_ear
], dtype=np.float64)

# AUX points
pts_3d_aux = np.array([
    [0.0,  1.0, 2.0],   # макушка
    [0.0, -1.4, 0.1],   # подбородок
    [0.0,  1.0, 0.1],   # лоб
    [0.0, -1.4, 2.0],   # основание шеи
    [0.0,  0.0, 2.3],   # затылок
], dtype=np.float64)

pts_3d_hat = np.array([
    [-1.0,  1.9,  -0.2],
    [-1.0,  1.9,   2.4],
    [-1.0,  0.4,  -0.2],
    [-1.0,  0.4,   2.4],
    [ 1.0,  1.9,  -0.2],
    [ 1.0,  1.9,   2.4],
    [ 1.0,  0.4,  -0.2],
    [ 1.0,  0.4,   2.4],
], dtype=np.float64)

# общий набор дополнительных точек
pts_3d_aux_all = np.vstack([pts_3d_aux, pts_3d_hat])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], c='blue')
ax.scatter(pts_3d_aux[:, 0], pts_3d_aux[:, 1], pts_3d_aux[:, 2], c='red')
ax.scatter(pts_3d_hat[:, 0], pts_3d_hat[:, 1], pts_3d_hat[:, 2], c='green')

# Подписываем номера точек
for i, (x, y, z) in enumerate(pts_3d):
    ax.text(x, y, z, str(i), fontsize=12)

ax.set_xlabel('X Label', labelpad=20)
ax.set_ylabel('Y Label', labelpad=20)
ax.set_zlabel('Z Label', labelpad=20)
ax.set_aspect('equal')
plt.show()

# 2D точки из Ultralytics (nose, left_eye, right_eye, left_ear, right_ear)
# 2D точки (x, y, confidence)
pts_2d = np.array([
    [2228.1, 682.77, 0.99602], # nose
    [2351.7, 581.83, 0.99903], # left_eye
    [2157.2, 612.30, 0.98245], # right_eye
    [2592.3, 676.13, 0.99403], # left_ear
    [2107.5, 728.74, 0.23805]  # right_ear
], dtype=np.float64)

# Рисуем точки
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pts_2d[:, 0], pts_2d[:, 1], color='blue', s=64)

# Подписываем номера
for i, (x, y, conf) in enumerate(pts_2d):
    ax.text(x + 5, y + 5, str(i), fontsize=12)

# Отражаем ось Y
ax.invert_yaxis()
ax.set_aspect('equal')

ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_title('2D Keypoints')

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Приближённые интринсики
W, H = 3840, 2160
fx = fy = W
K = np.array([[fx, 0, W/2],
              [0, fy, H/2],
              [0,  0,   1]], dtype=np.float64)
dist = np.zeros(4)

pts_2d_cv = np.ascontiguousarray(pts_2d[:, :2], dtype=np.float64).reshape(-1, 1, 2)
pts_3d_cv = np.ascontiguousarray(pts_3d, dtype=np.float64)

print("pts_3d:", pts_3d_cv.shape, pts_3d_cv.dtype, pts_3d_cv.flags['C_CONTIGUOUS'])
print("pts_2d:", pts_2d_cv.shape, pts_2d_cv.dtype, pts_2d_cv.flags['C_CONTIGUOUS'])

success, rvec, tvec = cv2.solvePnP(
    pts_3d_cv, pts_2d_cv, K, dist,
    flags=cv2.SOLVEPNP_EPNP
)

print(success, rvec, tvec)

# Проецируем все дополнительные точки
pts_2d_aux_all, _ = cv2.projectPoints(
    pts_3d_aux_all, rvec, tvec, K, dist
)
pts_2d_aux_all = pts_2d_aux_all.reshape(-1, 2)

# разделяем обратно
pts_2d_aux = pts_2d_aux_all[:len(pts_3d_aux)]
pts_2d_hat = pts_2d_aux_all[len(pts_3d_aux):]

# Рисуем точки
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pts_2d[:, 0], pts_2d[:, 1], color='blue', s=64)
ax.scatter(pts_2d_aux[:, 0], pts_2d_aux[:, 1], color='red', s=64)
ax.scatter(pts_2d_hat[:, 0], pts_2d_hat[:, 1], color='green', s=64)

# Подписываем номера
for i, (x, y, conf) in enumerate(pts_2d):
    ax.text(x + 5, y + 5, str(i), fontsize=12)

ax.invert_yaxis()
ax.set_aspect('equal')

ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_title('2D Keypoints')

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

R, _ = cv2.Rodrigues(rvec)
tvec_flat = tvec.reshape(3)

pts_3d_aux_cam = pts_3d_aux @ R.T + tvec_flat
pts_3d_hat_cam = pts_3d_hat @ R.T + tvec_flat
pts_3d_cam = pts_3d @ R.T + tvec_flat

# вектор затылок -> нос
face_vec = pts_3d_aux_cam[4] - pts_3d_cam[0]

# нормализуем лицевой вектор
face_vec = face_vec / np.linalg.norm(face_vec)

# угол между нормалью к камере и лицевым вектором
face_angle_grad = np.arccos(face_vec @ np.array([0, 0, 1])) * 180 / np.pi
print(f'Лицо смотрит в камеру под углом {face_angle_grad:0.1f}°')