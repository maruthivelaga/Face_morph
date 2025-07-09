import cv2
import numpy as np
import mediapipe as mp
from skimage.exposure import match_histograms
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)

# Detect facial landmarks
def get_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    h, w = image.shape[:2]
    return np.array([[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark], np.float32)

# Delaunay triangulation
def calculate_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []
    pt_dict = {tuple(p): i for i, p in enumerate(points)}
    for t in triangle_list:
        pts = [(int(t[i]), int(t[i+1])) for i in range(0, 6, 2)]
        if all(0 <= p[0] < rect[2] and 0 <= p[1] < rect[3] for p in pts):
            try:
                idx = [pt_dict[tuple(min(points, key=lambda x: np.linalg.norm(np.array(x) - np.array(p))))] for p in pts]
                if len(set(idx)) == 3:
                    delaunay_tri.append(idx)
            except:
                continue
    return delaunay_tri

# Warp triangle region
def warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    t1_rect = [(p[0] - r1[0], p[1] - r1[1]) for p in t_src]
    t2_rect = [(p[0] - r2[0], p[1] - r2[1]) for p in t_dst]

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0))

    src_crop = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src_crop.shape[0] == 0 or src_crop.shape[1] == 0:
        return

    mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv2.warpAffine(src_crop, mat, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    dst_crop = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if dst_crop.shape == warped.shape:
        dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst_crop * (1 - mask) + warped * mask

# Morph and blend
def morph_faces(src_img, src_points, dst_img, dst_points, triangles):
    morphed = dst_img.copy()
    for tri in triangles:
        t1 = [src_points[i] for i in tri]
        t2 = [dst_points[i] for i in tri]
        warp_triangle(src_img, morphed, t1, t2)
    return morphed

def main():
    # Load and preprocess target
    target_img = cv2.imread("ranbir.jpeg")
    if target_img is None:
        print("❌ target.jpg not found.")
        return
    target_img = cv2.resize(target_img, (640, 480))

    target_landmarks = get_landmarks(target_img)
    if target_landmarks is None:
        print("❌ Could not detect face in target.")
        return
    print("✅ Target face loaded.")

    rect = (0, 0, 640, 480)
    triangles = calculate_delaunay_triangles(rect, target_landmarks)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    cv2.namedWindow("Live Morph")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        landmarks = get_landmarks(frame)

        if landmarks is not None:
            target_adjusted = match_histograms(target_img, frame, channel_axis=-1)
            warped = morph_faces(target_adjusted, target_landmarks, frame.copy(), landmarks, triangles)

            hull = cv2.convexHull(landmarks.astype(np.int32))
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # single channel mask
            cv2.fillConvexPoly(mask, hull, 255)

# Ensure mask matches expected input
            if mask.sum() == 0:
              cv2.imshow("Live Morph", frame)
              continue

# Calculate center safely
            raw_center = np.mean(landmarks, axis=0).astype(int)
            center = (
            np.clip(raw_center[0], 0, frame.shape[1] - 1),
            np.clip(raw_center[1], 0, frame.shape[0] - 1)
            )

# Perform seamless clone
            try:
                output = cv2.seamlessClone(warped, frame, mask, center, cv2.NORMAL_CLONE)
                cv2.imshow("Live Morph", output)
            except cv2.error as e:
                 print(f"[!] seamlessClone error: {e}")
                 cv2.imshow("Live Morph", frame)
                 cv2.imshow("Live Morph", output)
        else:
            cv2.putText(frame, "❌ No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Live Morph", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
