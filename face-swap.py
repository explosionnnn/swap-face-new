import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
from numpy.linalg import norm
import torch

# ----------------------------
# 初始化 InsightFace
# ----------------------------
print("Loading InsightFace...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace loaded.")

# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Face Swap using InsightFace + GFPGAN + Reference Face")
    parser.add_argument('--source', type=str, required=True, help='C:/Users/tony0/Downloads/Homework1/Ma.bmp"')
    parser.add_argument('--input', type=str, required=True, help='C:/Users/tony0/Downloads/Homework1/AI-muscle.mp4')
    parser.add_argument('--output', type=str, required=True, help='C:/Users/tony0/Downloads/Homework1/output.mp4')
    parser.add_argument('--reference', type=str, required=True, help='C:/Users/tony0/Downloads/Homework1/螢幕擷取畫面 2025-10-22 094048.png')
    parser.add_argument('--gfpgan', action='store_true', help='Use GFPGAN enhancement')
    parser.add_argument('--threshold', type=float, default=0.3, help='Cosine similarity threshold')
    return parser.parse_args()

# ----------------------------
# 載入影像
# ----------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image from {path}")
    return img

# ----------------------------
# 計算 cosine similarity
# ----------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

# ----------------------------
# 初始化 GFPGAN
# ----------------------------
def init_gfpgan():
    use_cuda = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    device = "cuda" if use_cuda else "cpu"
    print(f"Initializing GFPGAN on {device}...")

    try:
        gfpgan_model = GFPGANer(
            model_path='experiments/pretrained_models/GFPGANv1.3.pth',
            upscale=1,
            arch='clean',
            device=device
        )
        print(f"GFPGAN loaded on {device}.")
        return gfpgan_model
    except Exception as e:
        print(f"GFPGAN initialization failed: {e}")
        return None


def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv2.warpAffine(
        img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]],
        M,
        (r2[2], r2[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    img2_rect = img2_rect * (1 - mask) + warped * mask
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_rect

# ----------------------------
# 換臉函數
# ----------------------------
def swap_faces(source_img, target_img, ref_embedding, gfpgan=None, alpha=0.65, threshold=0.3):

    source_faces = app.get(source_img)
    target_faces = app.get(target_img)
    if len(source_faces) == 0 or len(target_faces) == 0:
        return target_img

    source_face = source_faces[0]
    source_landmarks = source_face.landmark_2d_106

    output_img = target_img.copy()

    for t_face in target_faces:
        target_embedding = t_face.embedding
        sim = cosine_similarity(ref_embedding, target_embedding)
        if sim < threshold:
            continue

        target_landmarks = t_face.landmark_2d_106.astype(np.float32)

        top_y = np.min(target_landmarks[:, 1])
        top_x = np.mean(target_landmarks[:, 0])
        h = np.max(target_landmarks[:, 1]) - np.min(target_landmarks[:, 1])
        w = np.max(target_landmarks[:, 0]) - np.min(target_landmarks[:, 0])

        # =====================================================
        #  A) 25 點額頭網格：上/中/下 三層（7 + 9 + 9 = 25）
        #     橫向覆蓋額頭與髮際下緣，等距分佈以穩定 Delaunay。
        # =====================================================
        layers = [
            (-0.12 * h, 7),   # 接近眉上緣
            (-0.20 * h, 9),   # 額頭中層
            (-0.27 * h, 9),   # 接近髮際下緣
        ]
        extra_points_list = []
        for y_off, n_pts in layers:
            x_start = top_x - 0.35 * w
            x_end   = top_x + 0.35 * w
            xs = np.linspace(x_start, x_end, int(n_pts))
            ys = np.full_like(xs, top_y + y_off, dtype=np.float32)
            extra_points_list.append(np.stack([xs.astype(np.float32), ys], axis=1))
        extra_points = np.concatenate(extra_points_list, axis=0).astype(np.float32)  # shape: (25,2)

        all_target = np.vstack([target_landmarks, extra_points])  # (N+3,2)

        all_landmarks = all_target

        # ========= [這裡開始是你要插入的仿射估計區塊] =========
        # （選）只用「上半臉」來估仿射，對表情更穩
        tl = target_landmarks.astype(np.float32)
        sl = source_landmarks.astype(np.float32)
        y_med = np.median(tl[:, 1])
        upper_mask = tl[:, 1] < y_med  # 上半臉：眉/眼/鼻樑附近

        tl_fit = tl[upper_mask] if np.sum(upper_mask) >= 10 else tl
        sl_fit = sl[upper_mask] if np.sum(upper_mask) >= 10 else sl

        # 估 target→source 的相似/仿射（旋轉+縮放+平移）
        M, _ = cv2.estimateAffinePartial2D(
            tl_fit, sl_fit,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000, confidence=0.99
        )

        def apply_affine(pts, M):
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])  # (N,3)
            out = pts_h @ M.T  # (N,2)
            return out.astype(np.float32)

        if M is not None:
            extra_src = apply_affine(extra_points, M)  # 把額外點帶到來源臉座標
        else:
            # 估不到就退回你原本的中心平移法
            src_center = np.mean(source_landmarks, axis=0)
            tgt_center = np.mean(target_landmarks, axis=0)
            extra_src = src_center + (extra_points - tgt_center)

        # 來源所有點（106 + 額外對應點）
        all_source = np.vstack([source_landmarks.astype(np.float32), extra_src])
        # ========= [仿射估計區塊到此結束] =========


        # ---------- B) 逐三角形仿射：用 all_target 建 Delaunay ----------
        # 用 cv2.Subdiv2D，但要把三角形頂點「對回」我們的點索引
        rect = cv2.boundingRect(all_target.astype(np.float32))
        subdiv = cv2.Subdiv2D(rect)
        for p in all_target:
            subdiv.insert((float(p[0]), float(p[1])))

        triangles = subdiv.getTriangleList().reshape(-1, 6)  # [x1,y1,x2,y2,x3,y3]

        # 為了把 getTriangleList() 的浮點座標對回 our points，
        # 建立一個 KD 查表（或簡單最近鄰）：這次一定用 all_target，不是 target_landmarks！
        # （若擔心誤配，可先四捨五入後 dict 映射）
        # 這裡用最近鄰，並做唯一性檢查避免退化三角形。
        warped_source = np.zeros_like(target_img)

        for tri in triangles:
            p1 = np.array([tri[0], tri[1]], dtype=np.float32)
            p2 = np.array([tri[2], tri[3]], dtype=np.float32)
            p3 = np.array([tri[4], tri[5]], dtype=np.float32)

            # 找到各頂點在 all_target 中的索引
            idx1 = np.argmin(np.linalg.norm(all_target - p1, axis=1))
            idx2 = np.argmin(np.linalg.norm(all_target - p2, axis=1))
            idx3 = np.argmin(np.linalg.norm(all_target - p3, axis=1))

            # 跳過退化或重複索引的三角形
            if len({idx1, idx2, idx3}) < 3:
                continue

            pts_tgt = np.float32([all_target[idx1], all_target[idx2], all_target[idx3]])
            pts_src = np.float32([all_source[idx1], all_source[idx2], all_source[idx3]])

            warp_triangle(source_img, warped_source, pts_src, pts_tgt)

        # 🟢 2️⃣ 放大整體遮罩範圍
        center = np.mean(all_landmarks, axis=0)
        scale_x = 1.10
        scale_y = 1.25
        expanded_landmarks = (all_landmarks - center) * [scale_x, scale_y] + center

        # 🟣 3️⃣ 建立最終遮罩
        mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask,
                        cv2.convexHull(expanded_landmarks.astype(np.int32)),
                        255)


        # 邊緣平滑（注意 kernel 要奇數）
        # 先高斯再形態膨脹，讓兩側更「有料」
        mask = cv2.GaussianBlur(mask, (13, 13), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 與 warped_source 的有效區域取交集（避免額頭黑）
        valid = (warped_source.sum(axis=2) > 0).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, valid)

        # --------- C. 融合（疊到 output_img，非 target_img）---------
        mask_f = (mask.astype(np.float32) / 255.0)[..., None] * float(alpha)
        swapped = (warped_source.astype(np.float32) * mask_f
                   + output_img.astype(np.float32) * (1.0 - mask_f)).astype(np.uint8)


     # ---- 使用新版 GFPGAN API ----
        if gfpgan is not None:
            try:
                # 這裡用 warped_source 進修復，再重做融合會更自然
                restored_faces, restored_img, _ = gfpgan.enhance(
                    warped_source, has_aligned=False, only_center_face=False, paste_back=True
                )
                if isinstance(restored_img, np.ndarray) and restored_img.size > 0:
                    warped_source = restored_img
                    # 重新計算 valid & 融合（避免用舊的 swapped）
                    valid = (warped_source.sum(axis=2) > 0).astype(np.uint8) * 255
                    mask = cv2.bitwise_and(mask, valid)
                    mask_f = (mask.astype(np.float32) / 255.0)[..., None] * float(alpha)
                    swapped = (warped_source.astype(np.float32) * mask_f
                               + output_img.astype(np.float32) * (1.0 - mask_f)).astype(np.uint8)
            except Exception as e:
                print(f"GFPGAN enhance failed, fallback to original warp: {e}")
                # 失敗就使用原本 warp，不返回 None
                pass

        output_img = swapped

    return output_img


# ----------------------------
# 影片處理
# ----------------------------
def process_video(source_img, input_path, output_path, ref_embedding, gfpgan=None, threshold=0.3):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        swapped = swap_faces(source_img, frame, ref_embedding, gfpgan=gfpgan, threshold=threshold)
        if swapped is None:
            cap.release()
            out.release()
            print("\nGFPGAN failed during video processing. Re-running without GFPGAN...")
            return False

        out.write(swapped)
        frame_idx += 1

        # 顯示百分比進度條
        progress = (frame_idx / total_frames) * 100
        bar_length = 30  # 進度條長度
        filled_length = int(bar_length * frame_idx // total_frames)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\rProgress: |{bar}| {progress:6.2f}% ({frame_idx}/{total_frames})", end='')

    cap.release()
    out.release()
    print("\nVideo processing finished.")
    return True

# ----------------------------
# 三角形切割
# ----------------------------




# ----------------------------
# 主程式
# ----------------------------
def main():
    args = parse_args()
    source_img = load_image(args.source)
    reference_img = load_image(args.reference)

    ref_faces = app.get(reference_img)
    if len(ref_faces) == 0:
        raise ValueError("Reference face not detected!")
    ref_embedding = ref_faces[0].embedding

    gfpgan_model = None
    if args.gfpgan:
        gfpgan_model = init_gfpgan()

    success = process_video(source_img, args.input, args.output, ref_embedding, gfpgan=gfpgan_model, threshold=args.threshold)

    # 如果 GFPGAN 整段影片失敗，重新跑一次 CPU-only
    if not success:
        print("Re-running video without GFPGAN...")
        process_video(source_img, args.input, args.output, ref_embedding, gfpgan=None, threshold=args.threshold)

if __name__ == "__main__":
    main()