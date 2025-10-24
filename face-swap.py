import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
from numpy.linalg import norm
import torch
import os

# ----------------------------
# CUDA 加速設定
# ----------------------------
def setup_cuda_fast():
    if not torch.cuda.is_available():
        return

    # 1) 讓 cuDNN 自動挑最快的 conv 實作（輸入尺寸穩定時特別有效）
    torch.backends.cudnn.benchmark = True

    # 2) Ampere(8.x)/Ada(8.9) 可用 TF32：幾乎不影響畫質但大幅加速 matmul/conv
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 3) 允許更激進的 float32 matmul 近似（對推論 OK）
    torch.set_float32_matmul_precision('high')  # 'medium' 也行

    # 4) 預先建立一個專用 Stream（可選，避免與預設串流搶資源）
    global _fast_stream
    _fast_stream = torch.cuda.Stream()
    return

# ----------------------------
# 初始化 InsightFace
# ----------------------------
def init_insightface(det_size=(640, 640)):
    """
    初始化 InsightFace，自動使用 GPU (若可用) 或 CPU fallback。

    Args:
        det_size (tuple): 偵測輸入大小，預設 (640, 640)
    
    Returns:
        app (FaceAnalysis): 初始化好的 InsightFace 物件
    """
    # 判斷是否有 GPU
    use_gpu = torch.cuda.is_available()
    provider = 'CUDAExecutionProvider' if use_gpu else 'CPUExecutionProvider'
    ctx_id = 0 if use_gpu else -1

    print(f"Initializing InsightFace on {'GPU' if use_gpu else 'CPU'}...")

    # 建立 FaceAnalysis 物件
    app = FaceAnalysis(providers=[provider])

    # ctx_id = 0 表示第一張 GPU，-1 表示 CPU
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    print("InsightFace loaded.")
    return app

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
    use_cuda = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 6
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

# ----------------------------
# -- 安全版三角形 warp：避免 0 邊界吃黑，做 ROI 裁切與反射邊界 --
# ----------------------------
def warp_triangle(src_img, dst_img, pts_src, pts_dst):
    pts_src = np.float32(pts_src)
    pts_dst = np.float32(pts_dst)

    r1 = cv2.boundingRect(pts_src)
    r2 = cv2.boundingRect(pts_dst)

    # ROI
    src_roi = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    dst_roi = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    # 轉換到 ROI 座標
    t1 = pts_src - np.array([r1[0], r1[1]], dtype=np.float32)
    t2 = pts_dst - np.array([r2[0], r2[1]], dtype=np.float32)

    # 仿射
    M = cv2.getAffineTransform(t1, t2)
    warped = cv2.warpAffine(
        src_roi, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101   # 關鍵：避免黑邊
    )

    # 建三角形遮罩
    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t2), 255)
    mask = cv2.GaussianBlur(mask, (3,3), 0)

    # 貼回（局部 alpha）
    mask_f = (mask.astype(np.float32)/255.0)[..., None]
    dst_roi[:] = (warped*mask_f + dst_roi*(1.0 - mask_f)).astype(np.uint8)

# ----------------------------
# Gamma-aware alpha blending
# ----------------------------
def _srgb_to_linear(x):
    # x: float32 in [0,1]
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

# ----------------------------
# Gamma-aware alpha blending
# ----------------------------
def _linear_to_srgb(x):
    # x: float32 in [0,1]
    a = 0.055
    return np.where(x <= 0.0031308, 12.92*x, (1+a)*(x**(1/2.4)) - a)

# ----------------------------
# Gamma-aware alpha blending
# ----------------------------
def gamma_aware_blend(fg_bgr_u8, bg_bgr_u8, mask_u8, alpha=0.65):

    # 在「線性光空間」做 alpha 混合，最後轉回 sRGB。
    # fg_bgr_u8: 前景（warped_source），uint8 HxWx3
    # bg_bgr_u8: 背景（output_img），uint8 HxWx3
    # mask_u8  : 單通道 0~255 的遮罩（與圖同尺寸）
    # alpha    : 0~1 的融合係數

    # 保證連續 & dtype
    fg = np.ascontiguousarray(fg_bgr_u8).astype(np.float32) / 255.0
    bg = np.ascontiguousarray(bg_bgr_u8).astype(np.float32) / 255.0
    m  = (np.ascontiguousarray(mask_u8).astype(np.float32) / 255.0)[..., None]  # HxWx1

    # 有效權重（把全域 alpha 乘到遮罩）
    w = np.clip(m * float(alpha), 0.0, 1.0)

    # sRGB -> 線性光
    fg_lin = _srgb_to_linear(fg)
    bg_lin = _srgb_to_linear(bg)

    # 在線性光空間混合
    out_lin = fg_lin * w + bg_lin * (1.0 - w)

    # 線性光 -> sRGB
    out = _linear_to_srgb(out_lin)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

# ----------------------------
# 色彩貼合：Lab 空間 a/b 強匹配，L 小幅靠攏
# ----------------------------
def color_match_lab_ab(src_bgr, tgt_bgr, mask_u8=None, ab_ratio=0.9, l_ratio=0.3):

    # 將 src_bgr（換過來的臉）貼合 tgt_bgr（被換的人）的色彩。
    # 強匹配 a/b（色調與飽和）；L 亮度小幅靠攏。
    # mask_u8 可選：若提供，只在 mask>0 的區域統計並更新；否則用 src 非黑像素。

    src = src_bgr
    tgt = tgt_bgr

    # 建 boolean 遮罩 m，形狀必須是 (H, W)
    if mask_u8 is not None:
        m = (mask_u8 > 0)
    else:
        m = (src.sum(axis=2) > 0)
    if m.sum() < 100:
        return src

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src_lab.copy()

    eps = 1e-6

    # 把 a/b/L 的均值/標準差組成向量做 EMA（幀間穩定）
    # 會以函式屬性保存上一幀統計
    if not hasattr(color_match_lab_ab, "_stats"):          # <<<
        color_match_lab_ab._stats = None                   # <<<

    # ---- a、b 通道：整張算 matched_full，僅在 m 區更新 ----
    for c in (1, 2):
        s_mean = float(src_lab[..., c][m].mean())
        s_std  = float(src_lab[..., c][m].std() + eps)
        t_mean = float(tgt_lab[..., c][m].mean())
        t_std  = float(tgt_lab[..., c][m].std() + eps)

        matched_full = (src_lab[..., c] - s_mean) * (t_std / s_std) + t_mean  # (H,W)
        out_ch = out[..., c]                                                  # (H,W)
        out_ch[m] = matched_full[m] * ab_ratio + out_ch[m] * (1.0 - ab_ratio)
        out[..., c] = out_ch

    # ---- L 通道：小幅靠攏 ----
    s_mean = float(src_lab[..., 0][m].mean())
    s_std  = float(src_lab[..., 0][m].std() + eps)
    t_mean = float(tgt_lab[..., 0][m].mean())
    t_std  = float(tgt_lab[..., 0][m].std() + eps)

    matchedL_full = (src_lab[..., 0] - s_mean) * (t_std / s_std) + t_mean
    out_L = out[..., 0]
    out_L[m] = out_L[m] * (1.0 - l_ratio) + matchedL_full[m] * l_ratio
    out[..., 0] = out_L

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

# 你原本的 warp_triangle() 要存在
# def warp_triangle(src_img, dst_img, pts_src, pts_dst): ...

# ----------------------------
# 建立智慧遮罩
# ----------------------------
def build_smart_mask(target_landmarks, img_shape, extra_points=None, feather=15):

    # 根據臉部五官自動建立平滑遮罩。
    # target_landmarks : (N,2)
    # img_shape        : (H,W,3)
    # extra_points     : 額外點（如額頭擴展點）
    # feather          : 羽化程度（像素）

    h, w = img_shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # --- 基本臉區：下巴 + 雙頰 + 鼻翼 + 嘴部 ---
    jaw = target_landmarks[0:17]
    nose = target_landmarks[27:36]
    mouth = target_landmarks[48:60]
    cheeks = np.vstack([jaw[3:14], nose[4:5], mouth[3:10]])

    cv2.fillConvexPoly(mask, cv2.convexHull(cheeks.astype(np.int32)), 255)

    # --- 額頭區域（若提供額外點） ---
    if extra_points is not None and len(extra_points) > 0:
        top = np.vstack([target_landmarks[17:27], extra_points])
        cv2.fillConvexPoly(mask, cv2.convexHull(top.astype(np.int32)), 255)

    # --- 眼部周圍（輕微減弱，避免閃爍） ---
    left_eye = target_landmarks[36:42]
    right_eye = target_landmarks[42:48]
    cv2.fillConvexPoly(mask, cv2.convexHull(left_eye.astype(np.int32)), 180)
    cv2.fillConvexPoly(mask, cv2.convexHull(right_eye.astype(np.int32)), 180)

    # --- 羽化邊緣 ---
    mask = cv2.GaussianBlur(mask, (feather | 1, feather | 1), 0)
    mask = (mask * (255.0 / mask.max())).clip(0, 255).astype(np.uint8)
    return mask

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
        # === 25 額頭點：把水平線改成弧線 ===
        layers = [
            (-0.12 * h, 7),   # 上層
            (-0.20 * h, 9),   # 中層
            (-0.27 * h, 9),   # 下層（接近髮際）
        ]
        extra_points_list = []
        for y_off, n_pts in layers:
            x_start = top_x - 0.35 * w
            x_end   = top_x + 0.35 * w
            xs = np.linspace(x_start, x_end, int(n_pts)).astype(np.float32)

            # 做一個左右略高、中間略低的弧形（cos 曲線）
            # amp 可微調弧度：下層弧度大、上層小
            rel = (xs - top_x) / (0.35 * w)  # -1..1
            # 針對不同層給不同弧度
            if n_pts == 9 and abs(y_off + 0.27*h) < 1e-3:   # 下層
                amp = 0.06 * h
            elif n_pts == 9:                                 # 中層
                amp = 0.045 * h
            else:                                            # 上層
                amp = 0.03 * h

            curve = amp * (1 - np.cos(np.pi * (1 - np.abs(rel))))  # 兩側↑ 中間↓
            ys = (top_y + y_off - curve).astype(np.float32)

            extra_points_list.append(np.stack([xs, ys], axis=1))

        extra_points = np.concatenate(extra_points_list, axis=0).astype(np.float32)  # (25,2)

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
        warped_source = target_img.copy()   # ★ 改這裡：不要用 zeros_like

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

        # （選）GFPGAN：放在色彩對齊之前；此處不要動用 mask
        if gfpgan is not None:
            try:
                _, restored_img, _ = gfpgan.enhance(
                    np.ascontiguousarray(warped_source),
                    has_aligned=False, only_center_face=False, paste_back=True
                )
                if isinstance(restored_img, np.ndarray) and restored_img.size > 0:
                    warped_source = restored_img
            except Exception as e:
                print(f"GFPGAN enhance failed, fallback to original warp: {e}")



        # 原本你是這樣:
        # mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
        # cv2.fillConvexPoly(mask, cv2.convexHull(expanded_landmarks.astype(np.int32)), 255)
        # mask = cv2.GaussianBlur(mask, (13, 13), 0)

        # 改成這樣：
        mask = build_smart_mask(
            target_landmarks=target_landmarks,
            img_shape=target_img.shape,
            extra_points=extra_points,  # 你的額頭25點
            feather=17                  # 越高越柔順，建議 15~21
        )
        # 先擴一點再柔和，避免硬邊
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), 1)
        mask = cv2.GaussianBlur(mask, (11,11), 0)


        # 例：jaw_idx = [0..16]（請換成你模型的下頜線索引）
        jaw_idx = list(range(0,17))
        chin_pts = target_landmarks[jaw_idx]                      # 下頜線
        # 往下投射到頸部：向下平移 0.06*h（可調 0.05~0.09）
        neck_pts = chin_pts + np.array([0, 0.06*h], dtype=np.float32)
        # 組成「下巴到頸部」封閉多邊形
        poly = np.vstack([chin_pts, neck_pts[::-1]])
        mask_edge = np.zeros(target_img.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask_edge, poly.astype(np.int32), 255)
        # 與你的主遮罩 union 後再羽化
        mask = cv2.bitwise_or(mask, mask_edge)
        mask = cv2.GaussianBlur(mask, (13,13), 0)


        # 與 warped_source 有效區域取交集，避免黑洞
        valid = (warped_source.sum(axis=2) > 0).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, valid)


        # --- 色彩貼合：讓臉顏色貼合被換的人（背景/目標） ---
        valid_for_stats = (warped_source.sum(axis=2) > 0).astype(np.uint8) * 255
        warped_source = np.ascontiguousarray(warped_source)
        output_img    = np.ascontiguousarray(output_img)

        # 調整貼合力度：ab_ratio 越高越像對方、l_ratio 保持小
        warped_source = color_match_lab_ab(
            warped_source, output_img, valid_for_stats,
            ab_ratio=0.92,   # 0.85~0.95，偏暖/偏飽和就拉高
            l_ratio=0.25     # 0.2~0.35，別太大免得臉變平
        )


        # Alpha 融合
        output_img = gamma_aware_blend(warped_source, output_img, mask, alpha)


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
    setup_cuda_fast()
    global app
    app = init_insightface(det_size=(640, 640))
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