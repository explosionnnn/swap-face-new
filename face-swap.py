import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
from numpy.linalg import norm
import torch
import os
import subprocess
import tempfile
import shutil


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
    try:
        torch.set_float32_matmul_precision('high')  # 'medium' 也行
    except Exception:
        pass

    # 4) 預先建立一個專用 Stream（可選，避免與預設串流搶資源）
    global _fast_stream
    try:
        _fast_stream = torch.cuda.Stream()
    except Exception:
        _fast_stream = None
    return

# ----------------------------
# 初始化 InsightFace
# ----------------------------
def init_insightface(det_size=(640, 640)):
    """
    初始化 InsightFace，自動使用 GPU (若可用) 或 CPU fallback。
    """
    use_gpu = torch.cuda.is_available()
    provider = 'CUDAExecutionProvider' if use_gpu else 'CPUExecutionProvider'
    ctx_id = 0 if use_gpu else -1

    print(f"Initializing InsightFace on {'GPU' if use_gpu else 'CPU'}...")
    app = FaceAnalysis(providers=[provider])
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    print("InsightFace loaded.")
    return app

# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Face Swap using InsightFace + GFPGAN + Reference Face")
    parser.add_argument('--source', type=str, required=True, help='來源臉圖檔 (example: C:/path/source.png)')
    parser.add_argument('--input', type=str, required=True, help='待處理影片 (帶原始音訊)')
    parser.add_argument('--output', type=str, required=True, help='輸出最終影片（含音訊）')
    parser.add_argument('--reference', type=str, required=True, help='Reference 圖片（用於 embedding 比對）')
    parser.add_argument('--gfpgan', action='store_true', help='啟用 GFPGAN 影像增強')
    parser.add_argument('--threshold', type=float, default=0.3, help='embedding cosine 相似度門檻 (default: 0.3)')
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
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2)))

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
# 安全版三角形 warp（避免黑邊）
# ----------------------------
def warp_triangle(src_img, dst_img, pts_src, pts_dst):
    pts_src = np.float32(pts_src)
    pts_dst = np.float32(pts_dst)

    r1 = cv2.boundingRect(pts_src)
    r2 = cv2.boundingRect(pts_dst)

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    src_roi = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    dst_roi = dst_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    if src_roi.size == 0 or dst_roi.size == 0:
        return

    t1 = pts_src - np.array([r1[0], r1[1]], dtype=np.float32)
    t2 = pts_dst - np.array([r2[0], r2[1]], dtype=np.float32)

    M = cv2.getAffineTransform(t1, t2)
    warped = cv2.warpAffine(
        src_roi, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t2), 255)
    mask = cv2.GaussianBlur(mask, (3,3), 0)

    mask_f = (mask.astype(np.float32)/255.0)[..., None]
    dst_roi[:] = (warped*mask_f + dst_roi*(1.0 - mask_f)).astype(np.uint8)

# ----------------------------
# sRGB <-> 線性光轉換 & gamma-aware blend
# ----------------------------
def _srgb_to_linear(x):
    
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)


def _linear_to_srgb(x):
    
    a = 0.055
    return np.where(x <= 0.0031308, 12.92*x, (1+a)*(x**(1/2.4)) - a)


def gamma_aware_blend(fg_bgr_u8, bg_bgr_u8, mask_u8, alpha=0.65):
    
    fg = np.ascontiguousarray(fg_bgr_u8).astype(np.float32) / 255.0
    bg = np.ascontiguousarray(bg_bgr_u8).astype(np.float32) / 255.0
    m  = (np.ascontiguousarray(mask_u8).astype(np.float32) / 255.0)[..., None]

    
    w = np.clip(m * float(alpha), 0.0, 1.0)
    
    fg_lin = _srgb_to_linear(fg)
    bg_lin = _srgb_to_linear(bg)
    
    out_lin = fg_lin * w + bg_lin * (1.0 - w)
    
    out = _linear_to_srgb(out_lin)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

# ----------------------------
# 色彩貼合 Lab a/b 強匹配
# ----------------------------
def color_match_lab_ab(src_bgr, tgt_bgr, mask_u8=None, ab_ratio=0.9, l_ratio=0.3):
    
    src = src_bgr
    tgt = tgt_bgr


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
    if not hasattr(color_match_lab_ab, "_stats"):
        color_match_lab_ab._stats = None

    for c in (1, 2):
        s_mean = float(src_lab[..., c][m].mean())
        s_std  = float(src_lab[..., c][m].std() + eps)
        t_mean = float(tgt_lab[..., c][m].mean())
        t_std  = float(tgt_lab[..., c][m].std() + eps)

        matched_full = (src_lab[..., c] - s_mean) * (t_std / s_std) + t_mean
        out_ch = out[..., c]
        out_ch[m] = matched_full[m] * ab_ratio + out_ch[m] * (1.0 - ab_ratio)
        out[..., c] = out_ch


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



# ----------------------------
# 建立智慧遮罩
# ----------------------------
def build_smart_mask(target_landmarks, img_shape, extra_points=None, feather=15):
    
    h, w = img_shape[:2]
    mask = np.zeros((h, w), np.uint8)


    jaw = target_landmarks[0:17]
    nose = target_landmarks[27:36]
    mouth = target_landmarks[48:60]
    cheeks = np.vstack([jaw[3:14], nose[4:5], mouth[3:10]])

    cv2.fillConvexPoly(mask, cv2.convexHull(cheeks.astype(np.int32)), 255)


    if extra_points is not None and len(extra_points) > 0:
        top = np.vstack([target_landmarks[17:27], extra_points])
        cv2.fillConvexPoly(mask, cv2.convexHull(top.astype(np.int32)), 255)


    left_eye = target_landmarks[36:42]
    right_eye = target_landmarks[42:48]
    cv2.fillConvexPoly(mask, cv2.convexHull(left_eye.astype(np.int32)), 180)
    cv2.fillConvexPoly(mask, cv2.convexHull(right_eye.astype(np.int32)), 180)


    mask = cv2.GaussianBlur(mask, (feather | 1, feather | 1), 0)
    if mask.max() > 0:
        mask = (mask * (255.0 / mask.max())).clip(0, 255).astype(np.uint8)
    return mask

# ----------------------------
# 換臉主函式
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

        layers = [
            (-0.05 * h, 7),
            (-0.10 * h, 9),
            (-0.15 * h, 9),
        ]
        extra_points_list = []
        for y_off, n_pts in layers:
            x_start = top_x - 0.35 * w
            x_end   = top_x + 0.35 * w
            xs = np.linspace(x_start, x_end, int(n_pts)).astype(np.float32)

            rel = (xs - top_x) / (0.35 * w)
            if n_pts == 9 and abs(y_off + 0.27*h) < 1e-3:
                amp = 0.06 * h
            elif n_pts == 9:
                amp = 0.045 * h
            else:
                amp = 0.03 * h

            curve = amp * (1 - np.cos(np.pi * (1 - np.abs(rel))))
            ys = (top_y + y_off - curve).astype(np.float32)
            extra_points_list.append(np.stack([xs, ys], axis=1))

        extra_points = np.concatenate(extra_points_list, axis=0).astype(np.float32)
        all_target = np.vstack([target_landmarks, extra_points])

        tl = target_landmarks.astype(np.float32)
        sl = source_landmarks.astype(np.float32)
        y_med = np.median(tl[:, 1])
        upper_mask = tl[:, 1] < y_med

        tl_fit = tl[upper_mask] if np.sum(upper_mask) >= 10 else tl
        sl_fit = sl[upper_mask] if np.sum(upper_mask) >= 10 else sl

        M, _ = cv2.estimateAffinePartial2D(
            tl_fit, sl_fit,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000, confidence=0.99
        )

        def apply_affine(pts, M):
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
            out = pts_h @ M.T
            return out.astype(np.float32)

        if M is not None:
            extra_src = apply_affine(extra_points, M)
        else:
            src_center = np.mean(source_landmarks, axis=0)
            tgt_center = np.mean(target_landmarks, axis=0)
            extra_src = src_center + (extra_points - tgt_center)

        all_source = np.vstack([source_landmarks.astype(np.float32), extra_src])

        rect = cv2.boundingRect(all_target.astype(np.float32))
        subdiv = cv2.Subdiv2D(rect)
        for p in all_target:
            subdiv.insert((float(p[0]), float(p[1])))

        triangles = subdiv.getTriangleList().reshape(-1, 6)

        warped_source = target_img.copy()

        for tri in triangles:
            p1 = np.array([tri[0], tri[1]], dtype=np.float32)
            p2 = np.array([tri[2], tri[3]], dtype=np.float32)
            p3 = np.array([tri[4], tri[5]], dtype=np.float32)

            idx1 = np.argmin(np.linalg.norm(all_target - p1, axis=1))
            idx2 = np.argmin(np.linalg.norm(all_target - p2, axis=1))
            idx3 = np.argmin(np.linalg.norm(all_target - p3, axis=1))

            if len({idx1, idx2, idx3}) < 3:
                continue

            pts_tgt = np.float32([all_target[idx1], all_target[idx2], all_target[idx3]])
            pts_src = np.float32([all_source[idx1], all_source[idx2], all_source[idx3]])

            warp_triangle(source_img, warped_source, pts_src, pts_tgt)

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

        mask = build_smart_mask(
            target_landmarks=target_landmarks,
            img_shape=target_img.shape,
            extra_points=extra_points,
            feather=17
        )
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), 1)
        mask = cv2.GaussianBlur(mask, (11,11), 0)

        jaw_idx = list(range(0,17))
        chin_pts = target_landmarks[jaw_idx]
        neck_pts = chin_pts + np.array([0, 0.06*h], dtype=np.float32)
        poly = np.vstack([chin_pts, neck_pts[::-1]])
        mask_edge = np.zeros(target_img.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask_edge, poly.astype(np.int32), 255)
        mask = cv2.bitwise_or(mask, mask_edge)
        mask = cv2.GaussianBlur(mask, (13,13), 0)

        valid = (warped_source.sum(axis=2) > 0).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, valid)

        valid_for_stats = valid
        warped_source = np.ascontiguousarray(warped_source)
        output_img    = np.ascontiguousarray(output_img)

        warped_source = color_match_lab_ab(
            warped_source, output_img, valid_for_stats,
            ab_ratio=0.92,
            l_ratio=0.25
        )

        output_img = gamma_aware_blend(warped_source, output_img, mask, alpha)

    return output_img

# ----------------------------
# 將影片處理結果輸出為無音訊的暫存 mp4，回傳暫存路徑
# ----------------------------
def process_video(source_img, input_path, out_path_no_audio, ref_embedding, gfpgan=None, threshold=0.3):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out = cv2.VideoWriter(out_path_no_audio, fourcc, fps, (width, height))

    frame_idx = 0
    try:
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

            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0.0
            bar_length = 30
            filled_length = int(bar_length * frame_idx // total_frames) if total_frames > 0 else 0
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\rProgress: |{bar}| {progress:6.2f}% ({frame_idx}/{total_frames})", end='')

        cap.release()
        out.release()
        print("\nVideo processing finished.")
        return out_path_no_audio
    except Exception:
        cap.release()
        out.release()
        raise

# ----------------------------
# 使用 ffmpeg 將原始影片的音訊合併回來
# ----------------------------
def merge_audio_with_ffmpeg(video_no_audio_path, source_with_audio_path, final_output_path, prefer_copy=True):
    ffmpeg_cmd_exists = shutil.which("ffmpeg") is not None
    if not ffmpeg_cmd_exists:
        raise RuntimeError("ffmpeg not found in PATH — please install ffmpeg and ensure it's on your PATH.")

    if prefer_copy:
        cmd_copy = [
            "ffmpeg", "-y",
            "-i", video_no_audio_path,
            "-i", source_with_audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            final_output_path
        ]
        try:
            subprocess.run(cmd_copy, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            pass

    cmd_transcode = [
        "ffmpeg", "-y",
        "-i", video_no_audio_path,
        "-i", source_with_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_output_path
    ]
    try:
        subprocess.run(cmd_transcode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8') if isinstance(e.stderr, (bytes, bytearray)) else str(e)
        raise RuntimeError(f"ffmpeg failed to merge audio: {stderr}")

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

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    tmp_path2 = None
    try:
        print(f"Writing face-swapped (no audio) to temporary file: {tmp_path}")
        result = process_video(source_img, args.input, tmp_path, ref_embedding, gfpgan=gfpgan_model, threshold=args.threshold)
        if not result:
            print("Re-running video without GFPGAN (temporary file)...")
            tmp_fd2, tmp_path2 = tempfile.mkstemp(suffix=".mp4")
            os.close(tmp_fd2)
            result2 = process_video(source_img, args.input, tmp_path2, ref_embedding, gfpgan=None, threshold=args.threshold)
            if not result2:
                raise RuntimeError("Both GFPGAN and CPU-only runs failed.")
            # 刪除原先 tmp_path 並改為 tmp_path2
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            tmp_path = tmp_path2

        # tmp_path 現在是沒有音訊的換臉影片
        print("Merging audio from original input into swapped video using ffmpeg...")
        try:
            # 輸出到一個暫存最終檔，避免覆蓋時出問題
            final_tmp_fd, final_tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(final_tmp_fd)
            try:
                merge_audio_with_ffmpeg(tmp_path, args.input, final_tmp_path, prefer_copy=True)
                # 成功 -> 移動到使用者指定輸出
                shutil.move(final_tmp_path, args.output)
                print(f"Final output saved to: {args.output}")
            finally:
                if os.path.exists(final_tmp_path):
                    try:
                        os.remove(final_tmp_path)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Audio merge failed: {e}")
            # 若合併失敗，保留無音訊檔案（或搬到輸出）
            try:
                shutil.move(tmp_path, args.output)
                print(f"No-audio swapped video moved to: {args.output}")
            except Exception as e2:
                raise RuntimeError(f"Failed to merge audio and failed moving tmp file: {e2}") from e
    finally:
        # 清理暫存（若還存在）
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            if tmp_path2 and os.path.exists(tmp_path2):
                os.remove(tmp_path2)
        except Exception:
            pass

if __name__ == "__main__":
    main()
