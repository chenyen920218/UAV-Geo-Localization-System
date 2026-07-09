import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import json
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import mercantile
import cv2
from kornia.feature import LoFTR
import math
from tqdm import tqdm

# =====================================================================
# 1. 參數與輔助函數
# =====================================================================
DB_INDEX_PATH = "satellite_gallery.index"
DB_META_PATH = "satellite_meta.json"

# 衛星圖庫 (Gallery) 與 無人機測試圖 (Test)
GALLERY_DIR = "satellite_gallery_latest"
TEST_DIR = "competition_gallery_true" 
# 🚀 儲存所有匹配結果圖的資料夾
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_ground_truth_from_filename(image_path):
    filename = os.path.basename(image_path).replace(".jpg", "")
    parts = filename.split("_")
    bounds = mercantile.bounds(int(parts[1]), int(parts[2]), int(parts[0]))
    return (bounds.south + bounds.north) / 2.0, (bounds.west + bounds.east) / 2.0

def pixel_to_latlon(x_pixel, y_pixel, img_w, img_h, bounds):
    lon = bounds.west + (x_pixel / img_w) * (bounds.east - bounds.west)
    lat = bounds.north - (y_pixel / img_h) * (bounds.north - bounds.south)
    return lat, lon

def calculate_error(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# =====================================================================
# 2. 批次評估主程式
# =====================================================================
def run_batch_evaluation():
    print("【系統啟動】載入 EigenPlaces 模型與 LoFTR...")
    global_model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048).to(device).eval()
    local_matcher = LoFTR(pretrained='outdoor').to(device).eval()
    
    faiss_index = faiss.read_index(DB_INDEX_PATH)
    with open(DB_META_PATH, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_images = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
    if not test_images:
        print(f"【錯誤】在 {TEST_DIR} 中找不到測試圖片！")
        return

    print(f"【開始測試】總計 {len(test_images)} 張無人機測試圖，準備起飛...\n")

    error_list = []
    fallback_count = 0

    for img_path in tqdm(test_images, desc="定位進度"):
        try:
            base_filename = os.path.basename(img_path)
            
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = data_transforms(img_pil).unsqueeze(0).to(device)
            
            q_cv = cv2.resize(cv2.imread(img_path, 0), (256, 256))
            q_color = cv2.cvtColor(q_cv, cv2.COLOR_GRAY2BGR) 
            q_tensor = torch.from_numpy(q_cv).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

            # --- 階段一：全局檢索 (抓取 Top 3) ---
            with torch.no_grad():
                feat = global_model(img_tensor)
                feat = feat / torch.norm(feat, p=2, dim=1, keepdim=True)
            
            D, I = faiss_index.search(feat.cpu().numpy().astype('float32'), 3)
            
            best_inliers_count = -1
            best_lat, best_lon = None, None
            top1_bounds = None 
            
            best_s_color = None
            best_pts_q = None
            best_pts_s = None

            # --- 階段二：Top-3 逐一進行 LoFTR 驗證 ---
            for rank, idx in enumerate(I[0]):
                matched = metadata_list[idx]
                matched_path = os.path.join(GALLERY_DIR, matched['filename'])
                bounds = mercantile.bounds(matched['x'], matched['y'], matched['z'])
                
                if rank == 0:
                    top1_bounds = bounds

                s_cv = cv2.resize(cv2.imread(matched_path, 0), (256, 256))
                s_color = cv2.cvtColor(s_cv, cv2.COLOR_GRAY2BGR) 
                s_tensor = torch.from_numpy(s_cv).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
                
                input_dict = {"image0": q_tensor, "image1": s_tensor}
                
                with torch.no_grad():
                    corrs = local_matcher(input_dict)
                pts_q, pts_s = corrs['keypoints0'].cpu().numpy(), corrs['keypoints1'].cpu().numpy()

                # --- 階段三：嚴格座標解算 ---
                if len(pts_q) >= 15:
                    H, inliers = cv2.findHomography(pts_q, pts_s, cv2.USAC_MAGSAC, 5.0)
                    
                    if inliers is not None:
                        current_inliers = np.sum(inliers)
                        if current_inliers >= 10:
                            px, py = cv2.perspectiveTransform(np.array([[[128.0, 128.0]]]), H)[0][0]
                            
                            if -50 <= px <= 300 and -50 <= py <= 300:
                                if current_inliers > best_inliers_count:
                                    best_inliers_count = current_inliers
                                    best_lat, best_lon = pixel_to_latlon(px, py, 256, 256, bounds)
                                    best_s_color = s_color
                                    best_pts_q = pts_q
                                    best_pts_s = pts_s
            
            if best_s_color is None:
                matched_path = os.path.join(GALLERY_DIR, metadata_list[I[0][0]]['filename'])
                s_cv = cv2.resize(cv2.imread(matched_path, 0), (256, 256))
                best_s_color = cv2.cvtColor(s_cv, cv2.COLOR_GRAY2BGR)
                best_pts_q, best_pts_s = [], [] 

            # --- 決定最終輸出 ---
            is_fallback = False
            if best_lat is not None and best_lon is not None:
                ex_lat, ex_lon = best_lat, best_lon
            else:
                fallback_count += 1
                is_fallback = True
                ex_lat, ex_lon = (top1_bounds.south + top1_bounds.north)/2.0, (top1_bounds.west + top1_bounds.east)/2.0

            # --- 階段四：計算真實誤差 ---
            gt_lat, gt_lon = get_ground_truth_from_filename(img_path)
            err = calculate_error(ex_lat, ex_lon, gt_lat, gt_lon)
            error_list.append(err)

            # --- 🚀 階段五：繪製並儲存匹配圖 ---
            img_draw = np.hstack((q_color, best_s_color))
            
            if not is_fallback and len(best_pts_q) > 0:
                for i in range(len(best_pts_q)):
                    pt1 = (int(best_pts_q[i][0]), int(best_pts_q[i][1]))
                    pt2 = (int(best_pts_s[i][0]) + 256, int(best_pts_s[i][1]))
                    cv2.line(img_draw, pt1, pt2, (0, 255, 0), 1)
                    cv2.circle(img_draw, pt1, 2, (0, 0, 255), -1)
                    cv2.circle(img_draw, pt2, 2, (0, 0, 255), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # 1. 左上角：印出誤差與有效點數量
            status_text = f"Error: {err:.1f}m"
            if is_fallback:
                status_text += " (FALLBACK)"
            color = (0, 0, 255) if err > 50 else ((0, 165, 255) if err > 5 else (0, 255, 0))
            cv2.putText(img_draw, status_text, (10, 30), font, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(img_draw, f"Inliers: {best_inliers_count if best_inliers_count > 0 else 0}", (10, 55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 2. 🚀 新增：左邊下方印出 真實經緯度 (Ground Truth)
            gt_text = f"GT: {gt_lat:.5f}, {gt_lon:.5f}"
            # 先畫黑色粗底當陰影，再畫白色細字，確保在亮/暗色背景都清晰可見
            cv2.putText(img_draw, gt_text, (10, 245), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA) 
            cv2.putText(img_draw, gt_text, (10, 245), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. 🚀 新增：右邊下方印出 AI 預測經緯度 (AI Prediction)
            # 右半邊的圖片 X 座標起始於 256，所以設定 266 (稍微留白)
            pred_text = f"AI: {ex_lat:.5f}, {ex_lon:.5f}"
            cv2.putText(img_draw, pred_text, (266, 245), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img_draw, pred_text, (266, 245), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA) # 黃色字體以示區分

            # 儲存圖片
            save_filename = f"{base_filename}"
            cv2.imwrite(os.path.join(RESULTS_DIR, save_filename), img_draw)

        except Exception as e:
            print(f"\n[跳過] 影像 {os.path.basename(img_path)} 處理失敗: {e}")

    # =====================================================================
    # 3. 統計與戰報輸出
    # =====================================================================
    if error_list:
        errors = np.array(error_list)
        avg_err = np.mean(errors)
        median_err = np.median(errors)
        max_err = np.max(errors)
        min_err = np.min(errors)
        
        acc_5m = np.sum(errors <= 5.0) / len(errors) * 100
        acc_15m = np.sum(errors <= 15.0) / len(errors) * 100
        acc_50m = np.sum(errors <= 50.0) / len(errors) * 100

        print("\n" + "="*50)
        print(" 🏆 【無人機高精度定位系統 - 批次測試戰報】 🏆")
        print("="*50)
        print(f" 📊 測試總數: {len(errors)} 張影像")
        print(f" 📉 平均誤差 (Mean):   {avg_err:.2f} 公尺")
        print(f" 🎯 中位數誤差 (Median): {median_err:.2f} 公尺")
        print(f" ✨ 最小誤差 (Min):    {min_err:.2f} 公尺")
        print(f" ⚠️ 最大誤差 (Max):    {max_err:.2f} 公尺")
        print("-" * 50)
        print(f" 🛡️ 觸發降級保護次數: {fallback_count} 次 (退回中心點)")
        print("-" * 50)
        print(" 🎯 系統達標率:")
        print(f"    ✔️ 誤差 <=  5 公尺 (極高精度): {acc_5m:.1f}%")
        print(f"    ✔️ 誤差 <= 15 公尺 (實戰可用):   {acc_15m:.1f}%")
        print(f"    ✔️ 誤差 <= 50 公尺 (大致鎖定):   {acc_50m:.1f}%")
        print("="*50)
        print(f" 📸 所有匹配結果圖片已儲存至: {RESULTS_DIR}/ 資料夾")

if __name__ == "__main__":
    run_batch_evaluation()