import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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

# =====================================================================
# 1. 參數與路徑配置
# =====================================================================
DB_INDEX_PATH = "satellite_gallery.index"
DB_META_PATH = "satellite_meta.json"
GALLERY_DIR = "satellite_gallery_latest" # 確保對應你的最新圖庫

# 測試輸入的無人機畫面
QUERY_IMAGE_PATH = "competition_gallery_true_18/18_218655_113499.jpg"

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
# 2. 主程式：無人機即時定位流水線 (含 Top-3 與嚴格過濾)
# =====================================================================
def run_drone_localization():
    print("【系統啟動】載入真正的 EigenPlaces 模型與 FAISS...")
    global_model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048).to(device).eval()
    local_matcher = LoFTR(pretrained='outdoor').to(device).eval()
    
    faiss_index = faiss.read_index(DB_INDEX_PATH)
    with open(DB_META_PATH, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    if not os.path.exists(QUERY_IMAGE_PATH):
        print(f"【錯誤】找不到測試圖片 {QUERY_IMAGE_PATH}")
        return

    # --- 影像預處理 ---
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = data_transforms(Image.open(QUERY_IMAGE_PATH).convert('RGB')).unsqueeze(0).to(device)
    q_cv = cv2.resize(cv2.imread(QUERY_IMAGE_PATH, 0), (256, 256))
    q_tensor = torch.from_numpy(q_cv).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

    print("\n==================================================")
    print(" 🚀 階段一：全局檢索 (Top-3 防邊界死角)")
    print("==================================================")
    
    with torch.no_grad():
        feat = global_model(img_tensor)
        feat = feat / torch.norm(feat, p=2, dim=1, keepdim=True)
    
    # 找尋前 3 名最像的衛星圖
    D, I = faiss_index.search(feat.cpu().numpy().astype('float32'), 3)
    
    best_inliers_count = -1
    best_lat, best_lon = None, None
    best_img_draw = None
    best_matched_filename = None
    top1_bounds = None

    print("\n==================================================")
    print(" 🎯 階段二：LoFTR 局部特徵精確匹配與防呆過濾")
    print("==================================================")

    # 針對前三名逐一驗證，選出「特徵點最真實」的那一張
    for rank, idx in enumerate(I[0]):
        matched = metadata_list[idx]
        matched_path = os.path.join(GALLERY_DIR, matched['filename'])
        bounds = mercantile.bounds(matched['x'], matched['y'], matched['z'])
        
        if rank == 0:
            top1_bounds = bounds

        s_cv = cv2.resize(cv2.imread(matched_path, 0), (256, 256))
        s_tensor = torch.from_numpy(s_cv).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        
        with torch.no_grad():
            corrs = local_matcher({"image0": q_tensor, "image1": s_tensor})
        pts_q, pts_s = corrs['keypoints0'].cpu().numpy(), corrs['keypoints1'].cpu().numpy()

        # 繪製匹配圖 (這一步保證一定會畫圖)
        query_color = cv2.cvtColor(q_cv, cv2.COLOR_GRAY2BGR)
        sat_color = cv2.cvtColor(s_cv, cv2.COLOR_GRAY2BGR)
        img_draw = np.hstack((query_color, sat_color))
        for i in range(len(pts_q)):
            pt1 = (int(pts_q[i][0]), int(pts_q[i][1]))
            pt2 = (int(pts_s[i][0]) + 256, int(pts_s[i][1]))
            cv2.line(img_draw, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(img_draw, pt1, 2, (0, 0, 255), -1) # 補回紅點
            cv2.circle(img_draw, pt2, 2, (0, 0, 255), -1) # 補回紅點

        # 嚴格過濾機制
        if len(pts_q) >= 15:
            H, inliers = cv2.findHomography(pts_q, pts_s, cv2.USAC_MAGSAC, 5.0)
            if inliers is not None:
                current_inliers = np.sum(inliers)
                if current_inliers >= 10:
                    px, py = cv2.perspectiveTransform(np.array([[[128.0, 128.0]]]), H)[0][0]
                    # 邊界防呆
                    if -50 <= px <= 300 and -50 <= py <= 300:
                        if current_inliers > best_inliers_count:
                            best_inliers_count = current_inliers
                            best_lat, best_lon = pixel_to_latlon(px, py, 256, 256, bounds)
                            best_img_draw = img_draw
                            best_matched_filename = matched['filename']
        
        # 降級畫圖預備：如果三張圖都沒過關，至少要把第一名的圖保留下來輸出
        if rank == 0 and best_img_draw is None:
            best_img_draw = img_draw
            best_matched_filename = matched['filename']

    # --- 輸出除錯圖 ---
    if best_img_draw is not None:
        cv2.imwrite("loftr_matches_debug.jpg", best_img_draw)
        print(f"📸 成功產出視覺化偵錯圖: loftr_matches_debug.jpg (鎖定瓦片: {best_matched_filename})")

    # --- 決定最終輸出 ---
    if best_lat is not None and best_lon is not None:
        print(f"✅ 成功鎖定高精度座標！(採用局內點數量: {best_inliers_count})")
        ex_lat, ex_lon = best_lat, best_lon
    else:
        print("【警告】防護機制觸發：特徵過少或疑似農田誤判，回退至最佳瓦片中心點。")
        ex_lat, ex_lon = (top1_bounds.south + top1_bounds.north)/2.0, (top1_bounds.west + top1_bounds.east)/2.0

    gt_lat, gt_lon = get_ground_truth_from_filename(QUERY_IMAGE_PATH)
    err = calculate_error(ex_lat, ex_lon, gt_lat, gt_lon)
    
    print("\n==================================================")
    print(" 🏁 【最終高精度解算結果】")
    print(f"   AI 定位座標:  {ex_lat:.6f}, {ex_lon:.6f}")
    print(f"   標準真實座標: {gt_lat:.6f}, {gt_lon:.6f}")
    print(f"   🎯 當前定位誤差: {err:.2f} 公尺")
    print("==================================================")

if __name__ == "__main__":
    run_drone_localization()