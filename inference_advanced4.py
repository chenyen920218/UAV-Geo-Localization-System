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
GALLERY_DIR = "esri_gallery_2km"

# 測試輸入的無人機畫面
QUERY_IMAGE_PATH = "competition_gallery_2km/19_437301_227002.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# =====================================================================
# 2. 地理數學與評估函數
# =====================================================================
def pixel_to_latlon(x_pixel, y_pixel, img_w, img_h, bounds):
    """將地圖瓦片上的像素座標 (x, y) 轉換為真實經緯度"""
    lon = bounds.west + (x_pixel / img_w) * (bounds.east - bounds.west)
    lat = bounds.north - (y_pixel / img_h) * (bounds.north - bounds.south)
    return lat, lon

def get_ground_truth_from_filename(image_path):
    """從檔名解析真實座標"""
    filename = os.path.basename(image_path).replace(".jpg", "")
    parts = filename.split("_")
    z, x, y = int(parts[0]), int(parts[1]), int(parts[2])
    bounds = mercantile.bounds(x, y, z)
    return (bounds.south + bounds.north) / 2.0, (bounds.west + bounds.east) / 2.0

def calculate_error(lat1, lon1, lat2, lon2):
    """計算 Haversine 真實地球距離 (公尺)"""
    R = 6371000
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# =====================================================================
# 3. 主程式：無人機即時定位流水線
# =====================================================================
def run_drone_localization():
    print("【系統啟動】載入真正的 EigenPlaces 模型與地理資料庫...")
    
    # 載入真正的跨視角辨識大腦
    global_model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048).to(device).eval()
    local_matcher = LoFTR(pretrained='outdoor').to(device).eval()
    
    faiss_index = faiss.read_index(DB_INDEX_PATH)
    with open(DB_META_PATH, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    if not os.path.exists(QUERY_IMAGE_PATH):
        print(f"【錯誤】找不到測試圖片 {QUERY_IMAGE_PATH}")
        return

    print("\n==================================================")
    print(" 🚀 階段一：全局檢索 (找尋衛星瓦片)")
    print("==================================================")
    
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = data_transforms(Image.open(QUERY_IMAGE_PATH).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = global_model(img_tensor)
        feat = feat / torch.norm(feat, p=2, dim=1, keepdim=True)
        query_np = feat.cpu().numpy().astype('float32')
    
    D, I = faiss_index.search(query_np, 1)
    matched_data = metadata_list[I[0][0]]
    matched_filename = matched_data['filename']
    matched_filepath = os.path.join(GALLERY_DIR, matched_filename)
    
    bounds = mercantile.bounds(matched_data['x'], matched_data['y'], matched_data['z'])
    print(f"🎯 鎖定衛星瓦片: {matched_filename} (相似度 {D[0][0]:.4f})")
    
    print("\n==================================================")
    print(" 🎯 階段二：LoFTR 局部特徵精確匹配")
    print("==================================================")
    
    query_cv = cv2.imread(QUERY_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    query_cv = cv2.resize(query_cv, (256, 256)) 
    sat_cv = cv2.imread(matched_filepath, cv2.IMREAD_GRAYSCALE)
    sat_cv = cv2.resize(sat_cv, (256, 256))

    input_dict = {
        "image0": torch.from_numpy(query_cv).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0,
        "image1": torch.from_numpy(sat_cv).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
    }

    with torch.no_grad():
        correspondences = local_matcher(input_dict)
    
    pts_query = correspondences['keypoints0'].cpu().numpy()
    pts_sat = correspondences['keypoints1'].cpu().numpy()
    print(f"🔍 LoFTR 成功找出 {len(pts_query)} 對特徵點！")

    # 繪製除錯圖
    try:
        img_draw = np.hstack((cv2.cvtColor(query_cv, cv2.COLOR_GRAY2BGR), cv2.cvtColor(sat_cv, cv2.COLOR_GRAY2BGR)))
        for i in range(len(pts_query)):
            pt1 = (int(pts_query[i][0]), int(pts_query[i][1]))
            pt2 = (int(pts_sat[i][0]) + 256, int(pts_sat[i][1]))
            cv2.line(img_draw, pt1, pt2, (0, 255, 0), 1)
        cv2.imwrite("loftr_matches_debug.jpg", img_draw)
        print("📸 成功產出特徵匹配圖: loftr_matches_debug.jpg")
    except Exception as e:
        print(f"繪圖失敗: {e}")

    print("\n==================================================")
    print(" 🛡️ 階段三：嚴格座標解算與防呆保護")
    print("==================================================")

    H = None
    # 1. 嚴格特徵點數量要求 (至少 15 個才算數)
    if len(pts_query) >= 15:
        H, inliers = cv2.findHomography(pts_query, pts_sat, cv2.USAC_MAGSAC, 5.0)
        # 2. 嚴格局內點要求 (符合物理透視的點必須大於 10 個)
        if inliers is None or np.sum(inliers) < 10:
            H = None

    if H is not None:
        query_center = np.array([[[128.0, 128.0]]])
        sat_center_projected = cv2.perspectiveTransform(query_center, H)[0][0]
        px, py = sat_center_projected[0], sat_center_projected[1]
        
        # 3. 邊界防護網 (防止座標飛出圖片外)
        if -50 <= px <= 300 and -50 <= py <= 300:
            print(f"📍 無人機準星對應在衛星瓦片上的像素: X={px:.1f}, Y={py:.1f}")
            exact_lat, exact_lon = pixel_to_latlon(px, py, 256, 256, bounds)
        else:
            print(f"【警告】像素座標 (X={px:.1f}, Y={py:.1f}) 飛出合理範圍，疑似農田誤判。")
            H = None

    # 如果觸發任何保護機制，降級為瓦片中心點
    if H is None:
        print("【降級】啟動防護機制，回退至衛星瓦片中心點座標。")
        exact_lat, exact_lon = (bounds.south + bounds.north) / 2.0, (bounds.west + bounds.east) / 2.0

    print("\n==================================================")
    print(" 🏁 【最終高精度解算結果】")
    print("==================================================")
    print(f"   AI 定位座標: {exact_lat:.6f}, {exact_lon:.6f}")
    
    # 獲取標準答案並計算誤差
    gt_lat, gt_lon = get_ground_truth_from_filename(QUERY_IMAGE_PATH)
    err = calculate_error(exact_lat, exact_lon, gt_lat, gt_lon)
    
    print(f"   標準真實座標: {gt_lat:.6f}, {gt_lon:.6f}")
    print(f"   🎯 當前定位誤差: {err:.2f} 公尺")
    print("==================================================")

if __name__ == "__main__":
    run_drone_localization()