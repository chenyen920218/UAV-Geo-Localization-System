import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import faiss
import mercantile
import cv2
import kornia
from kornia.feature import LoFTR
import timm
import math

# =====================================================================
# 1. 參數與路徑配置
# =====================================================================
WEIGHT_PATH = "net_best.pth" 
DB_INDEX_PATH = "satellite_gallery.index"
DB_META_PATH = "satellite_meta.json"
GALLERY_DIR = "esri_gallery_2km" # 你的衛星圖庫資料夾

# 測試輸入的無人機畫面
QUERY_IMAGE_PATH = "competition_gallery_2km/19_437301_227002.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# =====================================================================
# 2. 定義神經網路 (全局檢索 ResNet + 局部匹配 LoFTR)
# =====================================================================
# --- 全局特徵提取網路 ---
class ft_net(nn.Module):
    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(*list(model_ft.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

def init_global_model():
    print("【系統升級】正在載入現代化 EigenPlaces 特徵提取器...")
    # 載入預訓練的 ResNet50 (timm 會自動處理權重下載)
    model = timm.create_model('resnet50', pretrained=True, num_classes=0)
    
    # EigenPlaces 論文建議將輸出維度標準化為 2048 (與 FAISS 對齊)
    # 確保模型處於評估模式
    model.eval()
    return model.to(device)

# --- 局部特徵匹配網路 (LoFTR) ---
def init_local_matcher():
    # 使用預訓練的戶外模型 (outdoor)
    matcher = LoFTR(pretrained='outdoor').to(device).eval()
    return matcher

# 影像預處理 (ResNet 用)
data_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================================================================
# 3. 地理數學換算函數
# =====================================================================
def pixel_to_latlon(x_pixel, y_pixel, img_w, img_h, bounds):
    """將地圖瓦片上的像素座標 (x, y) 轉換為真實經緯度"""
    # 經度：從西邊界依比例向東加
    lon = bounds.west + (x_pixel / img_w) * (bounds.east - bounds.west)
    # 緯度：從北邊界依比例向南減 (注意地圖 Y 軸向下，緯度是越往南越小)
    lat = bounds.north - (y_pixel / img_h) * (bounds.north - bounds.south)
    return lat, lon

# =====================================================================
# 4. 主程式：無人機即時定位流水線
# =====================================================================
def run_drone_localization():
    print("【系統啟動】載入神經網路與地理資料庫...")
    global_model = init_global_model()
    local_matcher = init_local_matcher()
    
    faiss_index = faiss.read_index(DB_INDEX_PATH)
    with open(DB_META_PATH, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    if not os.path.exists(QUERY_IMAGE_PATH):
        print(f"【錯誤】找不到測試圖片 {QUERY_IMAGE_PATH}")
        return

    print("\n==================================================")
    print(" 🚀 階段一：全局檢索 (找尋衛星瓦片)")
    print("==================================================")
    
    # 讀取 Query
    img_pil = Image.open(QUERY_IMAGE_PATH).convert('RGB')
    img_tensor = data_transforms(img_pil).unsqueeze(0).to(device)

    # 提取特徵並比對
    with torch.no_grad():
        query_np = global_model(img_tensor).cpu().numpy().astype('float32')
    
    D, I = faiss_index.search(query_np, 1)
    matched_data = metadata_list[I[0][0]]
    matched_filename = matched_data['filename']
    matched_filepath = os.path.join(GALLERY_DIR, matched_filename)
    
    bounds = mercantile.bounds(matched_data['x'], matched_data['y'], matched_data['z'])
    print(f"🎯 鎖定衛星瓦片: {matched_filename} (相似度 {D[0][0]:.4f})")
    
    print("\n==================================================")
    print(" 🎯 階段二：LoFTR 局部特徵精確匹配")
    print("==================================================")
    
    # LoFTR 需要灰階 (Grayscale) 圖片，且需要調整尺寸讓長寬為 8 的倍數
    # 讀取 Query 影像與對應到的 Satellite 影像
    query_cv = cv2.imread(QUERY_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    query_cv = cv2.resize(query_cv, (256, 256)) 
    
    sat_cv = cv2.imread(matched_filepath, cv2.IMREAD_GRAYSCALE)
    sat_cv = cv2.resize(sat_cv, (256, 256))
    sat_h, sat_w = sat_cv.shape

    # 改用 PyTorch 原生方法：將 OpenCV 的 (H, W) 陣列，擴充為 (Batch, Channel, H, W)
    query_tensor = torch.from_numpy(query_cv).unsqueeze(0).unsqueeze(0).float() / 255.0
    sat_tensor = torch.from_numpy(sat_cv).unsqueeze(0).unsqueeze(0).float() / 255.0
    
    input_dict = {
        "image0": query_tensor.to(device), # 無人機畫面
        "image1": sat_tensor.to(device)    # 衛星畫面
    }

    # 執行 LoFTR 匹配
    with torch.no_grad():
        correspondences = local_matcher(input_dict)
    
    # 提取匹配好的特徵點座標 (N, 2)
    pts_query = correspondences['keypoints0'].cpu().numpy()
    pts_sat = correspondences['keypoints1'].cpu().numpy()
    
    print(f"🔍 LoFTR 成功找出 {len(pts_query)} 對特徵點！")

    # ==================================================
    # 🌟 貼在這裡！因為上面剛剛把 pts_query 和 pts_sat 算出來了！
    # ==================================================
    try:
        # 將原本的灰階圖轉成彩色，這樣才能畫彩色的線
        query_color = cv2.cvtColor(query_cv, cv2.COLOR_GRAY2BGR)
        sat_color = cv2.cvtColor(sat_cv, cv2.COLOR_GRAY2BGR)
        
        # 把兩張圖左右拼起來 (無人機在左，衛星圖在右)
        img_draw = np.hstack((query_color, sat_color))
        
        # 畫出那 25 個點的連線
        for i in range(len(pts_query)):
            pt1 = (int(pts_query[i][0]), int(pts_query[i][1]))
            pt2 = (int(pts_sat[i][0]) + 256, int(pts_sat[i][1]))
            
            cv2.line(img_draw, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(img_draw, pt1, 2, (0, 0, 255), -1)
            cv2.circle(img_draw, pt2, 2, (0, 0, 255), -1)
            
        cv2.imwrite("loftr_matches_debug.jpg", img_draw)
        print("📸 成功產出特徵匹配圖！請查看資料夾內的 loftr_matches_debug.jpg")
    except Exception as e:
        print(f"繪圖失敗: {e}")

    # 提取匹配好的特徵點座標 (N, 2)
    pts_query = correspondences['keypoints0'].cpu().numpy()
    pts_sat = correspondences['keypoints1'].cpu().numpy()
    
    print(f"🔍 LoFTR 成功找出 {len(pts_query)} 對特徵點！")

    if len(pts_query) < 4:
        print("【警告】特徵點太少，無法計算精確偏移，回退至瓦片中心點座標。")
        exact_lat, exact_lon = (bounds.south + bounds.north) / 2.0, (bounds.west + bounds.east) / 2.0
    else:
        # 計算單應性矩陣 Homography (算出無人機畫面相對於衛星圖的變形與位移)
        H, inliers = cv2.findHomography(pts_query, pts_sat, cv2.USAC_MAGSAC, 1.5)
        
        # 我們要知道無人機的「正下方」在哪裡，也就是無人機畫面的「正中心點」
        query_center = np.array([[[128.0, 128.0]]]) # 256x256 的中心
        
        # 透過矩陣把無人機中心點，投射到衛星瓦片的像素座標上
        sat_center_projected = cv2.perspectiveTransform(query_center, H)[0][0]
        px, py = sat_center_projected[0], sat_center_projected[1]
        
        print(f"📍 無人機準星對應在衛星瓦片上的像素座標: X={px:.1f}, Y={py:.1f}")
        
        # 將像素座標轉換為真實世界經緯度
        exact_lat, exact_lon = pixel_to_latlon(px, py, sat_w, sat_h, bounds)

    print("\n==================================================")
    print(" 🏁 【最終高精度解算結果】")
    print(f"   Latitude (緯度):  {exact_lat:.6f}")
    print(f"   Longitude (經度): {exact_lon:.6f}")
    print("==================================================")

if __name__ == "__main__":
    run_drone_localization()