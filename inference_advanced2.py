import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import mercantile
import cv2
from kornia.feature import LoFTR
import timm
import math

# =====================================================================
# 1. 輔助函數區
# =====================================================================
def get_ground_truth_from_filename(image_path):
    filename = os.path.basename(image_path).replace(".jpg", "")
    parts = filename.split("_")
    z, x, y = int(parts[0]), int(parts[1]), int(parts[2])
    bounds = mercantile.bounds(x, y, z)
    return (bounds.south + bounds.north) / 2.0, (bounds.west + bounds.east) / 2.0

def pixel_to_latlon(x_pixel, y_pixel, img_w, img_h, bounds):
    lon = bounds.west + (x_pixel / img_w) * (bounds.east - bounds.west)
    lat = bounds.north - (y_pixel / img_h) * (bounds.north - bounds.south)
    return lat, lon

# =====================================================================
# 2. 主程式流水線
# =====================================================================
def run_drone_localization():
    # 設定環境
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    GALLERY_DIR = "esri_gallery_2km"
    QUERY_IMAGE_PATH = "competition_gallery_2km/19_437300_227000.jpg"
    
    print("【系統啟動】載入模型...")
    global_model = timm.create_model('resnet50', pretrained=True, num_classes=0).to(device).eval()
    local_matcher = LoFTR(pretrained='outdoor').to(device).eval()
    
    faiss_index = faiss.read_index("satellite_gallery.index")
    with open("satellite_meta.json", 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    # 全局檢索
    data_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tensor = data_transforms(Image.open(QUERY_IMAGE_PATH).convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_feat = global_model(img_tensor)
        query_feat = query_feat / torch.norm(query_feat, p=2, dim=1, keepdim=True)
    
    D, I = faiss_index.search(query_feat.cpu().numpy().astype('float32'), 1)
    matched = metadata_list[I[0][0]]
    matched_path = os.path.join(GALLERY_DIR, matched['filename'])
    bounds = mercantile.bounds(matched['x'], matched['y'], matched['z'])
    print(f"🎯 鎖定衛星瓦片: {matched['filename']} (相似度 {D[0][0]:.4f})")
    
    # LoFTR 匹配
    q_cv = cv2.resize(cv2.imread(QUERY_IMAGE_PATH, 0), (256, 256))
    s_cv = cv2.resize(cv2.imread(matched_path, 0), (256, 256))
    
    input_dict = {
        "image0": torch.from_numpy(q_cv).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0,
        "image1": torch.from_numpy(s_cv).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
    }
    
    with torch.no_grad():
        corrs = local_matcher(input_dict)
    
    pts_q, pts_s = corrs['keypoints0'].cpu().numpy(), corrs['keypoints1'].cpu().numpy()
    
    # 視覺化匹配點
    img_draw = np.hstack((cv2.cvtColor(q_cv, cv2.COLOR_GRAY2BGR), cv2.cvtColor(s_cv, cv2.COLOR_GRAY2BGR)))
    for i in range(len(pts_q)):
        cv2.line(img_draw, (int(pts_q[i][0]), int(pts_q[i][1])), (int(pts_s[i][0])+256, int(pts_s[i][1])), (0, 255, 0), 1)
    cv2.imwrite("debug_matches.jpg", img_draw)
    print("📸 已產出視覺化偵錯圖: debug_matches.jpg")

    # 座標解算與誤差計算
    if len(pts_q) < 4:
        print("【警告】特徵點不足，回退至瓦片中心")
        ex_lat, ex_lon = (bounds.south + bounds.north)/2.0, (bounds.west + bounds.east)/2.0
    else:
        H, _ = cv2.findHomography(pts_q, pts_s, cv2.USAC_MAGSAC, 1.5)
        px, py = cv2.perspectiveTransform(np.array([[[128.0, 128.0]]]), H)[0][0]
        ex_lat, ex_lon = pixel_to_latlon(px, py, 256, 256, bounds)

    gt_lat, gt_lon = get_ground_truth_from_filename(QUERY_IMAGE_PATH)
    dlat, dlon = math.radians(gt_lat - ex_lat), math.radians(gt_lon - ex_lon)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(ex_lat))*math.cos(math.radians(gt_lat))*math.sin(dlon/2)**2
    err = 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    print(f"\n🏁 【最終結果】\nAI 定位: {ex_lat:.6f}, {ex_lon:.6f}\n真實座標: {gt_lat:.6f}, {gt_lon:.6f}\n🎯 當前定位誤差: {err:.2f} 公尺")

if __name__ == "__main__":
    run_drone_localization()