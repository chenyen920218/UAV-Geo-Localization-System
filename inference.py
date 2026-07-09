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

# =====================================================================
# 1. 參數與路徑配置
# =====================================================================
WEIGHT_PATH = "net_best.pth" 
DB_INDEX_PATH = "satellite_gallery.index"
DB_META_PATH = "satellite_meta.json"

# 【測試輸入】請隨便挑選一張你 esri_gallery_2km 資料夾裡的圖片來模擬無人機畫面
# (因為目前是 ImageNet 權重，拿同一張圖測試，相似度應該要接近 100%)
QUERY_IMAGE_PATH = "competition_gallery_2km/19_437345_226987.jpg" # <--- 記得改成你資料夾裡確實存在的一張圖片檔名

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# =====================================================================
# 2. 定義神經網路 (必須與建庫時一模一樣)
# =====================================================================
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

def init_model():
    model = ft_net()
    if os.path.exists(WEIGHT_PATH):
        model_dict = model.state_dict()
        state_dict = torch.load(WEIGHT_PATH, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        # 如果沒權重，同樣使用 ImageNet 權重作為系統測試用
        base_resnet = models.resnet50(pretrained=True)
        model.features = nn.Sequential(*list(base_resnet.children())[:-1])

    model = model.to(device)
    model.eval()
    return model

data_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================================================================
# 3. 主程式：無人機即時定位
# =====================================================================
def run_drone_localization():
    print("【系統啟動】無人機視覺定位系統初始化中...")
    
    # 載入模型
    model = init_model()
    
    # 載入 FAISS 資料庫與 Metadata
    print(f"載入 FAISS 索引檔: {DB_INDEX_PATH}")
    faiss_index = faiss.read_index(DB_INDEX_PATH)
    
    print(f"載入地理關聯檔: {DB_META_PATH}")
    with open(DB_META_PATH, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    print("\n==================================================")
    print("【無人機任務開始】正在獲取相機影像並解算座標...")
    print("==================================================")

    # 步驟 A: 讀取無人機當下畫面
    if not os.path.exists(QUERY_IMAGE_PATH):
        print(f"【錯誤】找不到測試圖片 {QUERY_IMAGE_PATH}，請檢查檔名。")
        return
        
    img = Image.open(QUERY_IMAGE_PATH).convert('RGB')
    img_tensor = data_transforms(img).unsqueeze(0).to(device)

    # 步驟 B: 提取特徵
    with torch.no_grad():
        query_feature = model(img_tensor)
        query_np = query_feature.cpu().numpy().astype('float32')

    # 步驟 C: FAISS 極速檢索 (尋找 Top 1 最相似的衛星圖)
    # k=1 代表只找第一名，D 是相似度分數 (距離)，I 是 FAISS 裡的流水號 ID
    k = 1
    D, I = faiss_index.search(query_np, k)
    
    best_match_id = I[0][0]
    similarity_score = D[0][0]

    # 步驟 D: 解析出對應的地圖瓦片編號
    matched_data = metadata_list[best_match_id]
    z, x, y = matched_data['z'], matched_data['x'], matched_data['y']
    matched_filename = matched_data['filename']

    # 步驟 E: 瓦片編號轉換為 WGS84 精確經緯度
    bounds = mercantile.bounds(x, y, z)
    center_lat = (bounds.south + bounds.north) / 2.0
    center_lon = (bounds.west + bounds.east) / 2.0

    print(f"✅ 成功找到匹配的衛星區塊！")
    print(f"🎯 匹配圖片: {matched_filename}")
    print(f"📊 信心指數 (Cosine Similarity): {similarity_score:.4f} (滿分 1.0)")
    print(f"📍 瓦片編號: Zoom {z}, X {x}, Y {y}")
    print("\n🚀 【無人機當前解算經緯度】")
    print(f"   Latitude (緯度):  {center_lat:.6f}")
    print(f"   Longitude (經度): {center_lon:.6f}")
    print("==================================================")

if __name__ == "__main__":
    run_drone_localization()