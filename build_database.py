import os
import glob
import json
import torch
import torch.nn as nn
import timm # 建議直接使用 timm 來載入強大的預訓練模型
from PIL import Image
import numpy as np
import faiss
from tqdm import tqdm
from torchvision import transforms

# =====================================================================
# 1. 參數與路徑配置
# =====================================================================
GALLERY_DIR = "satellite_gallery_latest"
DB_INDEX_PATH = "satellite_gallery.index"
DB_META_PATH = "satellite_meta.json"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# =====================================================================
# 2. 現代化模型提取器 (直接呼叫 EigenPlaces 預訓練權重)
# =====================================================================
def get_model():
    print(f"【系統載入】正在從 PyTorch Hub 下載真正的 EigenPlaces 權重...")
    # 這是學術界官方的呼叫方式，自動下載真正的地貌辨識大腦
    model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
    model = model.to(device)
    model.eval()
    return model

# =====================================================================
# 3. 處理影像
# =====================================================================
# 建議移除 RandomPerspective，只在訓練時做；建資料庫時要保證圖像是「乾淨且一致」的
data_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_vector_database():
    model = get_model()
    image_paths = sorted(glob.glob(os.path.join(GALLERY_DIR, "*.jpg")))
    
    if not image_paths:
        print("【錯誤】找不到圖片！")
        return

    # 【自動獲取維度】先跑一張圖看看模型輸出維度，不用手寫 2048
    test_tensor = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        test_feat = model(test_tensor)
        feature_dim = test_feat.shape[1]
    
    print(f"【檢測】模型輸出特徵維度為: {feature_dim}")
    
    faiss_index = faiss.IndexFlatIP(feature_dim)
    metadata_list = []

    print(f"【開始建庫】共 {len(image_paths)} 張圖...")
    
    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(image_paths)):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = data_transforms(img).unsqueeze(0).to(device)
                
                # 特徵提取
                features = model(img_tensor)
                # L2 正規化 (這是確保 Cosine Similarity 有效的關鍵)
                features = features / torch.norm(features, p=2, dim=1, keepdim=True)
                features_np = features.cpu().numpy().astype('float32')
                
                faiss_index.add(features_np)
                
                # 解析檔名 z, x, y
                filename = os.path.basename(img_path)
                parts = filename.replace(".jpg", "").split("_")
                metadata_list.append({
                    "faiss_id": idx, "filename": filename,
                    "z": int(parts[0]), "x": int(parts[1]), "y": int(parts[2])
                })
            except Exception as e:
                print(f"跳過錯誤影像: {img_path}, 錯誤: {e}")

    faiss.write_index(faiss_index, DB_INDEX_PATH)
    with open(DB_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4)
        
    print("【建庫完成】已更新索引檔案。")

if __name__ == "__main__":
    build_vector_database()