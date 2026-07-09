import os
import time
import requests
import mercantile
import urllib3

# 關閉不安全連線的警告 (以防萬一)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 賽場核心參數：亞創中心半徑 2 公里
# ==========================================
WEST_LNG = 120.267221  # 嘉義縣朴子市/太保市交界 (西側)
SOUTH_LAT = 23.432240  # (南側)
EAST_LNG = 120.306421  # (東側)
NORTH_LAT = 23.468240  # (北側)

# 縮放層級
ZOOM_LEVEL = 12

# 儲存路徑
SAVE_DIR = "esri_gallery_2km"
os.makedirs(SAVE_DIR, exist_ok=True)

# Esri World Imagery 的公共 XYZ 瓦片伺服器網址
# 注意：Esri 的排序是 {z}/{y}/{x}
ESRI_URL_TEMPLATE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# 偽裝成普通瀏覽器，避免被伺服器阻擋
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

def download_esri_tiles():
    # 計算該範圍內的所有瓦片
    tiles = list(mercantile.tiles(WEST_LNG, SOUTH_LAT, EAST_LNG, NORTH_LAT, ZOOM_LEVEL))
    total_tiles = len(tiles)
    
    print(f"【Esri 免費圖資下載】")
    print(f"總計需要下載 {total_tiles} 張無浮水印衛星瓦片。")

    success_count = 0
    skip_count = 0

    for i, tile in enumerate(tiles, 1):
        # 組合 URL (帶入 z, y, x)
        url = ESRI_URL_TEMPLATE.format(z=tile.z, y=tile.y, x=tile.x)
        
        filename = f"{tile.z}_{tile.x}_{tile.y}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        # 斷點續傳機制
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            skip_count += 1
            continue

        try:
            # 發送請求，不驗證 SSL
            response = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            
            if response.status_code == 200:
                # Esri 有時候如果該區域沒有圖，會回傳一張全白或「Data not available」的預設圖
                # 但亞創中心是絕對有圖的
                with open(filepath, "wb") as f:
                    f.write(response.content)
                success_count += 1
                
                # 每 50 張回報一次進度
                if i % 50 == 0 or i == total_tiles:
                    print(f"下載進度: [{i}/{total_tiles}] - 成功下載: {success_count} 張, 跳過: {skip_count} 張")
            else:
                print(f"\n【警告】下載失敗 ({filename}) 狀態碼: {response.status_code}")
                if response.status_code == 403:
                    print("可能被伺服器限流了，程式將暫停 5 秒...")
                    time.sleep(5)
                
        except Exception as e:
            print(f"\n【異常】連線錯誤 ({filename}): {e}")

        # 【極度重要】因為是免費用戶，請保持 0.1 秒的延遲，保護你的 IP 不被 Ban
        time.sleep(0.1)

    print(f"\n【任務結束】下載完成！")
    print(f"實際成功下載: {success_count} 張")
    print(f"自動跳過: {skip_count} 張")

if __name__ == "__main__":
    download_esri_tiles()