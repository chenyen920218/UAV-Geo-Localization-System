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
WEST_LNG = 120.276647  # 嘉義縣朴子市/太保市交界 (西側)
SOUTH_LAT = 23.448404  # (南側)
EAST_LNG = 120.291195  # (東側)
NORTH_LAT = 23.456003  # (北側)

# 縮放層級 (Zoom Level)
# Level 18: 約 0.59 公尺/像素 (適合一般無人機高空)
# Level 19: 約 0.29 公尺/像素 (非常清晰，圖檔量會變 4 倍)
ZOOM_LEVEL = 18

# 設定儲存資料夾
SAVE_DIR = "/Users/chenyenyu/Desktop/國防無人機挑戰賽/competition_gallery_true_18"
os.makedirs(SAVE_DIR, exist_ok=True)

# 國土測繪中心正射影像 WMTS URL (注意：這裡的 {z}/{y}/{x} 會由程式替換)
WMTS_URL_TEMPLATE = "https://wmts.nlsc.gov.tw/wmts/PHOTO2/default/GoogleMapsCompatible/{z}/{y}/{x}"

# 加入 Headers 偽裝成瀏覽器，避免被伺服器阻擋
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

def download_tiles():
    # 2. 獲取涵蓋該經緯度範圍的所有瓦片編號
    # mercantile.tiles 會回傳一個 Generator，包含所有涵蓋此區域的 Tile 物件 (x, y, z)
    tiles = list(mercantile.tiles(WEST_LNG, SOUTH_LAT, EAST_LNG, NORTH_LAT, ZOOM_LEVEL))
    
    print(f"總共需要下載 {len(tiles)} 張地圖瓦片。")
    
    # 3. 迴圈下載每張圖片
    for tile in tiles:
        # 將瓦片編號帶入 URL (注意 y 和 x 的順序)
        url = WMTS_URL_TEMPLATE.format(z=tile.z, y=tile.y, x=tile.x)
        
        # 定義存檔檔名，例如 "18_219356_112704.jpg"
        # 檔名保留 z_x_y，之後 AI 算經緯度時需要靠它換算回來
        filename = f"{tile.z}_{tile.x}_{tile.y}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # 如果檔案已經存在，就跳過 (方便中斷後繼續下載)
        if os.path.exists(filepath):
            continue
            
        try:
            # 發送請求下載圖片
            response = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"成功下載: {filename}")
            else:
                print(f"下載失敗 (Status {response.status_code}): {url}")
                
        except Exception as e:
            print(f"發生錯誤 {filename}: {e}")
            
        # 禮貌性延遲，避免短時間發出幾千個請求被政府伺服器 Ban IP
        time.sleep(0.2) 

    print("下載任務完成！")

if __name__ == "__main__":
    download_tiles()