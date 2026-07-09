import os
import time
import requests
import mercantile
import urllib3

# 關閉不安全連線警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 賽場核心參數：亞創中心半徑 2 公里
# ==========================================
WEST_LNG = 120.276647  # 嘉義縣朴子市/太保市交界 (西側)
SOUTH_LAT = 23.448404  # (南側)
EAST_LNG = 120.291195  # (東側)
NORTH_LAT = 23.456003  # (北側)

ZOOM_LEVEL = 20
# 建立一個新的資料夾存放最新圖資，避免跟舊的混在一起
SAVE_DIR = "satellite_gallery_latest"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 圖資來源切換區 (請把想要用的取消註解，其他的加上 #)
# ==========================================

# 🌟 選項 A: Google Maps 衛星圖 (通常最新、色彩最鮮豔)
URL_TEMPLATE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

# 🇹🇼 選項 B: 台灣國土測繪中心 正射影像 (幾何最準、無偏移)
# URL_TEMPLATE = "https://wmts.nlsc.gov.tw/wmts/PHOTO2/default/GoogleMapsCompatible/{z}/{y}/{x}"

# 🗺️ 選項 C: Esri World Imagery (你原本使用的)
# URL_TEMPLATE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

def download_latest_tiles():
    tiles = list(mercantile.tiles(WEST_LNG, SOUTH_LAT, EAST_LNG, NORTH_LAT, ZOOM_LEVEL))
    total_tiles = len(tiles)
    
    print(f"【最新圖資下載啟動】")
    print(f"來源網址: {URL_TEMPLATE.split('//')[1].split('/')[0]}")
    print(f"總計需要下載 {total_tiles} 張 Zoom {ZOOM_LEVEL} 瓦片。")

    success_count = 0
    skip_count = 0

    for i, tile in enumerate(tiles, 1):
        url = URL_TEMPLATE.format(z=tile.z, y=tile.y, x=tile.x)
        filename = f"{tile.z}_{tile.x}_{tile.y}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            skip_count += 1
            continue

        try:
            response = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                success_count += 1
                if i % 50 == 0 or i == total_tiles:
                    print(f"下載進度: [{i}/{total_tiles}] - 成功: {success_count} 張, 跳過: {skip_count} 張")
            else:
                print(f"\n【警告】下載失敗 ({filename}) 狀態碼: {response.status_code}")
                # 遇到 403 阻擋時，強制暫停久一點
                if response.status_code == 403:
                    time.sleep(5)
                    
        except Exception as e:
            print(f"\n【異常】連線錯誤 ({filename}): {e}")

        # 禮貌性延遲，防 Ban IP (Google 抓太快容易被擋)
        time.sleep(0.15)

    print(f"\n【任務結束】實際成功下載: {success_count} 張")

if __name__ == "__main__":
    download_latest_tiles()