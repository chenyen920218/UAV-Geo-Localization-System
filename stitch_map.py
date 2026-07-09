import os
import glob
from PIL import Image
from tqdm import tqdm

# =====================================================================
# 1. 參數設定
# =====================================================================
TILE_DIR = "satellite_gallery_latest"           # 你的衛星圖資料夾
OUTPUT_FILE = "satellite_gallery_latest_Zoom20.jpg" # 輸出的超大地圖檔名
ZOOM_TARGET = "20"                      # 指定要拼的 Zoom Level
TILE_SIZE = 256                         # 每張瓦片的像素大小

# 【重要】解除 PIL 對超大圖片的安全限制 (防爆記憶體機制)
# 因為拼接起來的地圖極大，不加這行程式會報錯 DecompressionBombError
Image.MAX_IMAGE_PIXELS = None 

def stitch_tiles():
    # 尋找所有 18_ 開頭的圖片
    search_pattern = os.path.join(TILE_DIR, f"{ZOOM_TARGET}_*_{'*'}.jpg")
    tile_files = glob.glob(search_pattern)
    
    if not tile_files:
        print(f"【錯誤】在 {TILE_DIR} 找不到任何 Zoom {ZOOM_TARGET} 的圖片！")
        return

    print(f"【準備中】找到 {len(tile_files)} 張瓦片，正在計算畫布大小...")

    # 儲存所有的 x, y 座標以便計算邊界
    x_coords = []
    y_coords = []
    tiles_info = [] # 儲存 (x, y, 檔案路徑)

    for filepath in tile_files:
        try:
            # 解析檔名 (格式: z_x_y.jpg)
            filename = os.path.basename(filepath).replace(".jpg", "")
            parts = filename.split("_")
            z, x, y = int(parts[0]), int(parts[1]), int(parts[2])
            
            x_coords.append(x)
            y_coords.append(y)
            tiles_info.append((x, y, filepath))
        except Exception as e:
            print(f"跳過無法解析的檔案 {filepath}: {e}")

    # 計算網格的邊界
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # 計算最終大圖的長寬 (以瓦片為單位)
    grid_width = max_x - min_x + 1
    grid_height = max_y - min_y + 1
    
    # 轉換為真實像素大小
    pixel_width = grid_width * TILE_SIZE
    pixel_height = grid_height * TILE_SIZE

    print("==================================================")
    print(f" 📐 網格範圍: 水平 {grid_width} 張 x 垂直 {grid_height} 張")
    print(f" 🖼️ 輸出解析度: {pixel_width} x {pixel_height} 像素")
    print("==================================================")

    # 建立純黑的空白大畫布
    print("【繪製中】建立超大畫布並開始拼接...")
    canvas = Image.new('RGB', (pixel_width, pixel_height), color=(0, 0, 0))

    # 開始把每張小圖貼上去
    for x, y, filepath in tqdm(tiles_info, desc="拼接進度"):
        try:
            img = Image.open(filepath)
            
            # 計算這張瓦片在大圖上的精確位置
            # (x - min_x) 代表從最左邊數來第幾格
            # (y - min_y) 代表從最上面數來第幾格
            paste_x = (x - min_x) * TILE_SIZE
            paste_y = (y - min_y) * TILE_SIZE
            
            canvas.paste(img, (paste_x, paste_y))
        except Exception as e:
            print(f"\n無法貼上圖片 {filepath}: {e}")

    # 存檔
    print(f"\n【儲存中】正在將大圖寫入硬碟 (可能需要數十秒至一分鐘)...")
    canvas.save(OUTPUT_FILE, quality=90)
    
    # 算一下檔案大小
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print("==================================================")
    print(f" 🏁 【拼接完成】")
    print(f" 檔案名稱: {OUTPUT_FILE}")
    print(f" 檔案大小: {file_size_mb:.2f} MB")
    print("==================================================")

if __name__ == "__main__":
    stitch_tiles()