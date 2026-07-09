import mercantile

z, x, y = 19, 437301, 227002
bounds = mercantile.bounds(x, y, z)
center_lat = (bounds.south + bounds.north) / 2.0
center_lon = (bounds.west + bounds.east) / 2.0

print(f"【真實標準答案】")
print(f"中心緯度: {center_lat:.6f}")
print(f"中心經度: {center_lon:.6f}")