from PIL import Image
import numpy as np

img = Image.open('mlzukan-img.png').convert('L')
width, height = img.size
img_pixels = []
for y in range(height):
    for x in range(width):
    # getpixelで指定した位置のピクセル値を取得.
        img_pixels.append(img.getpixel((x,y)))

print(img_pixels)
print(len(img_pixels))
