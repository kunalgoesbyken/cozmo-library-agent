import sys

from PIL import Image, ImageFilter
FPS = 30
W, H = 1920, 1080
pel = Image.open(sys.argv[1]).convert("RGB")
lib = Image.open(sys.argv[2]).convert("RGB")
PEL_Y = 430

def pelican(t):
    if t < 4: return 0, 0
    if t < 30: return (t - 4) * 30, 0
    if t < 32.5: return 780 + (t - 30) * 300, 14
    if t < 33.5: return 1530, 7
    if t < 38.5: return 1530 - (t - 33.5) * 150, 3
    if t < 76.5: return 780 + (t - 38.5) * 30, 0
    return 1920, 0

def frame(t):
    if t < 80:
        x, blur = pelican(t)
        img = pel.crop((int(x), PEL_Y, int(x) + W, PEL_Y + H))
    else:
        y = min(200, (t - 80) * 8)
        img = lib.crop((0, int(y), W, int(y) + H))
        blur = 0
    if blur:
        img = img.filter(ImageFilter.BoxBlur(blur))
    return img

with open("camera.mjpeg", "wb") as out:
    total = 112 * FPS
    for i in range(total):
        frame(i / FPS).save(out, "JPEG", quality=80)
print("frames", total)
