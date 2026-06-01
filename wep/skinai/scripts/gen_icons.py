from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "public" / "icons"
OUT.mkdir(parents=True, exist_ok=True)
BG = (39, 174, 96)
FG = (255, 255, 255)


def make(size: int, maskable: bool = False) -> Image.Image:
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    safe = size * 0.8 if maskable else size
    text = "S"
    try:
        font = ImageFont.truetype("arial.ttf", int(safe * 0.62))
    except OSError:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) / 2 - bbox[0]
    y = (size - th) / 2 - bbox[1]
    d.text((x, y), text, fill=FG, font=font)
    return img


for s in (192, 512):
    make(s).save(OUT / f"icon-{s}.png", optimize=True)
    make(s, maskable=True).save(OUT / f"icon-{s}-maskable.png", optimize=True)
print("wrote", sorted(p.name for p in OUT.glob("icon-*.png")))
