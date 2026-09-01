"""Converte o dataset de detecção epi-v1 (YOLO) em dataset de classificação
recortando cada bounding box (crop) e colocando na pasta da classe."""
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset" / "exports" / "epi-v1"
DST = ROOT / "dataset_classificacao"
CLASSES = ["Capacete", "Colete", "Pessoa"]

def crops_from_image(img_path, label_path, out_dir):
    """Recorta cada bounding box e salva como imagem de classe."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    if not label_path.exists():
        return 0
    count = 0
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:5])
        x1 = int((x_c - w / 2) * W)
        y1 = int((y_c - h / 2) * H)
        x2 = int((x_c + w / 2) * W)
        y2 = int((y_c + h / 2) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2))
        cls = CLASSES[cid]
        class_dir = out_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{img_path.stem}_{count}.jpg"
        crop.save(class_dir / out_name)
        count += 1
    return count

def build_split(split_name):
    img_dir = SRC / split_name / "images"
    lab_dir = SRC / split_name / "labels"
    out = DST / split_name
    counts = {c: 0 for c in CLASSES}
    total_crops = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        lab_path = lab_dir / (img_path.stem + ".txt")
        n = crops_from_image(img_path, lab_path, out)
        if n:
            total_crops += n
    for c in CLASSES:
        counts[c] = len(list((out / c).glob("*.jpg"))) if (out / c).exists() else 0
    print(f"[{split_name}] Total crops: {total_crops}")
    for c in CLASSES:
        print(f"  {c}: {counts[c]}")
    return total_crops

if __name__ == "__main__":
    for split in ("train", "valid", "test"):
        if not (SRC / split / "images").exists():
            print(f"Aviso: split '{split}' não encontrado, pulando.")
            continue
        build_split(split)
    print("Dataset de classificação (crops) criado em:", DST)
