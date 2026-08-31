"""Gera comparativo visual BGR vs RGB para inspeção."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, '.')

img_path = sorted(Path("dataset/exports/epi-v1/valid/images").glob("*.jpg"))[0]
frame = cv2.imread(str(img_path))

# Salva o frame como está (BGR interpretado) e a versão correta RGB.
# Para inspeção visual correta, ambas devem ser reconvertidas para BGR
# na gravação (imwrite espera BGR).
cv2.imwrite("preprocessing/outputs/e1_rgb_correto.jpg", frame)
cv2.imwrite("preprocessing/outputs/e1_rgb_com_erro.jpg",
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

print("Imagens salvas em preprocessing/outputs/")
print("Do seu computador, substitua <IP_DO_PI> e rode:")
print("IP_DO_PI=<seu-IP>")
print("scp pi@$IP_DO_PI:~/yolo-edge-api/preprocessing/outputs/*.jpg .")
