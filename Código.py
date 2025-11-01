!pip install opencv-python matplotlib numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# ======================================================
# 🟢 Implementação do algoritmo de Convex Hull (C++ → Python)
# ======================================================

class Pt:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

def orientation(a, b, c):
    v = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)
    if v < 0:
        return -1  # sentido horário
    if v > 0:
        return +1  # sentido anti-horário
    return 0

def cw(a, b, c, include_collinear):
    o = orientation(a, b, c)
    return o < 0 or (include_collinear and o == 0)

def ccw(a, b, c, include_collinear):
    o = orientation(a, b, c)
    return o > 0 or (include_collinear and o == 0)

def convex_hull(points, include_collinear=False):
    a = sorted(points, key=lambda p: (p.x, p.y))
    if len(a) == 1:
        return a

    p1, p2 = a[0], a[-1]
    up, down = [p1], [p1]

    for i in range(1, len(a)):
        if i == len(a) - 1 or cw(p1, a[i], p2, include_collinear):
            while len(up) >= 2 and not cw(up[-2], up[-1], a[i], include_collinear):
                up.pop()
            up.append(a[i])
        if i == len(a) - 1 or ccw(p1, a[i], p2, include_collinear):
            while len(down) >= 2 and not ccw(down[-2], down[-1], a[i], include_collinear):
                down.pop()
            down.append(a[i])

    if include_collinear and len(up) == len(a):
        return list(reversed(a))

    hull = up + down[-2:0:-1]
    return hull


# ======================================================
# 🟦 Processamento das 3 Imagens
# ======================================================

img1 = cv2.imread("vaso.jpg")
img2 = cv2.imread("arvore.png")
img3 = cv2.imread("images.jpeg")

if img1 is None: raise FileNotFoundError("Arquivo 'bolinha.jpg' não encontrado!")
if img2 is None: raise FileNotFoundError("Arquivo 'vazo.png' não encontrado!")
if img3 is None: raise FileNotFoundError("Arquivo 'vaso23.png' não encontrado!")

# 1️⃣ Conversão para tons de cinza
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# 2️⃣ Limiarização (binarização)
_, thresh1 = cv2.threshold(gray1, 150, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(gray2, 150, 255, cv2.THRESH_BINARY)
_, thresh3 = cv2.threshold(gray3, 150, 255, cv2.THRESH_BINARY)

# 3️⃣ Detecção de bordas (Canny)
edges1 = cv2.Canny(gray1, 50, 180)
edges2 = cv2.Canny(gray2, 50, 180)
edges3 = cv2.Canny(gray3, 50, 180)

print(" Etapas intermediárias das 3 imagens")
cv2_imshow(gray1)
cv2_imshow(gray2)
cv2_imshow(gray3)

print("=============================Otsu!=============================")
print("Imagem 1 ")
cv2_imshow(thresh1)
print("Imagem 2 ")
cv2_imshow(thresh2)
print("Imagem 3 ")
cv2_imshow(thresh3)
print("=============================Detecçao de borda  Canny=============================")
print("Imagem 1 ")
cv2_imshow(edges1)
print("Imagem 2 ")
cv2_imshow(edges2)
print("Imagem 3 ")
cv2_imshow(edges3)


# ======================================================
# 🟥 Extração dos contornos (Edges + Threshold)
# ======================================================

contoursE1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursE2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursE3, _ = cv2.findContours(edges3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contoursT1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursT2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursT3, _ = cv2.findContours(thresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# ======================================================
# 🟩 Aplicação do Convex Hull em todos os conjuntos
# ======================================================

conjuntos = [
    (img1, contoursE1, "Borda Canny 1"),
    (img2, contoursE2, "Borda Canny 2"),
    (img3, contoursE3, "Borda Canny 3"),
    (img1, contoursT1, "Otsu 1"),
    (img2, contoursT2, "OTsu 2"),
    (img3, contoursT3, "Otsu 3"),
]

for img, conts, nome in conjuntos:
    print(f"\n==============================")
    print(f"🟩 Processando: {nome}")
    print(f"==============================")

    # Junta todos os pontos dos contornos
    points = [Pt(p[0][0], p[0][1]) for cnt in conts for p in cnt]

    if len(points) < 3:
        print("⚠️ Poucos pontos detectados para formar um fecho convexo.")
        continue

    # Calcula o fecho convexo
    hull = convex_hull(points)
    hull_np = np.array([[int(p.x), int(p.y)] for p in hull], np.int32).reshape((-1, 1, 2))

    # Desenha o fecho convexo sobre a imagem
    img_hull = img.copy()
    cv2.polylines(img_hull, [hull_np], isClosed=True, color=(0, 255, 0), thickness=2)

    print("🟢 Fecho convexo detectado!")
    cv2_imshow(img_hull)
