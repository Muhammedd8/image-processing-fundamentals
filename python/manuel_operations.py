import numpy as np
import cv2
import math

##Griye Çevirme
def manuel_gri_cevir(img):
    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            out[i, j] = int(0.299*r + 0.587*g + 0.114*b)
    return out

##Parlaklık
def manuel_parlaklik(img, deger):
    h, w, c = img.shape
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                val = int(img[i, j, k]) + deger
                out[i, j, k] = 255 if val > 255 else (0 if val < 0 else val)
    return out

##Kontrast
def manuel_kontrast(img, deger):
    h, w, c = img.shape
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                val = float(img[i, j, k]) * deger
                out[i, j, k] = 255 if val > 255 else (0 if val < 0 else int(val))
    return out

##Negatif
def manuel_negatif(img):
    return 255 - img

##Eşikleme
def manuel_esikleme(img, esik):
    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            gray = 0.299*r + 0.587*g + 0.114*b
            out[i, j] = 255 if gray > esik else 0
    return out


##Histogram
def manuel_histogram_hesapla(img):
    hist = np.zeros(256, dtype=int)

    if len(img.shape) == 3:
        h, w = img.shape[:2]
        for i in range(h):
            for j in range(w):
                b, g, r = img[i, j]
                gray = int(0.299*r + 0.587*g + 0.114*b)
                hist[gray] += 1
    else:
        h, w = img.shape
        for i in range(h):
            for j in range(w):
                hist[img[i, j]] += 1
    return hist

##Kontrast Germe
def manuel_kontrast_germe(img):
    h, w = img.shape[:2]
    gray = np.zeros((h, w), dtype=int)
    min_v, max_v = 255, 0

    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            val = int(0.299*r + 0.587*g + 0.114*b)
            gray[i, j] = val
            if val < min_v: min_v = val
            if val > max_v: max_v = val
    if max_v == min_v: return gray.astype(np.uint8)
    
    #Germe işlemini uygulama
    out = np.zeros((h, w), dtype=np.uint8)
    factor = 255.0 / (max_v - min_v)
    for i in range(h):
        for j in range(w):
            out[i, j] = int((gray[i, j] - min_v) * factor)
    return out

##Histogram Eşitleme
def manuel_histogram_esitleme(img):
    hist = manuel_histogram_hesapla(img)
    cdf = [0]*256
    cdf[0] = hist[0]
    for i in range(1, 256): cdf[i] = cdf[i-1] + hist[i]
    
    h, w = img.shape[:2]
    total = h*w
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            gray = int(0.299*r + 0.587*g + 0.114*b)
            out[i, j] = int(cdf[gray] * 255 / total)
    return out

#--FİLTRELER--
def _konv(img, k):
    h, w = img.shape
    pad = len(k)//2
    out = np.zeros((h, w), dtype=np.float32)
    k_sum = sum(sum(row) for row in k)
    if k_sum == 0: k_sum = 1
    
    for i in range(pad, h-pad):
        for j in range(pad, w-pad):
            acc = 0
            for m in range(len(k)):
                for n in range(len(k)):
                    acc += img[i-pad+m, j-pad+n] * k[m][n]
            out[i, j] = acc
    if k_sum > 1: out /= k_sum
    return np.clip(out, 0, 255).astype(np.uint8)

#Mean Filter
def manuel_mean_filter(img):
    g = img if len(img.shape)==2 else np.mean(img, axis=2)
    return _konv(g, [[1,1,1],[1,1,1],[1,1,1]])

##Gauss Filter
def manuel_gauss_filter(img):
    g = img if len(img.shape)==2 else np.mean(img, axis=2)
    return _konv(g, [[1,2,1],[2,4,2],[1,2,1]])

##Median Filter
def manuel_median_filter(img):
    h, w = img.shape[:2]
    g = img if len(img.shape)==2 else np.mean(img, axis=2).astype(np.uint8)
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            win = []
            for m in range(-1, 2):
                for n in range(-1, 2):
                    win.append(g[i+m, j+n])
            win.sort()
            out[i, j] = win[4]
    return out
 
##Laplacian Filter
def manuel_laplasyan_filter(img):
    g = img if len(img.shape)==2 else np.mean(img, axis=2)
    return _konv(g, [[0,1,0],[1,-4,1],[0,1,0]])

##Sobel Filter
def manuel_sobel_filter(img):
    g = img if len(img.shape)==2 else np.mean(img, axis=2)
    gx = _konv(g, [[-1,0,1],[-2,0,2],[-1,0,1]]).astype(float)
    gy = _konv(g, [[-1,-2,-1],[0,0,0],[1,2,1]]).astype(float)
    return np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)

##Prewitt Filter
def manuel_prewitt_filter(img):
    g = img if len(img.shape)==2 else np.mean(img, axis=2)
    px = _konv(g, [[-1,0,1],[-1,0,1],[-1,0,1]]).astype(float)
    py = _konv(g, [[-1,-1,-1],[0,0,0],[1,1,1]]).astype(float)
    return np.clip(np.sqrt(px**2 + py**2), 0, 255).astype(np.uint8)

##Konvulasyon
def manuel_konvulasyon(img):
    g = img if len(img.shape)==2 else np.mean(img, axis=2)
    return _konv(g, [[0,-1,0],[-1,5,-1],[0,-1,0]])

##-MORFOLOJİK İŞLEMLER
def _morf(img, mod):
    h, w = img.shape[:2]
    g = np.mean(img, axis=2).astype(np.uint8) if len(img.shape)==3 else img
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            win = [g[i+m, j+n] for m in range(-1,2) for n in range(-1,2)]
            out[i, j] = max(win) if mod=='max' else min(win)
    return out

def manuel_yayma(img): return _morf(img, 'max')
def manuel_asindirma(img): return _morf(img, 'min')
def manuel_acma(img): return manuel_yayma(manuel_asindirma(img))
def manuel_kapama(img): return manuel_asindirma(manuel_yayma(img))


if __name__ == "__main__":
    
    img = cv2.imread("images/lenna.jpg")

    gri = manuel_gri_cevir(img)
    sobel = manuel_sobel_filter(gri)
    hist_eq = manuel_histogram_esitleme(img)

    cv2.imshow("Gri", gri)
    cv2.imshow("Sobel", sobel)
    cv2.imshow("Histogram Esitleme", hist_eq)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

