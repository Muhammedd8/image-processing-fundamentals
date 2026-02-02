import numpy as np
import cv2

# --- TEMEL ISLEMLER ---
def opencv_gri_cevir(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def opencv_parlaklik(img, val):
    return cv2.convertScaleAbs(img, alpha=1, beta=val)

def opencv_kontrast(img, val):
    return cv2.convertScaleAbs(img, alpha=val, beta=0)

def opencv_negatif(img):
    return cv2.bitwise_not(img)

def opencv_esikleme(img, val):
    gray = opencv_gri_cevir(img)
    _, bw = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
    return bw

# --- HISTOGRAM ---
def opencv_histogram_hesapla(img):
    gray = opencv_gri_cevir(img) if len(img.shape) == 3 else img
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    return hist.flatten().astype(int)

def opencv_kontrast_germe(img):
    gray = opencv_gri_cevir(img)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

def opencv_histogram_esitleme(img):
    gray = opencv_gri_cevir(img)
    return cv2.equalizeHist(gray)

# --- FILTRELER ---
def opencv_mean_filter(img):
    return cv2.blur(img, (7,7))

def opencv_gauss_filter(img):
    return cv2.GaussianBlur(img, (3,3), 0)

def opencv_median_filter(img):
    return cv2.medianBlur(img, 3)

def opencv_laplasyan(img):
    g = opencv_gri_cevir(img)
    return np.uint8(np.absolute(cv2.Laplacian(g, cv2.CV_64F)))

def opencv_sobel(img):
    g = opencv_gri_cevir(img)
    sx = np.absolute(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3))
    sy = np.absolute(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3))
    return cv2.addWeighted(sx, 0.5, sy, 0.5, 0).astype(np.uint8)

def opencv_prewitt(img):
    g = opencv_gri_cevir(img)
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    px = cv2.filter2D(g, -1, kx)
    py = cv2.filter2D(g, -1, ky)
    return cv2.addWeighted(np.abs(px), 0.5, np.abs(py), 0.5, 0).astype(np.uint8)

def opencv_konvulasyon(img):
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, k)

# --- MORFOLOJI ---
def opencv_yayma(img):
    g = opencv_gri_cevir(img)
    return cv2.dilate(g, np.ones((3,3),np.uint8))

def opencv_asindirma(img):
    g = opencv_gri_cevir(img)
    return cv2.erode(g, np.ones((3,3),np.uint8))

def opencv_acma(img):
    g = opencv_gri_cevir(img)
    return cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

def opencv_kapama(img):
    g = opencv_gri_cevir(img)
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
