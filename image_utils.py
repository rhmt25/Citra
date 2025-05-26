import cv2
import numpy as np

# Fungsi untuk mengubah citra berwarna menjadi grayscale
# def to_grayscale(image):
#     return ...
def to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image


# Fungsi untuk menerapkan Gaussian blur
# def gaussian_blur(image, ksize=(3,3)):
#     return ...
def gaussian_blur(image, ksize=(3, 3)):
    return cv2.GaussianBlur(image, ksize, 0)


# Fungsi untuk menerapkan thresholding Otsu pada citra grayscale
# def apply_otsu_threshold(gray):
#     _, thresh = cv2.threshold(...)
#     return ...
def apply_otsu_threshold(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# Fungsi untuk mendeteksi tepi menggunakan operator Prewitt
# def prewitt_edge(image):
#     kernelx = np.array([...])
#     kernely = np.array([...])
#     x = cv2.filter2D(...)
#     y = cv2.filter2D(...)
#     return ...
def prewitt_edge(image):
    kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    x = cv2.filter2D(image, -1, kernelx)
    y = cv2.filter2D(image, -1, kernely)
    return cv2.addWeighted(x, 1, y, 1, 0)


# Fungsi untuk mendeteksi tepi menggunakan operator Sobel
# def sobel_edge(image):
#     grad_x = cv2.Sobel(...)
#     grad_y = cv2.Sobel(...)
#     abs_grad_x = cv2.convertScaleAbs(...)
#     abs_grad_y = cv2.convertScaleAbs(...)
#     return cv2.addWeighted(....)
def sobel_edge(image):
    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


# Fungsi untuk melakukan histogram equalization
# def histogram_equalization(image):
#     return ...
def histogram_equalization(image):
    return cv2.equalizeHist(image)


# Fungsi untuk mengompresi citra grayscale menggunakan kuantisasi
# def quantize_compression(image, levels=16):
#     image = cv2.cvtColor(...) if len(image.shape) == 3 else image
#     step = ...
#     compressed = ...
#     return compressed.astype(...)
def quantize_compression(image, levels=16):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    step = 256 // levels
    compressed = np.floor(image / step) * step
    return compressed.astype(np.uint8)

