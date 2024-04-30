import cv2 as cv
import numpy as np

def display_image_at_path(path):
    img = cv.imread(path)
    cv.imshow("img", img)
    cv.waitKey(0)

def save_image(img=None,filename="test.jpg"):
    cv.imwrite(filename, img)

def display_image(img):
    cv.imshow("img", img)
    cv.waitKey(0)

def balck_and_white(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img_gray


def vignette_filter(img,level = 3):
    height, width = img.shape[:2]
    X_kernel = cv.getGaussianKernel(width, width / level)
    Y_kernel = cv.getGaussianKernel(height, height / level)
    main_kernel = Y_kernel * X_kernel.T
    mask = main_kernel / main_kernel.max()
    img_vignette = np.copy(img)

    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask

    return img_vignette

def blur_filter(img, kernel=19):
    blur = cv.GaussianBlur(img, (kernel,kernel), 0, 0)
    return blur

def HDR_filter(img , sigma_s=10,sigma_r=0.2):
    hdr = cv.detailEnhance(img, sigma_s, sigma_r)
    return hdr

def sharpening_filter(img, k=9):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  k, -1],
                       [ 0, -1,  0]])
    sharp = cv.filter2D(img, ddepth = -1, kernel = kernel)
    sharp = np.clip(sharp, 0, 255) #clipping to maintin image
    return sharp

def digital_art_filter(img,sigma_s=10,sigma_r=0.3,blur = 7):
    blur_img = cv.GaussianBlur(img, (blur, blur), 0, 0)
    digital_form = cv.stylization(blur_img, sigma_s, sigma_r)
    return  digital_form

def sketch_filter(img,blur=5):
    img_blur = cv.GaussianBlur(img, (blur,blur), 0, 0)
    sketch, _ = cv.pencilSketch(img_blur)
    return sketch

def canny_filter(img,blur=3,lw = 100, up=120):
    img_blur = cv.GaussianBlur(img, (blur, blur), 0, 0)
    edges = cv.Canny(img_blur, lw, up)
    return edges

def bright_filter(img, level=3):
    img_bright = cv.convertScaleAbs(img, beta = level)
    return img_bright



