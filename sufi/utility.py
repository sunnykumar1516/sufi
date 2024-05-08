import cv2 as cv
import numpy as np


vignette_filter_level = 9
blur_filter_kernel = 5
HDR_filter_sigma_s = 10
HDR_filter_sigma_r = 0.2
sharpening_filter_K = 9
digital_art_filter_sigma_s = 10
digital_art_filter_sigma_r = 0.3
digital_art_filter_blur = 7
sketch_filter_blur = 5
canny_filter_lw = 100
canny_filter_up = 120
bright_filter_level = 3


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


def vignette_filter(img,level = vignette_filter_level):
    height, width = img.shape[:2]
    X_kernel = cv.getGaussianKernel(width, width / level)
    Y_kernel = cv.getGaussianKernel(height, height / level)
    main_kernel = Y_kernel * X_kernel.T
    mask = main_kernel / main_kernel.max()
    img_vignette = np.copy(img)

    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask

    return img_vignette

def blur_filter(img, kernel=blur_filter_kernel):
    blur = cv.GaussianBlur(img, (kernel,kernel), 0, 0)
    return blur

def HDR_filter(img , sigma_s=HDR_filter_sigma_s,sigma_r=HDR_filter_sigma_r):
    hdr = cv.detailEnhance(img, sigma_s, sigma_r)
    return hdr

def sharpening_filter(img, k=sharpening_filter_K):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  k, -1],
                       [ 0, -1,  0]])
    sharp = cv.filter2D(img, ddepth = -1, kernel = kernel)
    sharp = np.clip(sharp, 0, 255) #clipping to maintin image
    return sharp

def digital_art_filter(img,sigma_s=digital_art_filter_sigma_s,
                       sigma_r=digital_art_filter_sigma_r,
                       blur = digital_art_filter_blur):
    blur_img = cv.GaussianBlur(img, (blur, blur), 0, 0)
    digital_form = cv.stylization(blur_img, sigma_s, sigma_r)
    return  digital_form

def sketch_filter(img,blur=sketch_filter_blur):
    img_blur = cv.GaussianBlur(img, (blur,blur), 0, 0)
    sketch, _ = cv.pencilSketch(img_blur)
    return sketch

def canny_filter(img,blur=3,
                 lw = canny_filter_lw
                 , up=canny_filter_up):
    img_blur = cv.GaussianBlur(img, (blur, blur), 0, 0)
    edges = cv.Canny(img_blur, lw, up)
    return edges

def bright_filter(img, level=bright_filter_level):
    img_bright = cv.convertScaleAbs(img, beta = level)
    return img_bright


##-------------magic filter helper fun----------------------------

def highlight_line(frame, p1, p2, color=(0, 255, 255), size=3):
    cv.line(frame, p1, p2, color, size)
    return frame


def apply_stange_Filter(targetImg, frame, x, y, size=None):
    if size is not None:
        targetImg = cv.resize(targetImg, size)
    
    newFrame = frame.copy()
    b, g, r, a = cv.split(targetImg)
    overlay_color = cv.merge((b, g, r))
    mask = cv.medianBlur(a, 1)
    h, w, _ = overlay_color.shape
    roi = newFrame[y:y + h, x:x + w]
    
    img1_bg = cv.bitwise_and(roi.copy(), roi.copy(), mask=cv.bitwise_not(mask))
    img2_fg = cv.bitwise_and(overlay_color, overlay_color, mask=mask)
    newFrame[y:y + h, x:x + w] = cv.add(img1_bg, img2_fg)
    return newFrame


def resize_img(cx, width, cy, height, size=(300, 300)):
    x1 = (cx + 300 - width)
    y1 = (cy + 300 - height)
    if (cx + 300 > width):
        size = (300 - x1, 300 - x1)
    elif (cy + 300 > height):
        size = (300 - y1, 300 - y1)
    
    print("current size:", size)
    return size


def get_distance(p1, p2):
    distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return distance


def get_size(z, size=0):
    size = 0
    if (z > 0):
        z = abs(z)
        size = int(z * 100)
    elif (z <= 0):
        z = abs(z)
        size = int(z * 1000)
    return size


def draw_circle(frame, center):
    radius = 10
    color = (0, 255, 200)
    thick = -1
    img = cv.circle(frame, center, radius, color, thick)
    return img


def chech_crash_condition():
    print("ignore")


