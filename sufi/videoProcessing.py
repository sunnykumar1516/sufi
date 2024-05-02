import cv2 as cv
import numpy as np

# adding constant values
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


#display image function
def display_image_at_path(path):
    img = cv.imread(path)
    cv.imshow("img", img)
    cv.waitKey(0)

# save image function
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



def apply_filter_on_video(path,
                          black_and_white=False,
                          vignette_filter=False,
                          blur_filter=False,
                          HDR_filter=False,
                          sharpening_filter=False,
                          digital_art_filter=False,
                          sketch_filter=False,
                          canny_filter = False,
                          bright_filter = False,
                          fps = 24
                          ):


    writer = None
    (W, H) = (None, None)
    vs = cv.VideoCapture(path)
    count = 0
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        if writer is None:
            # initialize our video writer
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter("output.mp4", fourcc, fps,
                                    (frame.shape[1], frame.shape[0]), True)

        if(black_and_white):
            frame = balck_and_white(frame)
        if(vignette_filter):
            frame = vignette_filter(frame)
        if(blur_filter):
            frame = blur_filter(frame)
        if(HDR_filter):
            frame = HDR_filter(frame)
        if(sharpening_filter):
            frame = sharpening_filter(frame)
        if(digital_art_filter):
            frame = digital_art_filter(frame)
        if(sketch_filter):
            frame = sketch_filter(frame)
        if(canny_filter):
            frame = canny_filter(frame)
        if(bright_filter):
            frame = bright_filter(frame)

        count = count + 1
        print("___writing the frame:-",count)

        writer.write(frame)


def extract_frames(path,delay):# dealy is which frame you want to extract, like every 10th
    writer = None
    (W, H) = (None, None)
    vs = cv.VideoCapture(path)
    fp = vs.get(cv.CAP_PROP_FPS)
    print('frames per second =', fp)
    count = 0
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        count = count+1
        if(count % delay == 0):
            fName = str(count)+".png"
            cv.imwrite(fName,frame)
            print("___writing the frame:-", count)

