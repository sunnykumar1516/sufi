import cv2 as cv
import numpy as np
import utility as ut
import mediapipe as mp

def display_image_at_path(path):
    img = cv.imread(path)
    cv.imshow("img", img)
    cv.waitKey(0)

def save_image(img=None,filename="test.jpg"):
    cv.imwrite(filename, img)

def display_image(img):
    cv.imshow("img", img)
    cv.waitKey(0)

def black_and_white(img):
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



#-------- filters for super hero-----


mp_pose = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_pose.Hands()
circle = cv.imread("red.png", -1)
full_circle = cv.imread("pentagram.png", -1)
shield = cv.imread("shield.png", -1)


def apply_spell():
    rotation = 0
    
    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while cap.isOpened():
        size = (300, 300)
        size_out = (500, 500)
        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        height, width, _ = frame.shape
        results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                positions = []
                for index, item in enumerate(hand.landmark):
                    h, w, _ = frame.shape
                    positions.append([int(item.x * w), int(item.y * h), item.z])
            
            '''for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,
                                          hand_landmarks,
                                          connections=mp_hands.HAND_CONNECTIONS)'''
            
            index = (positions[8][0], positions[8][1])
            thumb = (positions[4][0], positions[4][1])
            centre = (positions[9][0], positions[9][1])
            index_top = (positions[7][0], positions[7][1])
            cx, cy = centre[0], centre[1]
            frame = ut.draw_circle(frame, centre)
            rotation = rotation + 3
            
            frame = ut.highlight_line(frame, index, thumb)
            frame = ut.highlight_line(frame, index_top, thumb)
            distance = ut.get_distance(index, thumb)
            distance2 = ut.get_distance(index_top, thumb)
            print("shield dist:-", distance2)
            
            if distance > 200:
                print("got you")
                
                # inner circle
                h, w, _ = circle.shape
                cen = (h // 2, w // 2)
                cx = cx - cen[1] + 100
                cy = cy - cen[0] + 100
                
                # outer circle
                h_out, w_out, _ = full_circle.shape
                cen_out = (h_out // 2, w_out // 2)
                cx_out = centre[0] - cen_out[1] + 260
                cy_out = centre[1] - cen_out[0] + 260
                
                size_out = (500, 500)
                if (cx + 300 > width or cy + 300 > height):
                    size = ut.resize_img(cx, width, cy, height)
                if (cx_out + 400 > width or cy_out + 400 > height):
                    size_out = ut.resize_img(cx_out, width, cy_out, height, size=size_out)
                
                r1 = rotate_img(cen, rotation, circle, w, h)
                r2 = rotate_img(cen_out, (360 - rotation), full_circle, w_out, h_out)
                try:
                    
                    frame = ut.apply_stange_Filter(r1,
                                                   frame,
                                                   x=cx,
                                                   y=cy,
                                                   size=size)
                    
                    frame = ut.apply_stange_Filter(r2,
                                                   frame,
                                                   x=cx_out,
                                                   y=cy_out,
                                                   size=size_out)
                
                except:
                    print("some error")
            
            if (distance2 < 30):
                # inner circle
                h, w, _ = shield.shape
                cen = (h // 2, w // 2)
                cx = cx - cen[1] + 400
                cy = cy - cen[0] - 200
                
                size = (500, 500)
                if (cx + 500 > width or cy + 500 > height):
                    size = ut.resize_img(cx, width, cy, height)
                try:
                    r1_shield = rotate_img(cen, rotation, shield, w, h)
                    frame = ut.apply_stange_Filter(r1_shield,
                                                   frame,
                                                   x=cx,
                                                   y=cy,
                                                   size=size)
                except Exception as e:
                    print(str(e))
        
        cv.imshow("Image", frame)
        if cv.waitKey(1) == ord("q"):
            break
    
    cap.release()
    cv.destroyAllWindows()


def rotate_img(cen, rotation, circle, w, h):
    rotate_1 = cv.getRotationMatrix2D(cen, round(rotation), 1.0)
    rotate_final = cv.warpAffine(circle, rotate_1, (w, h))
    return rotate_final


def get_position(cx, cy, img, offset=100):
    h, w, _ = img.shape
    cen = (h // 2, w // 2)
    cx = cx - cen[1] + offset
    cy = cy - cen[0] + offset
    return cen, cx, cy



