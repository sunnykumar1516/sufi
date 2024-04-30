import cv2 as cv
import utility as ut


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
            frame = ut.balck_and_white(frame)
        if(vignette_filter):
            frame = ut.vignette_filter(frame)
        if(blur_filter):
            frame = ut.blur_filter(frame)
        if(HDR_filter):
            frame = ut.HDR_filter(frame)
        if(sharpening_filter):
            frame = ut.sharpening_filter(frame)
        if(digital_art_filter):
            frame = ut.digital_art_filter(frame)
        if(sketch_filter):
            frame = ut.sketch_filter(frame)
        if(canny_filter):
            frame = ut.canny_filter(frame)
        if(bright_filter):
            frame = ut.bright_filter(frame)

        print("___writing the frame")

        writer.write(frame)


