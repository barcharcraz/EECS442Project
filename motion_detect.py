# import the necessary packages
import argparse
import datetime
import time
import imutils
import numpy as np
import cv2
from os import path


def process(video_path, output_path=None):
    # must provide a valid path to a video
    if not path.isfile(video_path):
        raise RuntimeError("Incorrect path to video file")
    process_frame_diff(video_path, output_path)
    # process_optical_LK(video_path)
    # process_optical_flow(video_path)
    # process_frame_diff_optical(video_path)
    # process_MOG(video_path)


def get_video_name(video_path):
    return str.rsplit(path.basename(path.normpath(video_path)), '.', 1)[0]


def get_video_writer(cap, video_path, output_path=None):
    if output_path is None:
        output_path = ""
    output_path = path.join(output_path, get_video_name(video_path))
    output_path += "_" + str(time.strftime("%d-%m-%Y-%H-%M-%S")) + '.avi'
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    # See this link for what each index corresponds to:
    # http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    return cv2.VideoWriter(output_path, fourcc, cap.get(5), (int(cap.get(3)), int(cap.get(4))))


def grab_and_convert_frame(cap):
    ret, orig = cap.read()
    # No more frames left to grab or something went wrong
    if not ret:
        print('No frame could be grabbed.')
        return None, None
    frame = imutils.resize(orig, width=600)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), orig


def process_frame_diff(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    (background_model, _) = grab_and_convert_frame(cap)
    # No more frames left to grab or something went wrong
    if background_model is None:
        return
    dilation_kernel = np.ones((3, 3), np.uint8)
    writer = get_video_writer(cap, video_path, output_path)

    while True:
        frame, orig = grab_and_convert_frame(cap)
        if frame is None:
            break

        # calculate the difference
        delta = cv2.absdiff(frame, background_model)
        thresh = cv2.threshold(delta, 50, 255, cv2.THRESH_BINARY)[1]
        dilation = cv2.dilate(thresh, dilation_kernel, iterations=1)
        # found foreground so write to video
        if cv2.countNonZero(dilation) > 0:
            print("writing frame")
            writer.write(orig)

        # display frames
        cv2.imshow("Current frame", frame)
        # cv2.imshow("Background model", background_model)
        cv2.imshow("Diff", dilation)

        # current frame becomes background model
        background_model = frame

        # break loop on user input
        k = cv2.waitKey(1) & 0xff
        if k == ord("q"):
            break
        elif k == ord('p'):
            cv2.imwrite("test_frame_" + str(time.strftime("%d-%m-%Y-%H-%M-%S")) + ".png", frame)
            cv2.imwrite("test_thresh_" + str(time.strftime("%d-%m-%Y-%H-%M-%S")) + ".png", dilation)

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


def process_demo(video_path):
    """process the video provided by the video_path to extract portions with motion"""
    camera = cv2.VideoCapture(video_path)
    # initialize the first frame in the video stream. Note, we assume the first frame is the background
    firstFrame = None
    # loop over the frames of the video
    while True:
        # grab the current frame and initialize the occupied/unoccupied text
        (grabbed, frame) = camera.read()
        text = "Unoccupied"

        # if the frame could not be grabbed, then we have reached the end of the video
        if not grabbed:
            print('No frame could be grabbed. Exiting video processing...')
            break

        # resize the frame, convert it to grayscale, and blur it a little for noise reduction
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue

            # compute the bounding box for the contour, draw it on the frame and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


def process_MOG(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.BackgroundSubtractorMOG(5, 5, 0.01)

    while (1):
        grabbed, frame = cap.read()
        text = "Unoccupied"

        # if the frame could not be grabbed, then we have reached the end of the video
        if not grabbed:
            print('No frame could be grabbed. Exiting video processing...')
            break

        frame = imutils.resize(frame, width=600)
        # gray = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        fgmask = fgbg.apply(gray)
        cv2.imshow('mask', fgmask)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_GMG(video_path):
    cap = cv2.VideoCapture(video_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.BackgroundSubtractorMOG2()

    while (1):
        grabbed, frame = cap.read()
        text = "Unoccupied"

        # if the frame could not be grabbed, then we have reached the end of the video
        if not grabbed:
            print('No frame could be grabbed. Exiting video processing...')
            break

        frame = imutils.resize(frame, width=800)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(frame, (21, 21), 0)

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    frame1 = imutils.resize(frame1, width=800)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    prvs = cv2.GaussianBlur(prvs, (21, 21), 0)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while True:
        ret, frame2 = cap.read()
        frame2 = imutils.resize(frame2, width=800)
        next_f = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        next_f = cv2.GaussianBlur(next_f, (21, 21), 0)

        # flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 1, 3, 15, 3, 5, 1)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_f, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png', frame2)
        #     cv2.imwrite('opticalhsv.png', rgb)
        prvs = next_f

    cap.release()
    cv2.destroyAllWindows()


def process_frame_diff_optical(video_path):
    cap = cv2.VideoCapture(video_path)
    (background_model, _) = grab_and_convert_frame(cap)
    # No more frames left to grab or something went wrong
    if background_model is None:
        return

    dilation_kernel = np.ones((3, 3), np.uint8)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    while True:
        frame, orig = grab_and_convert_frame(cap)
        if frame is None:
            break

        # calculate the difference
        delta = cv2.absdiff(frame, background_model)
        thresh = cv2.threshold(delta, 50, 255, cv2.THRESH_BINARY)[1]
        dilation = cv2.dilate(thresh, dilation_kernel, iterations=1)
        nonzeros = cv2.findNonZero(dilation)

        frame_changed = frame.copy();
        # Create a mask image for drawing purposes
        mask = np.zeros_like(background_model)
        if nonzeros is not None and len(nonzeros) > 0:
            nonzeros = np.float32(nonzeros)
            p1, st, err = cv2.calcOpticalFlowPyrLK(background_model, frame_changed, nonzeros, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = nonzeros[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # print(i, (new, old))
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(mask, (a, b), (c, d), color[i % 100].tolist(), 2)
                cv2.circle(frame_changed, (a, b), 5, color[i % 100].tolist(), -1)

            frame_changed = cv2.add(frame_changed, mask)

        # display frames
        cv2.imshow("Current frame", frame_changed)
        # cv2.imshow("Background model", background_model)
        cv2.imshow("Diff", dilation)

        # current frame becomes background model
        background_model = frame

        # break loop on user input
        k = cv2.waitKey(1) & 0xff
        if k == ord("q"):
            break
        elif k == ord('p'):
            cv2.imwrite("test_frame_" + str(time.strftime("%d-%m-%Y-%H-%M-%S")) + ".png", frame)
            cv2.imwrite("test_thresh_" + str(time.strftime("%d-%m-%Y-%H-%M-%S")) + ".png", dilation)

    cap.release()
    cv2.destroyAllWindows()


def process_optical_LK(video_path):
    cap = cv2.VideoCapture(video_path)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        return
    old_frame = imutils.resize(old_frame, width=800)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # old_gray, old_orig = grab_and_convert_frame(cap)
    # if old_gray is None:
    #     return
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        # frame_gray, orig = grab_and_convert_frame(cap)
        ret, frame = cap.read()
        if not ret:
            print('No frame could be grabbed. Exiting video processing...')
            break

        frame = imutils.resize(frame, width=800)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # if the frame could not be grabbed, then we have reached the end of the video
        # if frame_gray is None:
        #     print('No frame could be grabbed. Exiting video processing...')
        #     break

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # print(i, (new, old))
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(mask, (a, b), (c, d), color[i % 100].tolist(), 2)
            cv2.circle(frame_gray, (a, b), 5, color[i % 100].tolist(), -1)

        #
        # cv2.imshow('mask', mask)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()


def create_and_parse_args():
    """Create the args for the program"""
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="path to the video file")
    ap.add_argument("-o", "--out_dir", type=str, help="path to output directory")
    ap.add_argument("-a", "--min_area", type=int, default=500, help="minimum area size")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = create_and_parse_args()
    process(args.get('video', None), args.get('out_dir', None))