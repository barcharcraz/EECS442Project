# import the necessary packages
import argparse
import datetime
import time
import imutils
import numpy as np
import cv2
from os import path


def process(video_path, output_path=""):
    # must provide a valid path to a video
    if not path.isfile(video_path):
        raise RuntimeError("Incorrect path to video file")
    process_frame_diff_optical_flow(video_path, output_path)


def get_video_name(video_path):
    return str.rsplit(path.basename(path.normpath(video_path)), '.', 1)[0]


def get_video_writer(cap, video_path, output_path=""):
    output_path = path.join(output_path, get_video_name(video_path))
    output_path += "_" + str(time.strftime("%d-%m-%Y-%H-%M-%S")) + '.avi'

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, 24, (int(cap.get(3)), int(cap.get(4))))


def grab_and_convert_frame(cap):
    ret, orig = cap.read()
    # No more frames left to grab or something went wrong
    if not ret:
        print('No frame could be grabbed.')
        return None, None
    frame = imutils.resize(orig, width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian blur the image to reduce noise
    return cv2.GaussianBlur(frame, (21, 21), 0), orig

# uses MOG and dense optical flow
def process_frame_diff_optical_flow(video_path, output_path=""):
    cap = cv2.VideoCapture(video_path)
    first_frame, _ = grab_and_convert_frame(cap)
    # No more frames left to grab or something went wrong
    if first_frame is None:
        return

    writer = get_video_writer(cap, video_path, output_path)

    # Mixture of Gaussian background subtraction model
    fgbg = cv2.BackgroundSubtractorMOG2()
    fgmask_prev = fgbg.apply(first_frame)

    while True:
        # take only every third frame to speed things up
        frame, orig1 = grab_and_convert_frame(cap)
        frame, orig2 = grab_and_convert_frame(cap)
        frame, orig = grab_and_convert_frame(cap)
        orig3 = orig
        if frame is None:
            print('No frame could be grabbed, exiting video processing')
            break

        fgmask = fgbg.apply(frame)
        cv2.imshow('mask', fgmask)

        # resize original for easy viewing
        orig = imutils.resize(orig, width=600)
        cv2.imshow('orig', orig)
        flow = cv2.calcOpticalFlowFarneback(fgmask_prev, fgmask, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        max_mag = np.amax(mag)
        mag = mag / max_mag

        mean, stdDev = cv2.meanStdDev(mag)
        variance = stdDev * stdDev

        print (variance)

        if variance > 0.004:
            writer.write(orig1)
            writer.write(orig2)
            writer.write(orig3)

        fgmask_prev = fgmask

        # Apply optical flow to the foreground mask

        k = cv2.waitKey(30) & 0xff
        if k == ord("q"):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()


def create_and_parse_args():
    """Create the args for the program"""
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="path to the video file")
    ap.add_argument("-o", "--out_dir", type=str, help="path to output directory")
    ap.add_argument("-a", "--min_area", type=int, default=500, help="minimum area size")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = create_and_parse_args()
    process(args.get('video', None), args.get('out_dir', ""))