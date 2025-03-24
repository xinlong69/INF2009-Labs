#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =================================================================================================================== #
# A simple script to stream a video from the webcam and notify on screen if you detect motion using simple algorithm.
# =================================================================================================================== #

import cv2 # OpenCV library, used for computer vision tasks such as image processing and video capture.
import numpy as np # Used for numerical operations, though it's not explicitly used in this script.

# Initialize the webcam. Opens the default webcam (device index 0).
# If an external camera is used, it may require a different index (e.g., 1).
cap = cv2.VideoCapture(0)

# Read the first frame
# The first two frames are stored in frame1 and frame2, which will be used to detect differences (motion).
_, frame1 = cap.read()
_, frame2 = cap.read()

# Starts an infinite loop that continuously processes video frames.
while True:
    # Calculate the absolute difference between frames
    # Computes the absolute difference between two consecutive frames to highlight areas where movement has occurred.
    diff = cv2.absdiff(frame1, frame2)

    # Converts the difference image to grayscale to simplify processing.
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Applies a Gaussian blur with a 5x5 kernel to smooth the image and reduce noise.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Converts the blurred image into a binary image.
    #  *** Pixels with intensity > 20 are set to 255 (white).
    #  *** Pixels with intensity ≤ 20 are set to 0 (black).
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Expands white regions in the binary image to fill small gaps, making contours easier to detect.
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Detects the boundaries of moving objects.
    #  *** cv2.RETR_TREE: Retrieves all contours and reconstructs the full hierarchy.
    #  *** cv2.CHAIN_APPROX_SIMPLE: Removes unnecessary points to save memory.
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected contours
    # Iterates through all detected contours. 
    for contour in contours:

        # Finds the smallest rectangle that encloses the detected motion.
        (x, y, w, h) = cv2.boundingRect(contour)

        # Computes the area of the detected contour.
        # If the area is less than 900 pixels, it is ignored to avoid false positives (small movements like noise).
        if cv2.contourArea(contour) < 900:
            continue

        # Draws a green ((0, 255, 0)) rectangle around the detected moving object.
        # 2 specifies the thickness of the rectangle.
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Displays the text "Status: Movement" on the video feed.
        #  *** (10, 20): Position of the text.
        #  *** cv2.FONT_HERSHEY_SIMPLEX: Font style.
        #  *** 1: Font scale.
        #  *** (0, 0, 255): Red color for text.
        #  *** 3: Thickness of the text.
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # Displays the modified frame in a window named "feed".
    cv2.imshow("feed", frame1)

    # Shifts frame2 to frame1 for the next iteration.
    frame1 = frame2

    # Captures a new frame to use in the next iteration.
    _, frame2 = cap.read()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(40) == ord('q'):
        break

# Releases the webcam.
cap.release()

# Closes all OpenCV windows.
cv2.destroyAllWindows()
