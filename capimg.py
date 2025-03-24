#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================ #
# A simple script to capture a single video frame from the webcam and save it.
# ============================================================================ #

import cv2 # OpenCV library, used for computer vision tasks such as image processing and video capture.

# Initialize the webcam. Opens the default webcam (device index 0).
# If an external camera is used, it may require a different index (e.g., 1).
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
# If the webcam cannot be accessed (e.g., another program is using it), 
# an IOError is raised with the message "Cannot open webcam".
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Capture one (single) frame
#  *** ret: Boolean value indicating success (True) or failure (False) in capturing a frame.
#  *** frame: The actual image captured from the webcam.
ret, frame = cap.read()

# Save the captured image
# Saves the captured frame as a .jpg file named "captured_image.jpg".
# The first argument specifies the filename, and the second argument is the image data.
cv2.imwrite('captured_image.jpg', frame)

# Release the webcam
# Frees the webcam resource so other programs can use it.
cap.release()

print("Image captured and saved!")
