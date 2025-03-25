#%% Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/raspberry_pi/
# Download hand gesture detector model wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python # import the python wrapper
from mediapipe.tasks.python import vision # import the API for calling the recognizer and setting parameters
from mediapipe.framework.formats import landmark_pb2 #The base land mark atlas
'''
  cv2: This is the OpenCV library used for computer vision tasks such as image manipulation, displaying images, and video processing.

    mediapipe: Google’s library for building multimodal (e.g., vision, audio, etc.) machine learning pipelines, particularly used for tasks like hand tracking and gesture recognition.

    time: Python’s time module, used for timestamping and delays.

    python and vision from mediapipe.tasks: These are used for working with specific MediaPipe tasks (like gesture recognition) via Python.

    landmark_pb2: Protobuf (Protocol Buffers) class used to handle landmarks (points in a hand, face, or body) in MediaPipe.
'''

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
'''
  mp_hands: This module provides solutions for hand tracking and landmark detection.

  mp_drawing: This module contains utilities to draw landmarks and connections on images.

  mp_drawing_styles: This module defines drawing styles for landmarks (e.g., colors, thickness).
'''

#%% Parameters
numHands = 2 # Number of hands to be detected
'''
  Set numHands to 1 or 2 to see how the model performs with one versus two hands.
'''

model = 'gesture_recognizer.task' # Model for hand gesture detection Download using wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
minHandDetectionConfidence = 0.5 # Thresholds for detecting the hand
'''
  Change this to a lower or higher value (e.g., 0.3 or 0.8) to observe how well the model detects hands in different lighting or occlusion conditions.
'''

minHandPresenceConfidence = 0.5 # Threshold for detecting the presence of a hand
'''
  Change it to 0.6 or 0.9 to test how it impacts the detection of hands when they are partially visible.
'''

minTrackingConfidence = 0.5 # Threshold for tracking a hand once detected
'''
  Lowering this value might allow the model to track hands even in less stable or occluded conditions,
  while increasing it may make the tracking more accurate but only in ideal conditions.
'''

frameWidth = 640 # Width of the video capture frame
frameHeight = 480 # Height of the video capture frame

# Visualization parameters
row_size = 50  # pixels / Height of each row of text (used for displaying results)
left_margin = 24  # pixels / Margin for text
text_color = (0, 0, 0)  # black / Color for displaying text (black)
font_size = 1 # Size of the text
font_thickness = 1 # Thickness of the text

# Label box parameters
# # Label box parameters for displaying gesture recognition
label_text_color = (255, 255, 255)  # white / Color of the label text (white)
label_font_size = 1 # Font size for the label text
label_thickness = 2 # Thickness of the label text
'''
  These parameters control the hand detection, gesture recognition, and the visualization of results.

      numHands specifies how many hands to detect in a frame (1 or 2).

      model refers to the gesture recognition model used for identifying gestures.

      Threshold parameters (minHandDetectionConfidence, minHandPresenceConfidence, minTrackingConfidence) control the sensitivity of the hand detection and tracking.

      frameWidth and frameHeight specify the resolution of the video capture.

      Visualization parameters control how text and labels are drawn on the image.
'''

#%% Initializing results and save result call back for appending results.
recognition_frame = None
recognition_result_list = []

def save_result(result: vision.GestureRecognizerResult, unused_output_image: mp.Image,timestamp_ms: int):
    recognition_result_list.append(result)
'''
  recognition_result_list is a list where the results of the gesture recognition will be stored.

  save_result is a callback function that appends the recognition results into the recognition_result_list. This function is triggered whenever a gesture is recognized.
'''

#%% Create an Hand Gesture Control object.
# Initialize the gesture recognizer model
base_options = python.BaseOptions(model_asset_path=model)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=numHands,
    min_hand_detection_confidence=minHandDetectionConfidence,
    min_hand_presence_confidence=minHandPresenceConfidence,
    min_tracking_confidence=minTrackingConfidence,
    result_callback=save_result
)
recognizer = vision.GestureRecognizer.create_from_options(options)
'''
  base_options: Specifies the path to the model (gesture_recognizer.task).

  options: These options configure the gesture recognizer, specifying:

      The number of hands to detect.

      Thresholds for detection and tracking confidence.

      The result_callback is set to save_result, which stores recognition results.

  recognizer: The GestureRecognizer object is created using the specified options.


  running_mode=vision.RunningMode.SYNC,  # Synchronous processing
    Effect: The RunningMode parameter determines whether the gesture recognizer runs synchronously or asynchronously.
    Experiment: Test vision.RunningMode.SYNC for real-time processing versus vision.RunningMode.LIVE_STREAM for processing each frame asynchronously.
'''

#%% Open CV Video Capture and frame analysis (setting the size of the capture resolution as per the model requirements)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

'''
  cap: Initializes the webcam for video capture.

  set: Specifies the resolution (width and height) of the video capture.

  isOpened(): Checks if the webcam was successfully opened, raising an error if not.
'''

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame
        ret, frame = cap.read() 
        
        frame = cv2.flip(frame, 1) # To flip the image to match with camera flip
        
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame from BGR to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image) # Convert to MediaPipe image format
        
        current_frame = frame # Save the current frame to process
        
        # Run hand landmarker using the model.        
        recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000) # Recognize gestures asynchronously
        
        if recognition_result_list:
          
          # Draw landmarks and write the text for each hand.
          for hand_index, hand_landmarks in enumerate(
              recognition_result_list[0].hand_landmarks):
           
            # Calculate the bounding box of the hand
            x_min = min([landmark.x for landmark in hand_landmarks])
            y_min = min([landmark.y for landmark in hand_landmarks])
            y_max = max([landmark.y for landmark in hand_landmarks])
    
            # Convert normalized coordinates to pixel values
            frame_height, frame_width = current_frame.shape[:2]
            x_min_px = int(x_min * frame_width)
            y_min_px = int(y_min * frame_height)
            y_max_px = int(y_max * frame_height)
    
            # Get gesture classification results
            if recognition_result_list[0].gestures:
              gesture = recognition_result_list[0].gestures[hand_index]
              category_name = gesture[0].category_name
              score = round(gesture[0].score, 2)
              result_text = f'{category_name} ({score})'
    
              # Compute text size
              text_size = \
              cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size, label_thickness)[0]
              text_width, text_height = text_size
    
              # Calculate text position (above the hand)
              text_x = x_min_px
              text_y = y_min_px - 10  # Adjust this value as needed
    
              # Make sure the text is within the frame boundaries
              if text_y < 0:
                text_y = y_max_px + text_height
    
              # Draw the text
              cv2.putText(current_frame, result_text, (text_x, text_y),
                          cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                          label_text_color, label_thickness, cv2.LINE_AA)
    
            # Draw hand landmarks on the frame using the atlas
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp_drawing.draw_landmarks(
              current_frame,
              hand_landmarks_proto,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style()
            )
    
          recognition_frame = current_frame
          recognition_result_list.clear()
    
        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break
'''
  while True: This loop continuously captures frames from the webcam.

  frame: Each frame is read from the video feed.

  flip: Flips the image horizontally to match the camera view.

  recognizer.recognize_async(): Asynchronously processes the image for gesture recognition.

  Hand landmarks and gestures:

      The hand landmarks are used to calculate a bounding box for each detected hand.

      The gesture recognition results (if available) are displayed as text on the frame, showing the gesture's name and confidence score.

      Hand landmarks are drawn on the frame.

  cv2.putText: Displays the recognized gesture's name and confidence score on the frame.

  cv2.imshow: Displays the processed frame with gesture recognition and hand landmarks.
'''

cap.release()
cv2.destroyAllWindows()
'''
  cap.release(): Releases the webcam when done.

  cv2.destroyAllWindows(): Closes all OpenCV windows displaying the captured frames.
'''