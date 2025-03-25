'''
    This code utilizes the MediaPipe library along with OpenCV to perform hand landmark detection in real-time using a webcam.
    This code enables real-time hand landmark detection with OpenCV and MediaPipe,
    displaying visual feedback on detected thumb landmarks and whether the thumb is up.
'''

#%% Reference: https://github.com/googlesamples/mediapipe/tree/main/examples/hand_landmarker/raspberry_pi
# Download hand land mark detector model wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

'''
    cv2: OpenCV, a library for real-time computer vision tasks (e.g., image and video capture).

    mediapipe: A library by Google for various computer vision tasks such as hand tracking, face detection, etc.

    mediapipe.tasks.python: MediaPipe tasks API that includes predefined models for vision, audio, etc.
'''

#%% Parameters
numHands = 2 # Number of hands to be detected
'''
    numHands: Controls how many hands the model should detect at once.
    Setting it to 1 will only detect a single hand, while 2 will allow detection of both hands.
'''

model = 'hand_landmarker.task' # Model for finding the hand landmarks Download using wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
minHandDetectionConfidence = 0.5 # Thresholds for detecting the hand / Confidence threshold for detecting a hand
'''
    The confidence threshold for detecting a hand. Lower values (e.g., 0.3) will make the detector more sensitive and potentially detect hands at lower confidence,
    but could also increase false positives. Higher values (e.g., 0.7) make the detection stricter but reduce false positives.
'''

minHandPresenceConfidence = 0.5 # Confidence threshold for hand presence
'''
    Similar to the detection confidence, but for determining whether a hand is present.
    Lowering it may make the hand more likely to be detected in challenging conditions (e.g., partially obscured hands).
'''

minTrackingConfidence = 0.5 # Confidence threshold for tracking the hand
'''
    Threshold for tracking the hand once detected.
    Lower values will make the tracking more tolerant of small changes in position, but might result in less stable tracking.
'''

frameWidth = 640 # Frame width of the webcam input
frameHeight = 480 # Frame height of the webcam input
'''
    These control the resolution of the captured video stream.
    Lower values (e.g., 320x240) will process frames faster but with less detail,
    while higher values (e.g., 1280x720) will provide more detail but at a higher computational cost.
'''

# Visualization parameters
MARGIN = 10  # pixels / Margin for drawing shapes
FONT_SIZE = 1 # Font size for text display
FONT_THICKNESS = 1 # Font thickness for text
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green / Text color for displaying hand status
'''
    These are the parameters for controlling various aspects of the hand tracking:

        numHands: The number of hands to track simultaneously.

        model: The path to the model file (hand landmark model) that you need to download.

        minHandDetectionConfidence: Minimum confidence required to consider a hand as detected.

        minHandPresenceConfidence: Minimum confidence required to check if the hand is present in the frame.

        minTrackingConfidence: Confidence required for successfully tracking the hand.

        frameWidth and frameHeight: Define the resolution of the captured video stream.

        MARGIN, FONT_SIZE, FONT_THICKNESS, HANDEDNESS_TEXT_COLOR: Visualization and drawing settings for rendering output.
'''

#%% Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path=model)
options = vision.HandLandmarkerOptions(
        base_options=base_options,        
        num_hands=numHands,
        min_hand_detection_confidence=minHandDetectionConfidence,
        min_hand_presence_confidence=minHandPresenceConfidence,
        min_tracking_confidence=minTrackingConfidence)
detector = vision.HandLandmarker.create_from_options(options)
'''
    base_options: Specifies the model file (hand_landmarker.task) to be used for hand detection.

    HandLandmarkerOptions: Configuration options for the hand detection model. These include the model file path and thresholds for detecting and tracking hands.

    detector: The main object used for hand detection, created with the provided options.
'''

#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
'''
    cap = cv2.VideoCapture(0): Opens the webcam (device 0).

    cap.set(): Sets the resolution of the captured video stream to 640x480 pixels (defined earlier).
'''

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
'''
    Checks if the webcam is successfully opened. If not, it raises an error.
'''

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame
        ret, frame = cap.read() 
        
        frame = cv2.flip(frame, 1) # To flip the image to match with camera flip
        '''
            Captures one frame from the webcam and flips it horizontally so that it appears as if it is being viewed in a mirror.
        '''

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        '''
            Converts the frame from BGR (OpenCV default color format) to RGB (required by MediaPipe).
        '''
        
        # Run hand landmarker using the model.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = detector.detect(mp_image)
        '''
            mp_image: Creates an image object in MediaPipe format using the RGB frame.

            detector.detect(mp_image): Passes the image to the hand landmarker for hand detection. The result contains hand landmarks if detected.
        '''
        
        hand_landmarks_list = detection_result.hand_landmarks
        '''
            hand_landmarks_list: List of detected hand landmarks.
        '''
        
        #handedness_list = detection_result.handedness # Could be used to check for which hand
        
        # Loop through the detected hands to visualize.
        # Loops through each detected hand and visualizes the landmarks.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Detect Thumb and draw a circle on the thumb tip            
            x = int(hand_landmarks[4].x * frame.shape[1]) # Index 4 corresponds to the thump tip as from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
            y = int(hand_landmarks[4].y * frame.shape[0])           
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            '''
                The thumb tip (index 4) is a landmark, and its x and y coordinates are converted to pixel values. A circle is drawn on the thumb tip.
            '''
            
            # Detect Thumb and draw a circle on the index finger tip
            x = int(hand_landmarks[8].x * frame.shape[1]) # Index 8 corresponds to the index finger tip as from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
            y = int(hand_landmarks[8].y * frame.shape[0])           
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            '''
                Similarly, the index finger tip (index 8) is detected and visualized.
            '''

            # Define a threshold for thumb is up/down and display when thums up
            threshold = 0.1
            thumb_tip_y = hand_landmarks[4].y
            thumb_base_y = hand_landmarks[1].y # Index 1 corresponds to the thump base as from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
            thums_up = thumb_tip_y < thumb_base_y - threshold
            '''
                thumb_tip_y and thumb_base_y are the vertical positions of the thumb tip and base (index 1).

                If the thumb tip's vertical position is above the base's vertical position by more than the threshold, it is considered "thumb up."
            '''
            
            if thums_up:
                cv2.putText(frame, 'Thumb Up', (10,30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            '''
                If the thumb is up, the text "Thumb Up" is displayed on the frame.
            '''
                 
            # Displays the annotated frame (with hand landmarks and thumb status).
            cv2.imshow('Annotated Image', frame)
        
        # Exits the loop when the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
'''
    cap.release(): Releases the webcam capture.

    cv2.destroyAllWindows(): Closes any OpenCV windows that are open.
'''