'''
    This code captures video from a webcam, applies an object detection model (EfficientDet Lite), and visualizes the detection results (bounding boxes and labels) on the frames.
    The program processes each frame asynchronously, detecting objects, drawing bounding boxes around them, and showing the results in a live video feed. 
'''

#%% Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/raspberry_pi
# Download lightweight ftlite EfficientDet model using wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python # import the python wrapper
from mediapipe.tasks.python import vision # import the API for calling the recognizer and setting parameters
'''
    cv2: OpenCV library used for real-time computer vision tasks (like capturing video and processing images).

    mediapipe: Google’s MediaPipe library used for building cross-platform multimodal applied ML pipelines, such as object detection and pose estimation.

    time: To get the current timestamp (for async processing or logging).

    mediapipe.tasks.python: The Python wrapper for MediaPipe models, which makes it easier to interact with pre-trained models like object detection.
'''

#%% Parameters
maxResults = 5
scoreThreshold = 0.25
frameWidth = 640
frameHeight = 480
model = 'efficientdet.tflite'
'''
    maxResults: Specifies the maximum number of objects to detect in each frame (top 5 objects).

    scoreThreshold: The minimum confidence score for a detection to be considered valid (objects with scores less than 0.25 will not be detected).

    frameWidth and frameHeight: The resolution of the captured frames from the webcam, set to 640x480 pixels here.

    model: Path to the pre-trained EfficientDet Lite object detection model in TensorFlow Lite format (efficientdet.tflite).
'''

# Visualization parameters
MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # black
'''
    MARGIN: The distance from the top-left corner where the label text will be drawn.

    ROW_SIZE: Height space allocated for the label, ensuring it doesn't overlap with the object.

    FONT_SIZE, FONT_THICKNESS: Control the size and thickness of the text for the labels.

    TEXT_COLOR: Color of the text (black in this case).
'''


#%% Initializing results and save result call back for appending results.
detection_frame = None
detection_result_list = []
  
def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
      detection_result_list.append(result)
'''
    detection_frame: Used to store the processed frame with the detection result for later display.

    detection_result_list: A list that stores the object detection results.

    save_result: A callback function that appends the detection result (such as detected objects and their details) into detection_result_list. This function is called asynchronously with the detect_async method.
'''

#%% Create an object detection model object.
# Initialize the object detection model
base_options = python.BaseOptions(model_asset_path=model)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.LIVE_STREAM,
                                       max_results=maxResults, score_threshold=scoreThreshold,
                                       result_callback=save_result)
detector = vision.ObjectDetector.create_from_options(options)
'''
    BaseOptions: Specifies the path to the model file (efficientdet.tflite).

    ObjectDetectorOptions: Configures the object detector with several settings:

        running_mode=vision.RunningMode.LIVE_STREAM: Specifies that the model should run in real-time, processing each frame asynchronously.

        max_results=maxResults: Limits the number of detected objects to 5.

        score_threshold=scoreThreshold: Only detections with a score above 0.25 will be processed.

        result_callback=save_result: Specifies the callback function (save_result) to store the detection results asynchronously.

    detector: Creates an object detector using the configured options.
'''

#%% Open CV Video Capture and frame analysis (setting the size of the capture resolution as per the model requirements)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
'''
    cv2.VideoCapture(0): Initializes the webcam feed (index 0 typically refers to the default camera).

    cap.set(...): Configures the resolution of the webcam capture to 640x480 pixels.
'''

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
'''
    This checks if the webcam was successfully opened. If not, it raises an error.
'''

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame
        ret, frame = cap.read() 
        
        frame = cv2.flip(frame, 1) # To flip the image to match with camera flip
        
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        current_frame = frame

        '''
            cap.read(): Captures one frame from the webcam.

            cv2.flip(frame, 1): Flips the frame horizontally to match the natural orientation of the camera.

            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB): Converts the frame from BGR (OpenCV format) to RGB (used by TensorFlow Lite models).

            mp.Image(...): Converts the frame to MediaPipe's image format.
        '''
        
        # Run object detection using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        '''
            detector.detect_async(...): Runs object detection asynchronously on the provided image (mp_image). The timestamp is given in milliseconds.
        '''
        
        if detection_result_list:
            for detection in detection_result_list[0].detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                # Use the orange color for high visibility.
                cv2.rectangle(current_frame, start_point, end_point, (0, 165, 255), 3)
            
                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (MARGIN + bbox.origin_x,
                                 MARGIN + ROW_SIZE + bbox.origin_y)
                cv2.putText(current_frame, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            detection_frame = current_frame
            detection_result_list.clear()
        '''
            detection_result_list: If the detection results are available, it processes each detected object.

            Bounding box: The detected object's location in the frame is extracted and drawn using cv2.rectangle().

            Category and Probability: The category (e.g., "person", "car") and detection score are displayed as text using cv2.putText().

            detection_frame: Stores the processed frame for later display.

            detection_result_list.clear(): Clears the result list after processing.

        '''
    
        if detection_frame is not None:
            cv2.imshow('object_detection', detection_frame)
        '''
            Displays the processed frame with detected objects in a window titled object_detection.
        '''
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        '''
            This checks if the 'q' key is pressed. If so, it exits the loop.
        '''
   
    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
'''
    cap.release(): Releases the webcam resource.

    cv2.destroyAllWindows(): Closes any OpenCV windows opened during the program execution.
'''