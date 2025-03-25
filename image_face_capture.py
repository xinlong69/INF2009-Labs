import cv2
import mediapipe as mp

'''
    cv2 (OpenCV) → Handles video capture and image processing.

    mediapipe → Google's library for real-time face mesh detection.
'''

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
'''
    Loads the face_mesh module from Mediapipe, which detects facial landmarks.
'''

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
'''
    static_image_mode=False → Continuous video processing, not treating each frame as a new image.

    max_num_faces=1 → Detect only one face in the frame.

    min_detection_confidence=0.5 → Minimum 50% confidence required to detect a face.

    min_tracking_confidence=0.5 → Minimum 50% confidence required for tracking landmarks.


    When increase max_num_faces=3, can detect up to 3 faces instead of just 1. Useful for group detection.

    When increase min_detection_confidence=0.5 to 0.7 and min_tracking_confidence=0.5 to 0.7:
        Higher values → Fewer false positives, but may miss faint or small faces.
        Lower values → Detects more faces, but with possible errors.
'''

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
'''
    mp_drawing → Helps draw facial landmarks on the detected face.

    mp_drawing_styles → Provides default styles for drawing.
'''

# Open the camera feed
cap = cv2.VideoCapture(0)
'''
    Captures video from default camera (index 0).
'''

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

'''
    Checks if the camera is available.

    If unavailable, prints an error message and exits.
'''

# Keeps reading frames from the camera until stopped.
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    '''
        Reads a frame from the camera (ret is True if successful).

        If no frame is captured, prints an error and exits.
    '''

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    '''
        OpenCV captures images in BGR format, but Mediapipe expects RGB.

        This converts the image from BGR → RGB.


        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for Mediapipe
            Reduces computation while still allowing face detection.
    '''

    # Process the frame
    results = face_mesh.process(rgb_frame)
    '''
        Detects 468 facial landmarks and stores them in results.
    '''

    # Draw landmarks
    # Checks if any face is detected.
    if results.multi_face_landmarks:

        # Loops through all detected faces (in this case, up to 1).
        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            '''
                Draws the face mesh (triangular tessellation) on the detected face.
            '''
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            '''
                Draws face contours (outlines for eyes, nose, lips, etc.).
            '''

            '''
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,  # Draw only irises (pupils)
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

                # Only eye tracking is drawn, making it less cluttered.
            '''            

    # Display the frame
    # Displays the processed frame with the face mesh overlay.
    cv2.imshow('Mediapipe Face Mesh', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    '''
        Waits for a key press.

        If the user presses 'q', the loop exits.
    '''

cap.release()
cv2.destroyAllWindows()
