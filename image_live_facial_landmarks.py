import cv2
import face_recognition # Refer to https://github.com/ageitgey/face_recognition
'''
    cv2: OpenCV library used for computer vision tasks like video capture and image processing.

    face_recognition: An easy-to-use library that provides simple methods for facial recognition and landmark extraction.
'''

#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)
'''
    cv2.VideoCapture(0): Initializes the webcam feed. 0 refers to the default webcam.

    If there are multiple cameras, you can specify different indices (e.g., 1, 2, etc.).
'''

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
'''
    cap.isOpened(): Checks if the webcam feed was successfully opened.

    If not, an error is raised: "Cannot open webcam".
'''

# The loop will break on pressing the 'q' key
# The loop continuously reads frames from the webcam until manually stopped.
while True:
    try:
        # Capture one frame
        ret, frame = cap.read()
        '''
            cap.read() captures a single frame from the webcam.

            ret indicates whether the frame was captured successfully (True) or not (False).

            frame is the captured image.
        '''
        
        # Resizing for faster detection
        frame = cv2.resize(frame, (256, 256)) #Uncomment and see the speed up
        '''
            Resizes the frame to 256x256 pixels for faster detection.

            Comment this line to see the impact on speed and accuracy.
        '''
        
        # Extract face locations using the face_recogniton library
        face_locations = face_recognition.face_locations(frame)
        '''
            face_recognition.face_locations(frame) uses the face_recognition library to detect faces in the given frame.

            It returns a list of bounding boxes for the faces found, each represented by (top, right, bottom, left) coordinates.


            face_locations = face_recognition.face_locations(frame, model="hog")  # Change to "cnn" for more accurate but slower detection
                By default, face_recognition.face_locations uses the CNN-based model for face detection,
                but you can switch to the HOG-based model (which is faster but less accurate) using the model parameter.

                Effect of Change:
                    HOG (Histogram of Oriented Gradients) is faster and better suited for real-time applications but is less accurate, especially under poor lighting conditions.
                    CNN provides better accuracy, especially for detecting faces in complex conditions, but is slower.
        '''
              
        # Draw a rectangle around the face
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            '''
                Loops over each detected face's coordinates.

                Draws a rectangle around each detected face using OpenCV’s cv2.rectangle function.

                    (left, top) → Top-left corner of the rectangle.

                    (right, bottom) → Bottom-right corner.

                    (0, 255, 0) → Green color.

                    2 → Line thickness.   
            '''
        
            # Extract the landmarks based on the standard face model
            landmarks = face_recognition.face_landmarks(frame,[face_location])[0]
            '''
                face_recognition.face_landmarks(frame, [face_location]) extracts the facial landmarks (such as eyes, nose, and mouth) for the detected face (face_location).

                The method returns a dictionary with landmark types (eyes, nose, etc.) as keys and a list of points (coordinates of each landmark) as values.

                [face_location] specifies that we want landmarks for a specific face location.
            '''
            
            # Draw circles for each facial landmark
            for landmark_type, landmark_points in landmarks.items():
                for point in landmark_points:
                    cv2.circle(frame,point,2,(0,0,255),-1)
            '''
                Loops through each landmark type (e.g., eyes, mouth) and the corresponding landmark points.

                cv2.circle(frame, point, 2, (0, 0, 255), -1):

                    point → Coordinates of the landmark.

                    2 → Radius of the circle.

                    (0, 0, 255) → Red color for the circle.

                    -1 → Fills the circle (solid).

                
                Increase the radius of the circles or change the color to make the landmarks more noticeable or less distracting, depending on your needs.
            '''
        
        
        cv2.imshow("frame",frame)
        '''
            cv2.imshow("frame", frame): Displays the frame with rectangles around faces and circles around landmarks in a new window named "frame".
        '''
        
        if cv2.waitKey(10) == ord('q'):
           break
        '''
            cv2.waitKey(10): Waits for a key press for 10 milliseconds.

            If the 'q' key is pressed (ord('q')), the loop breaks, stopping the program.
        '''
    
    except KeyboardInterrupt:
        break
    '''
        Gracefully handles the KeyboardInterrupt if the user stops the program manually (e.g., using CTRL+C).
    '''

cap.release()
cv2.destroyAllWindows()
'''
    cap.release(): Releases the webcam resource.

    cv2.destroyAllWindows(): Closes all OpenCV windows opened during the execution.
'''