'''
    This code continuously captures video frames, processes them with a MobileNetV2 model (quantized or not), and logs the frame rate at regular intervals.
    The predictions are made but not printed unless the relevant lines are uncommented.
'''

import time
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import cv2
from PIL import Image
'''
    time : used for tracking and logging the frame rate and performance.

    torch : PyTorch library, which is used for working with deep learning models and tensors.

    numpy : commonly used for numerical computations, though it's not directly used in this code.

    models from torchvision : provides pre-trained deep learning models, and transforms, which is used for image preprocessing.

    MobileNet_V2_QuantizedWeights from torchvision.models.quantization : contains the quantized weights for MobileNetV2, a lightweight deep learning model for image classification.

    cv2 library (OpenCV) : used for computer vision tasks like video capture and image processing.

    Image class from the PIL (Python Imaging Library) : for image manipulation (though it's not directly used in this code).
'''

quantize = False
'''
    Sets a variable quantize to False, which will determine whether to use quantized weights for the model or not.
'''

if quantize:
    torch.backends.quantized.engine = 'qnnpack'
'''
    If quantize is True, this line sets the backend for quantized models to qnnpack,
    which is a lightweight, optimized engine for mobile devices.
    
    Since quantize is False, this block won't be executed.
'''

cap = cv2.VideoCapture(0)
'''
    Initializes video capture using OpenCV, where 0 specifies the default camera (usually the built-in webcam).

    It will be used to capture frames from the camera.
'''

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)
'''
    These lines set the properties of the video capture:

    CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT set the frame width and height to 224x224 pixels.

    CAP_PROP_FPS sets the frame rate to 36 frames per second.
'''

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
'''
    Defines a preprocessing pipeline for the images. The pipeline:

        Converts the image to a tensor (ToTensor).

        Normalizes the pixel values based on the mean and standard deviation of the ImageNet dataset (Normalize).
'''

weights = MobileNet_V2_QuantizedWeights.DEFAULT
'''
    Loads the default quantized weights for the MobileNetV2 model.
'''

classes = weights.meta["categories"]
'''
    Extracts the class labels (categories) from the quantized weights metadata.
    
    These labels correspond to the categories the MobileNetV2 model was trained to classify (e.g., "cat", "dog", etc.).

'''

net = models.quantization.mobilenet_v2(pretrained=True, quantize=quantize)
'''
    Initializes the MobileNetV2 model with pre-trained weights.

    If quantize is True, it loads the quantized version of the model; otherwise, it loads the regular model.
'''

started = time.time()
last_logged = time.time()
frame_count = 0
'''
    Initializes the following variables:

    started: Stores the start time for tracking the overall time the program has been running.

    last_logged: Stores the last time the frame rate was logged.

    frame_count: Keeps track of how many frames have been processed.
'''


# Starts a context manager to temporarily disable gradient calculation.
# This is useful for inference when gradients are not needed, reducing memory usage and speeding up the computation.
with torch.no_grad():

    # Starts an infinite loop to continuously capture and process frames.
    while True:

        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")
        '''
            Reads a frame from the video capture. ret is a boolean indicating whether the frame was read successfully, and image contains the frame data.
        
            If reading the frame fails (ret is False), it raises an error with a message.
        '''

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        '''
            Converts the image from BGR (OpenCV's default color format) to RGB by reordering the color channels.
        '''

        permuted = image
        '''
            Stores the RGB image in a new variable permuted, though this line doesn't seem to be necessary since permuted is not used elsewhere.
        '''

        # preprocess
        input_tensor = preprocess(image)
        '''
            Applies the preprocessing pipeline to the image, converting it to a tensor and normalizing the pixel values.
        '''

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)
        '''
            Adds an extra batch dimension to the tensor (unsqueeze(0)), converting the tensor from shape (3, 224, 224) to (1, 3, 224, 224) as expected by the model.
        '''

        # run model
        output = net(input_batch)
        '''
            Passes the input batch through the model to obtain the predictions (output).
        '''


        # Uncomment below 5 lines to print top 10 predictions
        #top = list(enumerate(output[0].softmax(dim=0)))
        #top.sort(key=lambda x: x[1], reverse=True)
        #for idx, val in top[:10]:
        #    print(f"{val.item()*100:.2f}% {classes[idx]}")
        #print(f"========================================================================")
        '''
            These commented-out lines would print the top 10 predicted classes (with their probabilities) if uncommented:

            softmax(dim=0) converts the raw output logits to probabilities.

            top stores the top 10 predictions.

            The for loop prints the top predictions along with their probabilities.
        '''
        
        # log model performance
        frame_count += 1
        '''
            Increments the frame count by 1 for each processed frame.
        '''

        # Records the current time.
        now = time.time()

        # Checks if at least 1 second has passed since the last time the frame rate was logged.
        if now - last_logged > 1:

            # Logs the current frame rate (frames per second) by dividing the number of frames processed by the time elapsed.
            print(f"============={frame_count / (now-last_logged)} fps =================")

            # Resets last_logged to the current time and resets frame_count to 0 after logging the frame rate.
            last_logged = now
            frame_count = 0

