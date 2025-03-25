#!/usr/bin/env python3
# Refer to https://github.com/Uberi/speech_recognition?tab=readme-ov-file#readme
# Demo for speech recognition. You need to speak only after it says Say something
#%% import all necessary libraries
import speech_recognition as sr
import time
import os

'''
    speech_recognition: A Python library for speech recognition.

    time: Used for measuring execution time.

    os: Used for clearing the terminal screen.
'''

#%% Recording from microphone
# obtain audio from the microphone
r = sr.Recognizer() #Initializing the Recognizer class

# Opens the microphone as an audio source.
# Ensures that the microphone is properly closed after recording.
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source) # Important step to identify the ambient noise and hence be silent during this phase
    os.system('clear') 
    print("Say something!")

    # Listens for speech and stores it as an audio object.
    audio = r.listen(source) # Listening from microphone

# Records the start time to measure how long recognition takes.
# recognize speech using Google Speech Recognition
start_time=time.time()  # start time

try:
    # Uses Google’s Speech-to-Text API to convert speech into text.
    # Displays the recognized text.

    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio))

# Handles cases where speech is unclear or not recognized.
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")

# Handles network issues or API failures.
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

print('Time for Google Speech Recognition recognition = {:.0f} seconds'.format(time.time()-start_time))

# recognize speech using Sphinx
start_time=time.time()  # start time
try:
     # Uses CMU Sphinx, an offline speech recognition engine, to transcribe speech.
     print("Sphinx thinks you said " + r.recognize_sphinx(audio))

# Handles unclear speech or recognition failure.
except sr.UnknownValueError:
    print("Sphinx could not understand audio")

# Handles request errors for Sphinx.
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))

# Calculates and prints the time taken for Sphinx to process the audio.
print('Time for Sphinx recognition = {:.0f} seconds'.format(time.time()-start_time))

