'''
    This script:

        Captures live audio from a microphone.

        Applies a bandpass filter to isolate frequencies between 19.4 kHz and 19.6 kHz.

        Plots the raw and filtered waveforms in real-time.

        Measures and prints the execution time for each frame.
'''

#%% Import the required libraries
import pyaudio # Refer to https://people.csail.mit.edu/hubert/pyaudio/
import struct  # Refer to https://docs.python.org/3/library/struct.html (Used for converting audio read as bytes to int16)
import numpy as np # Numerical computations
import matplotlib.pyplot as plt # Visualization
from scipy.signal import butter, sosfilt # Refer to https://docs.scipy.org/doc/scipy/reference/signal.html (Used for Bandpass filtering)
import time # In case time of execution is required  

'''
    pyaudio: Used for capturing and playing back audio.

    struct: Converts binary audio data into numerical format.

    numpy: Provides support for numerical operations such as array manipulation.

    matplotlib.pyplot: Used to plot and visualize the audio waveform.

    scipy.signal: Provides signal processing functions; here, it is used for designing and applying a bandpass filter.

    time: Used for measuring execution time.
'''

#%% Parameters
BUFFER = 1024 * 16          # samples per frame (you can change the same to acquire more or less samples)
FORMAT = pyaudio.paInt16    # audio format (bytes per sample)
CHANNELS = 1                # single channel for microphone
RATE = 44100                # samples per second
RECORD_SECONDS = 20         # Specify the time to record from the microphone in seconds

'''
    BUFFER: Determines how much audio data is processed at once. Larger buffers result in higher latency but more stable processing.

    FORMAT: Specifies that the audio samples are stored as 16-bit integers.

    CHANNELS: Uses a single channel (mono) audio input.

    RATE: The sample rate is set to 44.1 kHz, which is the standard for audio processing.

    RECORD_SECONDS: The total duration of the recording.
'''

#%% create matplotlib figure and axes with initial random plots as placeholder
# Creates a figure with two subplots (ax1 for raw audio, ax2 for filtered audio).
# Each subplot will display a waveform.
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# create a line object with random data
# x: Defines an array for sample indices.
x = np.arange(0, 2*BUFFER, 2)       # samples (waveform)

# line, line_filter: Initialize two plots with random values as placeholders.
line, = ax1.plot(x,np.random.rand(BUFFER), '-', lw=2)
line_filter, = ax2.plot(x,np.random.rand(BUFFER), '-', lw=2)

# basic formatting for the axes
# Configures axis labels, titles, and limits for better visualization.
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('amplitude')
ax1.set_ylim(-5000, 5000) # change this to see more amplitude values (when we speak)
ax1.set_xlim(0, BUFFER)

ax2.set_title('FILTERED')
ax2.set_xlabel('samples')
ax2.set_ylabel('amplitude')
ax2.set_ylim(-5000, 5000) 
ax2.set_xlim(0, BUFFER)

# show the plot
# Displays the plot without blocking program execution.
plt.show(block=False)

#%% Function for design of filter
# Creates a bandpass filter that allows only frequencies within the specified range.
# Uses Butterworth filter of order 3 for smooth filtering.

# Returns second-order sections (sos) format, which is numerically stable.
def design_filter(lowfreq, highfreq, fs, order=3):
    nyq = 0.5*fs
    low = lowfreq/nyq
    high = highfreq/nyq
    sos = butter(order, [low,high], btype='band',output='sos')
    return sos

# design the filter
# Creates a bandpass filter that allows frequencies between 19.4 kHz and 19.6 kHz.
# Sampling rate fs = 48 kHz.
sos = design_filter(19400, 19600, 48000, 3) #change the lower and higher freqcies according to choice


#%% Initialize the pyaudio class instance
# Initializes the PyAudio object to manage the audio stream.
audio = pyaudio.PyAudio()

# stream object to get data from microphone
# Opens a stream for real-time audio processing.
'''
Uses:
    FORMAT: 16-bit integer format.

    CHANNELS: Single-channel input.

    RATE: 44.1 kHz sampling rate.

    input=True: Enables microphone input.

    output=True: Allows playback of the processed audio.

    frames_per_buffer: Specifies how many samples are processed at a time.
'''
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=BUFFER
)

# Prints a message indicating that the audio stream has started.
# Initializes an empty list exec_time to store execution times.
print('stream started')
exec_time = []

# Loops for a duration of RECORD_SECONDS (20 seconds).
# The number of iterations is calculated based on the sample rate and buffer size.
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):   
       
    # binary data
    # Reads BUFFER samples from the audio stream.
    data = stream.read(BUFFER)  
   
    # convert data to 16bit integers
    # Converts binary audio data into an array of 16-bit integers.
    data_int = struct.unpack(str(BUFFER) + 'h', data)    
    
    # Bandpass filtering
    # Records the start time to calculate processing speed.
    start_time=time.time()  # for measuring frame rate

    # Applies bandpass filtering using sosfilt().
    yf = sosfilt(sos, data_int)
    
    # calculate average frame rate
    # Measures the execution time of filtering and stores it in exec_time.
    exec_time.append(time.time() - start_time)
    
    #update line plots for both axes
    '''
        Updates the plot in real-time with new audio data.

        line.set_ydata(data_int): Updates waveform plot.

        line_filter.set_ydata(yf): Updates filtered signal plot.

        fig.canvas.draw(): Redraws the figure.

        fig.canvas.flush_events(): Ensures the plot updates without freezing.
    '''
    line.set_ydata(data_int)
    line_filter.set_ydata(yf)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Stops and releases the audio resources.    
audio.terminate()
print('stream stopped')

# Computes the average execution time per frame and prints it.
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time)*1000))