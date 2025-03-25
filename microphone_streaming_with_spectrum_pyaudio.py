#%% Import the required libraries
import pyaudio # Refer to https://people.csail.mit.edu/hubert/pyaudio/
import struct  # Refer to https://docs.python.org/3/library/struct.html (Used for converting audio read as bytes to int16)
import numpy as np # For numerical computations
import matplotlib.pyplot as plt # For plotting waveforms and frequency spectra
from scipy.fftpack import fft, fftfreq # Refer to https://docs.scipy.org/doc/scipy/tutorial/fft.html (Used for Fourier Spectrum to display audio frequencies)
import time # In case time of execution is required

'''
    pyaudio: Provides audio streaming functionalities.

    struct: Converts raw byte data from the microphone into integer values.

    numpy: Used for numerical calculations, such as handling audio signals.

    matplotlib.pyplot: Visualizes the audio waveform and its frequency spectrum.

    scipy.fftpack: Performs Fast Fourier Transform (FFT) to analyze frequency components.

    time: Measures execution time for performance analysis.
'''

#%% Parameters
BUFFER = 1024 * 16           # samples per frame (you can change the same to acquire more or less samples)
FORMAT = pyaudio.paInt16     # audio format (bytes per sample)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
RECORD_SECONDS = 30          # Specify the time to record from the microphone in seconds

'''
    BUFFER: Determines how many audio samples are processed in each frame. A larger buffer provides more data but increases latency.

    FORMAT: Uses 16-bit integers (paInt16) to store audio data.

    CHANNELS: Sets the microphone to mono (1 channel).

    RATE: Defines the sampling rate of 44.1 kHz (standard for audio processing).

    RECORD_SECONDS: Specifies how long to record audio.
'''

#%% create matplotlib figure and axes with initial random plots as placeholder
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# create a line object with random data
x = np.arange(0, 2*BUFFER, 2)       # samples (waveform) # Time-domain sample indices
xf = fftfreq(BUFFER, (1/RATE))[:BUFFER//2]   # Frequency bins for FFT

line, = ax1.plot(x,np.random.rand(BUFFER), '-', lw=2) # Placeholder for waveform
line_fft, = ax2.plot(xf,np.random.rand(BUFFER//2), '-', lw=2) # Placeholder for spectrum

'''
    Creates a figure with two subplots:

        ax1: Displays the waveform in the time domain.

        ax2: Displays the frequency spectrum using FFT.

    x: Defines sample points in the time domain.

    xf: Defines frequency bins for the FFT output.

    Initializes two line plots (line and line_fft) with random placeholder data.
'''

# basic formatting for the axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-5000, 5000) # change this to see more amplitude values (when we speak)
ax1.set_xlim(0, BUFFER)

ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000) 
ax2.set_xlim(0, RATE/2)

'''
    Waveform Plot (ax1):

        Y-axis represents volume/amplitude.

        X-axis represents sample indices.

        Limits are set to ensure visibility.

    Frequency Spectrum (ax2):

        X-axis represents frequency (0 to Nyquist frequency, RATE/2).

        Y-axis represents log-magnitude values.
'''

# Do not show the plot yet
# Displays the plot without blocking execution so it can update dynamically.
plt.show(block=False)

#%% Initialize the pyaudio class instance
audio = pyaudio.PyAudio()

# stream object to get data from microphone
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=BUFFER
)

'''
    PyAudio(): Initializes an instance of PyAudio.

    audio.open():

        Opens a stream for both input (recording from the microphone) and output.

        Uses 16-bit PCM format, single-channel audio, and a buffer of 16,384 samples.
'''

# Initializes an empty list exec_time to store execution times.
print('stream started')
exec_time = []

# Loops through each buffer-sized audio frame.
# Loops for a duration of RECORD_SECONDS (20 seconds).
# The number of iterations is calculated based on the sample rate and buffer size.
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):   
       
    # binary data
    data = stream.read(BUFFER)  
   
    # convert data to 16bit integers
    data_int = struct.unpack(str(BUFFER) + 'h', data)    
    
    # compute FFT    
    start_time=time.time()  # for measuring frame rate
    yf = fft(data_int)
    
    # calculate time of execution of FFT
    exec_time.append(time.time() - start_time)
    
    #update line plots for both axes
    line.set_ydata(data_int)
    line_fft.set_ydata(2.0/BUFFER * np.abs(yf[0:BUFFER//2]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    
audio.terminate()
   
print('stream stopped')
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time)*1000))