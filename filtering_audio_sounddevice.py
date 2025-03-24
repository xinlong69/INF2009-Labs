'''
    This script:

        Captures live audio from a microphone.

        Applies a bandpass filter to isolate frequencies between 19.4 kHz and 19.6 kHz.

        Plots the raw and filtered waveforms in real-time.

        Measures and prints the execution time for each frame.
'''

#%% Import the required libraries
import sounddevice as sd # Refer to https://python-sounddevice.readthedocs.io/en/0.4.6/
import numpy as np 
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, sosfilt # Refer to https://docs.scipy.org/doc/scipy/reference/signal.html (Used for Bandpass filtering)
import time # In case time of execution is required  

'''
    sounddevice: Used to record audio from a microphone.

    numpy: Helps in numerical operations like array manipulation.

    matplotlib.pyplot: Used to visualize the audio waveform.

    time: Measures execution time for performance evaluation.

    scipy.signal: Used for designing and applying a bandpass filter.
'''

#%% Parameters
BUFFER = 1024 * 16          # samples per frame (you can change the same to acquire more or less samples)
CHANNELS = 1                # single channel for microphone
RATE = 44100                # samples per second
RECORD_SECONDS = 20         # Specify the time to record from the microphone in seconds

'''
    BUFFER: Defines the number of samples captured per frame.

    CHANNELS: Specifies that the audio input is mono.

    RATE: The number of audio samples per second (standard for high-quality audio).

    RECORD_SECONDS: Defines the duration for which the audio will be recorded.
'''

#%% create matplotlib figure and axes with initial random plots as placeholder
# Creates two subplots (ax1 and ax2) to visualize raw and filtered audio signals.
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))

# create a line object with random data
# Generates a range of indices for plotting the waveform.
x = np.arange(0, 2*BUFFER, 2)       # samples (waveform)

# Initializes two line plots with random values as placeholders. 
line, = ax1.plot(x,np.random.rand(BUFFER), '-', lw=2)
line_filter, = ax2.plot(x,np.random.rand(BUFFER), '-', lw=2)

# basic formatting for the axes
# Configures plot titles, labels, and axis limits.
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
# Displays the plot but allows the script to continue execution.
plt.show(block=False)

##%% Function for design of filter
# This function designs a bandpass filter using the butter function from SciPy.
def design_filter(lowfreq, highfreq, fs, order=3):
    nyq = 0.5*fs # Nyquist frequency (half of the sampling rate)
    low = lowfreq / nyq # Normalized low cutoff frequency
    high = highfreq/nyq # # Normalized high cutoff frequency
    sos = butter(order, [low,high], btype='band',output='sos') # Design a bandpass filter
    return sos

# design the filter
'''
    Designs a bandpass filter with:

    Low cutoff frequency: 19.4 kHz

    High cutoff frequency: 19.6 kHz

    Sampling frequency: 48 kHz

    Filter order: 3
'''
sos = design_filter(19400, 19600, 48000, 3) #change the lower and higher freqcies according to choice

# List to store execution time for each loop iteration
# Initializes a list to track execution time per frame.
exec_time = []

# Loops for the required number of iterations to cover RECORD_SECONDS.
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):   
       
    # Record the sound in int16 format and wait till recording is done
    '''
        Records audio using sounddevice.rec().

        frames=BUFFER: Number of samples per recording.

        samplerate=RATE: Defines sample rate (Hz).

        channels=CHANNELS: Mono recording.

        dtype='int16': Records in 16-bit integer format.

        blocking=True: Waits until recording is complete.
    '''
    data = sd.rec(frames=BUFFER,samplerate=RATE,channels=CHANNELS,dtype='int16',blocking=True)

    # Removes redundant dimensions from the recorded data.
    data = np.squeeze(data)      
    
    # Bandpass filtering
    start_time=time.time()  # for measuring frame rate
    yf = sosfilt(sos, data) # Apply bandpass filtering
    
    # calculate average frame rate
    exec_time.append(time.time() - start_time)
    
    #update line plots for both axes
    line.set_ydata(data) # Update raw audio waveform
    line_filter.set_ydata(yf) # Update filtered waveform
    fig.canvas.draw() # Redraw plot
    fig.canvas.flush_events() # Process UI events
    
print('stream stopped')

# Prints the average time taken to process each frame in milliseconds.
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time)*1000))
# %%
