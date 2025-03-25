#%% Import the required libraries
import sounddevice as sd # Refer to https://python-sounddevice.readthedocs.io/en/0.4.6/
import numpy as np # Numerical computations
import matplotlib.pyplot as plt # Visualization
import time # In case time of execution is required  

'''
    sounddevice: Used to record audio from a microphone.

    numpy: Provides support for numerical operations such as array manipulation.

    matplotlib.pyplot: Used to plot and visualize the audio waveform.

    time: Used for measuring execution time.
'''

#%% Parameters
BUFFER = 1024 * 16           # samples per frame (you can change the same to acquire more or less samples)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
RECORD_SECONDS = 30          # Specify the time to record from the microphone in seconds

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
xf = np.fft.fftfreq(BUFFER,1/RATE)[:BUFFER//2]

# Initializes two line plots with random values as placeholders.
line, = ax1.plot(x,np.random.rand(BUFFER), '-', lw=2)
line_fft, = ax2.plot(xf,np.random.rand(BUFFER//2), '-', lw=2)

# basic formatting for the axes
# Configures plot titles, labels, and axis limits.
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

# Do not show the plot yet
# Displays the plot but allows the script to continue execution.
plt.show(block=False)

#%% Reconrding the sound and constructing the spectrum
# Initializes a list to track execution time per frame.
exec_time = []

# Loops for the required number of iterations to cover RECORD_SECONDS.
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):

    # Record the sound in int16 format and wait till recording is done
    data = sd.rec(frames=BUFFER,samplerate=RATE,channels=CHANNELS,dtype='int16',blocking=True)
    '''
        Records audio using sounddevice.rec().

        frames=BUFFER: Number of samples per recording.

        samplerate=RATE: Defines sample rate (Hz).

        channels=CHANNELS: Mono recording.

        dtype='int16': Records in 16-bit integer format.

        blocking=True: Waits until recording is complete.
    '''

    # Removes redundant dimensions from the recorded data.
    data = np.squeeze(data)

    # compute FFT    
    start_time=time.time()  # for measuring frame rate

    # Computes the Fast Fourier Transform (FFT) of data, which is typically an array representing audio signal samples.
    fft_data = np.fft.fft(data)
    '''
        This transforms the signal from the time domain to the frequency domain.
    '''

    # Takes the magnitude of the FFT output.
    fft_data = np.abs(fft_data[:BUFFER//2])
    '''
        The FFT result is symmetrical around the midpoint, so only the first half (BUFFER//2) is retained to remove redundancy.
    '''
    
    # calculate time of execution of FFT
    exec_time.append(time.time() - start_time)
    
    # update line plots for both axes
    line.set_ydata(data) # Update raw audio waveform
    '''
        Updates the line plot with the original data values.

        This represents the raw audio waveform before processing.
    '''
    line_fft.set_ydata(fft_data) # Update filtered waveform
    '''
        Updates line_fft with the computed fft_data.

        However, this is immediately overwritten in the next line.
    '''

    # Scales the FFT output by a factor of 2.0/BUFFER, ensuring proper normalization of the frequency spectrum.
    # Overwrites the previous fft_data update.
    line_fft.set_ydata(2.0/BUFFER * fft_data)

    # Forces the plot to update immediately after modifying the y-axis data.
    fig.canvas.draw() # Redraw plot

    # Ensures that the UI processes any pending events, preventing the plot from freezing or becoming unresponsive.
    fig.canvas.flush_events() # Process UI events

  
print('stream stopped')

# Prints the average time taken to process each frame in milliseconds.
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time)*1000))