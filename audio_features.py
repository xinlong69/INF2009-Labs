#%% Import the required libraries

import numpy as np
import matplotlib.pyplot as plt
import librosa # Used for speech feature extraction: https://librosa.org/doc/
'''
    numpy: Provides numerical operations and array handling.

    matplotlib.pyplot: Used for plotting various audio features.

    librosa: A Python library for audio and music analysis.
'''

y, sr = librosa.load("test.wav", sr=None) # Save the microphone recording as test.wav 
'''
    librosa.load("test.wav", sr=None): Loads the audio file test.wav.

        y: Stores the audio time series as a NumPy array.

        sr: Stores the sample rate (number of samples per second).

        sr=None: Retains the original sampling rate of the file.
'''

#%% Compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))
'''
    librosa.stft(y): Computes the Short-Time Fourier Transform (STFT), which converts the audio signal into the frequency domain.

    librosa.magphase(...): Separates the STFT result into:

        S_full: The magnitude spectrogram (amplitude of frequencies).

        phase: The phase information.
'''


#%% Plot the time series and the frequency-time plot (spectrogram)
# Creates a figure with two subplots (ax1 and ax2) for waveform and spectrogram.
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# ax1.plot(y): Plots the raw waveform of the audio signal.
# Labels and title are set accordingly.
ax1.plot(y)
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')

# librosa.amplitude_to_db(S_full, ref=np.max): Converts the magnitude spectrogram to decibels.
# librosa.display.specshow(...): Displays the spectrogram with a logarithmic frequency scale.
img = librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax2)

# Adds a color bar to represent intensity.
# Displays the figure.
fig.colorbar(img, ax=ax2)
ax1.set(title='Time Series')
ax2.set(title='Spectrogram')
plt.show()

#%% Chroma Estimation
# Computes the power spectrogram using STFT with an FFT window size of 4096.
S = np.abs(librosa.stft(y, n_fft=4096))**2 # Experiment with # Experiment with 1024, 4096, etc.
'''
    A larger n_fft (i.e 4096) provides better frequency resolution but may reduce time resolution.

    A smaller hop_length increases sensitivity but might introduce noise.
'''

# Extracts chroma features, which represent the energy distribution of musical pitches.
chroma = librosa.feature.chroma_stft(S=S, sr=sr)
'''
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=2048)

    Tune Chroma Feature Extraction

        Adjust window size (n_fft) for better tonal sensitivity.

        Apply filtering techniques to remove irrelevant noise.

'''

# Creates two vertically aligned subplots.
fig, ax = plt.subplots(nrows=2, sharex=True)

# Displays the power spectrogram in decibels.
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].label_outer()
ax1.set(title='Power Spectrogram')

# Displays the chroma representation of the audio.
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax2.set(title='Chromogram')

# Renders the plots.
plt.show()

#%% Compute Mel-Spectrogram
S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000) # Try n_mels equals to 64, 128, etc.
'''
    Computes the Mel-Spectrogram, which represents the frequency spectrum using the Mel scale.

        n_mels=128: Uses 128 Mel filter bands. A higher value captures finer details.

        fmax=8000: Upper frequency limit is set to 8000 Hz. Adjusting this helps isolate movement-specific frequencies.
'''

# Converts the Mel-Spectrogram to decibel units.
fig, ax = plt.subplots()
S_mel_dB = librosa.power_to_db(S_mel, ref=np.max)

# Displays the Mel-Spectrogram.
img = librosa.display.specshow(S_mel_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()

#%% Compute MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # Try n_mfcc equals to 20, 40, etc.
'''
    Extracts MFCCs, which capture important speech characteristics.

        n_mfcc=40: Uses 40 coefficients. More coefficients capture finer audio details.

        Apply first and second derivatives (delta and delta-delta) to enhance movement sensitivity.
'''

# Computes the Mel-Spectrogram again.
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Plots the Mel-Spectrogram.
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()

# Displays the MFCC features.
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')
plt.show()