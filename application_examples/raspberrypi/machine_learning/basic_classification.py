#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This example shows how to record audio samples and use these samples to implement a basic
Machine Learning based classification using Neural Networks.
The default setting and sample data is meant to distinguish clap and snap sounds.
"""

"""
Dependencies: please check https://github.com/Infineon/i2s-microphone/wiki/Raspberry-Pi-Audio-Machine-Learning-with-Python
"""

import sounddevice as sd # Python-sounddevice is needed to access our I2S microphone directly from Python.
import numpy as np # Numpy is a standard tool to work with numbers and arrays in Python.
from scipy import signal # This is needed for the high-pass filter.
import pickle # We use Pickle to save and load our sample data.
import pandas as pd # Helps to represent the data in the correct format.
from sklearn import preprocessing # Preprocessing module to normalize X data.
from sklearn.model_selection import train_test_split # Automatically split train and test data.
from sklearn.neural_network import MLPClassifier  # The Multi-layer Perceptron classifier.
from sklearn.metrics import accuracy_score as accuracy # Our evaluation metric for this example.
import librosa # A mighty audio analysis library.

'''
The following values can be adapted according to your requirements. These values MUST ONLY be
changed when you want to record new samples (for this please delete the file "samples.p").
'''

# Audio sample rate in samples per second. It is recommended to leave this at 48kHz.
samplerate = 48000

# After recording, the samplerate can be reduced to save processing time.
# "2" stands for taking every 2nd sample.
downsample = 2

# DO NOT CHANGE -> this is the samplerate after downsampling.
dsr = int(samplerate/downsample)

# Here you can change the gain which is applied to your audio data in pre-processing.
# 12dB is recommended for usual environments, 0dB for very load environments and you
# can go up to 24dB (or even more) in very quiet environments.
input_gain_db = 12

# Here you can select the input device. The value below should work for every I2S microphone.
device = 'snd_rpi_i2s_card'

# If you want to record your own audio samples you can name the different classes here.
classes = ['clap', 'snap', 'other']

# Here you can decide how many samples you want to record per class. Be careful, if this
# value is too high your script might use too much memory and be killed by the OS.
samples_per_class = 50

# Duration of each recorded sample in seconds. Same here: Keep short, otherwise processing
# (and recording) will take ages!
sample_duration = 1

# Choose how much time the microphone should record in advance before recording the first
# sample. A value between 5 and 15 seconds is recommended.
init_time = 5

# Decide how much seconds you need BETWEEN DIFFERENT CLASSES while recording your own samples.
prepare_time = 8

# Decide how much seconds you need BETWEEN DIFFERENT SAMPLES while recording your own samples.
gap_time = 2

def butter_highpass(cutoff, fs, order=5):
    '''
    Helper function for the high-pass filter.
    Source: https://stackoverflow.com/a/39032946
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    '''
    High-pass filter for digital audio data.
    Source: https://stackoverflow.com/a/39032946
    '''
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def set_gain_db(audiodata, gain_db):
    '''
    This function allows to set the audio gain
    in decibel. Values above 1 or below -1 are set to
    the max/min values.
    '''
    audiodata *= np.power(10, gain_db/10)
    return np.array([1 if s > 1 else -1 if s < -1 else s for s in audiodata], dtype=np.float32)

def process_audio_data(audiodata):
    '''
    Some basic input processing of the recorded audio data.
    We remove the DC offset by applying a high-pass filter and
    increase the amplitude by setting a positive gain.
    '''
    # Extract mono channels from input data.
    ch1 = np.array(audiodata[::downsample, 0], dtype=np.float32)
    ch2 = np.array(audiodata[::downsample, 1], dtype=np.float32)

    # High-pass filter the data at a cutoff frequency of 10Hz.
    # This is required because I2S microhones have a certain DC offset
    # which we need to filter in order to amplify the volume later.
    ch1 = butter_highpass_filter(ch1, 10, dsr)
    ch2 = butter_highpass_filter(ch2, 10, dsr)

    # Amplify audio data.
    # Recommended, because the default input volume is very low.
    # Due to the DC offset this is not recommended without using
    # a high-pass filter in advance.
    ch1 = set_gain_db(ch1, input_gain_db)
    ch2 = set_gain_db(ch2, input_gain_db)

    # Output the data in the same format as it came in.
    return np.array([[ch1[i], ch2[i]] for i in range(len(ch1))], dtype=np.float32)

def record_samples():
    global init_time, prepare_time, gap_time

    # Calculate the total recording duration. All samples are recorded at once and separated later.
    # This is recommended, because the I2S interface does a loud "knack" sound always when starting
    # a new recording. This is also why we cut the init_time at the beginning.
    rec_duration = init_time + ((sample_duration+gap_time) * samples_per_class + prepare_time) * len(classes)

    # Start the stereo recording.
    rec = sd.rec(int(rec_duration * samplerate), samplerate=samplerate, channels=2)
    print("Recording started, BUT WAIT - give the microphone a bit time to settle...")
    sd.sleep(int(init_time * 1000))
    # Now we go through all classes set above and record audio samples for them.
    for cls in classes:
        print('Get ready to record samples for class "' + str(cls) + '"...')
        sd.sleep(int(prepare_time * 1000))
        for sample in range(samples_per_class):
            print("- RECORDING " + str(sample+1) + "/" + str(samples_per_class) + " -")
            sd.sleep(int(sample_duration * 1000))
            print("- STOP -")
            sd.sleep(int(gap_time * 1000))
    print("- DONE -")
    print("-" * 30)
    
    # Wait until the recording is done. Just to make sure that the recording is ready before
    # accessing it.
    sd.wait()
    
    # Process the audio data as explained above. Might take a while. If the script is killed by
    # your OS in this step try to go with less samples.
    print("Processing...")
    processed = process_audio_data(rec)
    print("Done.")

    return processed

def save_samples(recording):
    global init_time, prepare_time, gap_time
    '''
    This function separates the audio recording to samples and saves them as Pickle file.
    '''

    # Cut the start (because of above mentioned "knack" noise).
    start_offset = init_time * dsr
    
    # Go through the recording and get the recorded samples from the known positions.
    # This is done by indexing the sample rate (recording[from:to]).
    samples = {}
    for cls in classes:
        samples[cls] = []
        start_offset += int(prepare_time * dsr)
        for i in range(samples_per_class):
            sample = recording[start_offset:start_offset+int(sample_duration*dsr)]
            samples[cls].append(sample)
            start_offset += int((sample_duration + gap_time) * dsr)

    # Save the result as Pickle file. If we skip this step we always have to re-record the samples.
    pickle.dump(samples, open('samples.p', 'wb'))
    return samples

def generate_features(samples):
    '''
    Generate features (X-inputs of Machine Learning model) out of the audio samples.
    For this example the Mel Frequency Cepstral Coefficients for both audio channels
    are taken as features.
    '''
    features = {}
    for cls in samples:
        features[cls] = []
        for sample in samples[cls]:
            # Generate MFCCs for left channel.
            mfcc_l = librosa.feature.mfcc(sample[:,0], sr=dsr)
            # Generate MFCCs for right channel.
            mfcc_r = librosa.feature.mfcc(sample[:,1], sr=dsr)
            features[cls].append(mfcc_l[0] + mfcc_r[0])
    return features

def features_to_dataframe(features):
    '''
    Save the features and label (y-output of Machine Learning model) in a Pandas
    DataFrame. This can be easily used as input for a Machine Learning model afterwards.
    '''
    Xy = []
    for label in samples:
        for feature in features[label]:
            row = [classes.index(label)]
            column_names = ["label"]
            for idx,line in enumerate(feature):
                column_names.append('mfcc_l-' + str(idx))
                row.append(line)
                column_names.append('mfcc_r-' + str(idx))
                row.append(line)
            Xy.append(row)
    return pd.DataFrame(Xy, columns=column_names)

def load_samples():
    '''
    Try to load previously saved samples. Trigger re-recording when no saved samples
    are available.
    '''
    global samples_available
    try:
        samples = pickle.load(open('samples.p', 'rb'))
        samples_available = True
    except:
        samples = None
        samples_available = False
    return samples

# --------------------------------------------------------------- #

# Let's start doing actual work: we call the load_samples function to load
# or generate audio samples.
samples = load_samples()

if not samples_available:
    print('No sample file found - generating new one...')
    rec = record_samples()
    samples = save_samples(rec)
else:
    print('Samples are loaded from "samples.p". Delete this file to record new audio samples.')

features = generate_features(samples)

df = features_to_dataframe(features)

# Print the Pandas DataFrame - to get an impression how the input data for our Machine Learning model looks like.
print(df)

# Define our features X as all columns of the DataFrame excluding the label column.
X = df.drop(['label'], axis='columns')
# Define the label column of DataFrame as our label -> this represents one of the classes defined in the beginning.
y = df.label

# Normalize X
X = preprocessing.normalize(X, norm='l2')

# Split X and y into training and testing data each.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Prepare the classifier.
model = MLPClassifier(solver='lbfgs', max_iter=10000)

# Train the classifier (might take some minutes).
print("Start training...")
model.fit(X_train, y_train)

# Predict values with the trained model.
y_pred = model.predict(X_test)

# Evaluate the prediction performance using the accuracy metric and print the result.
score = accuracy(y_test, y_pred)
print("Accuracy: " + str(score))
