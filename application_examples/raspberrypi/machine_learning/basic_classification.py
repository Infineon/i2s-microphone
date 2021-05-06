import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from scipy import fft
import pickle
import pandas as pd
from sklearn import preprocessing # Preprocessing module to normalize X data
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy
from sklearn.model_selection import cross_val_score
import librosa

# These values can be adapted according to your requirements.
samplerate = 48000
downsample = 2
dsr = int(samplerate/downsample)
input_gain_db = 12
device = 'snd_rpi_i2s_card'
classes = ['clap', 'snip', 'other']
samples_per_class = 50
sample_duration = 1
output_folder = "wav_output"

init_time = 5
prepare_time = 8
gap_time = 2

def butter_highpass(cutoff, fs, order=5):
    '''
    Helper function for the high-pass filter.
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    '''
    High-pass filter for digital audio data.
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
    rec_duration = init_time + ((sample_duration+gap_time) * samples_per_class + prepare_time) * len(classes)

    rec = sd.rec(int(rec_duration * samplerate), samplerate=samplerate, channels=2)
    print("Recording started, BUT WAIT - give the microphone a bit time to settle...")
    sd.sleep(int(init_time * 1000))
    # Record sample data
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
    
    # Wait until the recording is done.
    sd.wait()
    
    # Process the audio data as explained above.
    print("Processing...")
    processed = process_audio_data(rec)
    print("Done.")

    return processed

def save_samples(recording):
    global init_time, prepare_time, gap_time
    # Generate samples from audio recording and save them.
    s = dsr
    start_offset = init_time * s
    
    samples = {}
    for cls in classes:
        samples[cls] = []
        start_offset += int(prepare_time * s)
        for i in range(samples_per_class):
            sample = recording[start_offset:start_offset+int(sample_duration*s)]
            samples[cls].append(sample)
            start_offset += int((sample_duration + gap_time) * s) 
    pickle.dump(samples, open('samples.p', 'wb'))
    return samples

def save_samples_as_wav(samples):
    for cls in samples:
        for i,sample in enumerate(samples[cls]):
            write(str(output_folder) + "/" + str(cls) + '-' + str(i) + ".wav", dsr, sample)

def generate_features(samples):
    features = {}
    for cls in samples:
        features[cls] = []
        for sample in samples[cls]:
            mfcc_l = librosa.feature.mfcc(sample[:,0], sr=dsr)
            mfcc_r = librosa.feature.mfcc(sample[:,1], sr=dsr)
            features[cls].append(mfcc_l[0] + mfcc_r[0])
    return features

def features_to_dataframe(features):
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
    global samples_available
    try:
        samples = pickle.load(open('samples.p', 'rb'))
        samples_available = True
    except:
        samples = None
        samples_available = False
    return samples

samples = load_samples()

if not samples_available:
    print('No sample file found - generating new one...')
    rec = record_samples()
    samples = save_samples(rec)

#save_samples_as_wav(samples)
features = generate_features(samples)

df = features_to_dataframe(features)
print(df)

# Define our features X as all columns of Xy excluding the label column.
X = df.drop(['label'], axis='columns')
# Define the label column of Xy as our label.
y = df.label

# Normalize X
X = preprocessing.normalize(X, norm='l2')

# Split X and y into training and testing data each.
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Prepare the classifier.
model = MLPClassifier(solver='lbfgs', max_iter=10000)

# Train the classifier (might take some minutes).
print("Start training...")
#model.fit(X_train, y_train)

# Predict values with the trained model.
#y_pred = model.predict(X_test)

scores = cross_val_score(model, X, y, cv=5)
print("Scores: ")
print(scores)

# Evaluate the prediction performance using the accuracy metric and print the result.
#score = accuracy(y_test, y_pred)
print("Accuracy: " + str(scores.mean()))
