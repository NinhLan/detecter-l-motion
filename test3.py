import librosa
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
import joblib



def extract_features(filename):
    audio, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_emotion(filename):
    features = extract_features(filename)
    scaler = preprocessing.StandardScaler().fit(features.reshape(-1, 1))
    features = scaler.transform(features.reshape(-1, 1))

    # Load the trained SVM model
    model = joblib.load("emotion_detection_model.pkl")

    # Predict the emotion using the extracted features
    emotion = model.predict(features.reshape(1, -1))[0]

    return emotion

if __name__ == "__main__":
    filename = "C:/Users/remde/Downloads/Emotions/Sad/03-01-04-01-01-01-02.wav"
    emotion = predict_emotion(filename)
    print("Emotion detected in the audio: ", emotion)