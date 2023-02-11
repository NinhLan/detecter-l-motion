import warnings
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['neutral', 'happy', 'sad']

#DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\ninht\\Downloads\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
    
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#DataFlair - Train the model
model.fit(x_train,y_train)
clf = RandomForestClassifier()
clf.fit(x_train, y_train)
y_pred1=clf.predict(x_test)


#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
accuracy1=accuracy_score(y_true=y_test, y_pred=y_pred1)
conf_matrix = confusion_matrix(y_test, y_pred, labels=observed_emotions)

#DataFlair - Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
print("Accuracy1: {:.2f}%".format(accuracy1*100))

#DataFlair - Calculate the accuracy of each emotion
neutral_accuracy = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1] + conf_matrix[0, 2])
happy_accuracy = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1] + conf_matrix[1, 2])
sad_accuracy = conf_matrix[2, 2] / (conf_matrix[2, 0] + conf_matrix[2, 1] + conf_matrix[2, 2])

def predict_emotion(file_name):
    feature = extract_feature(file_name, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1) # reshape to match the model's input shape
    prediction = model.predict(feature)
    return prediction[0]

# DataFlair - Predict the emotion of a specific audio file
file_name = 'C:/Users/ninht/Downloads/emotion_entrain/Sad/03-01-04-01-01-01-01.wav'
emotion = predict_emotion(file_name)
#DataFlair - Print the accuracy of each emotion
print("Accuracy of Neutral: {:.2f}%".format(neutral_accuracy * 100))
print("Accuracy of Happy: {:.2f}%".format(happy_accuracy * 100))
print("Accuracy of Sad: {:.2f}%".format(sad_accuracy * 100))
print(f'Emotion: {emotion}')
emotions = ['neutral', 'happy', 'sad']
percents = [neutral_accuracy, happy_accuracy,sad_accuracy ]
plt.bar(emotions, percents)
plt.ylabel('Percentage')
plt.xlabel('Emotion')
plt.title('Percentage of Correct Predictions by Emotion')

plt.show()