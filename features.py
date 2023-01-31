import librosa
import pandas as pd
import numpy as np
import glob

def extract_features(filename, length=167):
    # Chargement du fichier audio
    y, sr = librosa.load(filename)
    
    # Extraction des caractéristiques
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Adapter la taille des caractéristiques
    features = librosa.util.fix_length(features, axis=1, size=40)
    #features = librosa.util.fix_length(features, length, axis=1)
    
    return features

def main(filenames):
    # Création d'une liste pour stocker les caractéristiques de chaque fichier audio
    features_list = []
    
    # Boucle sur chaque fichier audio
    for filename in filenames:
        # Extraction des caractéristiques
        features = extract_features(filename)
        
        # Stockage des caractéristiques dans la liste
        features_list.append(features)
    
    # Conversion de la liste en DataFrame
    df = pd.DataFrame(np.vstack(features_list))
    #df = pd.DataFrame(features_list)
    
    # Ajout des étiquettes d'émotion à la colonne d'étiquette
    df_len = len(df)
    #emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    emotions = ['Positive', 'Negative', 'Neutre']
    df['emotion'] = [emotions[i % len(emotions)] for i in range(df_len)]
    #df['emotion'] = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    # Enregistrement du DataFrame en tant que fichier CSV
    df.to_csv("C:/Users/remde/Downloads/features1.csv", index=False)
    
    print("Features extracted and saved to features1.csv")

if __name__ == "__main__":
    # Liste de noms de fichiers audio
    #filenames = ["C:/Users/remde/Downloads/Emotions/Angry/03-01-05-01-01-01-02.wav", "C:/Users/remde/Downloads/Emotions/Disgusted/03-01-07-01-01-01-02.wav","C:/Users/remde/Downloads/Emotions/Fearful/03-01-06-01-01-01-02.wav","C:/Users/remde/Downloads/Emotions/Happy/03-01-03-01-01-01-02.wav", "C:/Users/remde/Downloads/Emotions/Neutral/03-01-01-01-01-01-02.wav", "C:/Users/remde/Downloads/Emotions/Sad/03-01-04-01-01-01-02.wav", "C:/Users/remde/Downloads/Emotions/Suprised/03-01-08-01-01-01-02.wav"]
    emotion_folders = ["C:/Users/remde/Downloads/emotionlearn/Positive",
                    "C:/Users/remde/Downloads/emotionlearn/Negative",
                    "C:/Users/remde/Downloads/emotionlearn/Neutre"]

    filenames = []

    # Boucle sur chaque dossier d'émotion
    for emotion_folder in emotion_folders:
        # Ajout des chemins de fichier pour chaque fichier audio dans le dossier
        filenames += glob.glob(emotion_folder + "/*.wav")
    # Exécution de la fonction principale
    main(filenames)