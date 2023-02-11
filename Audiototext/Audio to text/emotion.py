"""
                    ____
                  ;`    `'-._
                 / \        /\
               /`   \      |  ;
              /      \     |  |
             /        `\   |  |
            /           \_ /  |
           ;            / `\  |
          ,|_  __       \__/  |
          _\_o/_(             |_
         /`"=/\==""=="=="=="=="`\
         |   )/                 |
         \                      /
         /';=""==""==""==""==";`\
         |  /`   /~\  /~\   `\  |
         |  \  _ \o/  \o/ _  /  |
         \  ; (_)   __   (_) ;  /
         /  |\_.-""(__)""-._/|  \
        |   \       /\       /   |
       /     '.___.'__'.___.'     \
      |             \/             |
      |                            |
      \                            /
      |                            |
       \                          /
        '.                      .'
          '-.__            __.-'
               '---'--'---'


                                                                            Ce programme permet de faire l'analyse d'émotions d'un fichier texte
                                                                            On vient également créer un fichier texte pour chaque prise de parole
                                                                            on enregistre également les valeurs des émitoins, pour voir celles qui prédominent.  
"""



from speech_recognition import Recognizer, Microphone
import speech_recognition as sr
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import matplotlib.pyplot as plt
import numpy as np
 
locuteurs = ['Locuteur A', 'Locuteur B']


r = sr.Recognizer()
Moyenne_A = [] #On va ici stocker chacune des valeurs des émotions du locuteur A enfin d'en faire la moyenne
Moyenne_B = [] #On va ici stocker chacune des valeurs des émotions du locuteur B enfin d'en faire la moyenne

# Locuteur A

for i in range (1,6):
    harvard = sr.AudioFile('V:\Audiototext\Audio to text/A/'+'output'+str(i)+'A.wav')
    with harvard as source:
      audio = r.record(source)
    
    try:    
            text = r.recognize_google(
                    audio, 
                    language="fr-FR"
                )
                
            with open("V:\Audiototext\Audio to text/A/TextA/data"+str(i)+"txt", "w") as txtfile:
                print("{}".format(text), file=txtfile)  

            #print("Le texte a été bien enregistré dans le fichier data.txt")
    except Exception as ex:
            print(ex)

        #Partie emotion

    with open("V:\Audiototext\Audio to text/A/TextA/data"+str(i)+"txt") as d:
            lines = d.read() ##Assume the sample file has 3 lines
            first = lines.split('\n', 1)[0]

    #a= print("vous dites:{}".format(first))

    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

    blob1 = tb(u"{}".format(first))
    phrase = blob1.sentiment
    emotion = phrase[0]
    #print(emotion)
    """if emotion > 0:
            print("La phrase indiqué est une phrase joyeuse")
    elif emotion < 0:
            print("La phrase indiqué est une phrase triste")
    else:
            print("La phrase indiqué est une phrase sans emotion")"""
    Moyenne_A.append(emotion)



#Locuteur_B 
for i in range (1,6):
    harvard = sr.AudioFile('V:\Audiototext\Audio to text/B/'+'output'+str(i)+'B.wav')
#harvard = sr.AudioFile('A\output7outputA.wav')
    with harvard as source:
        audio = r.record(source)

    try:
        
        text = r.recognize_google(
                audio, 
                language="fr-FR"
            )
            
        with open("V:\Audiototext\Audio to text/B/TextB/data"+str(i)+"txt", "w") as txtfile:
            print("{}".format(text), file=txtfile)  

        #print("Le texte a été bien enregistré dans le fichier data.txt")
    except Exception as ex:
        print(ex)

    #Partie emotion

    with open("V:\Audiototext\Audio to text/B/TextB/data"+str(i)+"txt") as d:
        linesB = d.read() ##Assume the sample file has 3 lines
        firstB = linesB.split('\n', 1)[0]

    #b= print("vous dites:{}".format(first))

    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

    blob2 = tb(u"{}".format(first))
    phraseB = blob2.sentiment
    emotionB = phrase[0]
    #print(emotionB)
    """if emotionB > 0:
        print("La phrase indiqué est une phrase joyeuse")
    elif emotionB < 0:
        print("La phrase indiqué est une phrase triste")
    else:
        print("La phrase indiqué est une phrase sans emotion")"""
    Moyenne_B.append(emotionB)
A1 = []
A2 = []
A3 = []    

#print(Moyenne_A)
#print(Moyenne_B)

print(Moyenne_A)
moyenne = sum(Moyenne_A)/len(Moyenne_A)
moyenneB = sum(Moyenne_B)/len(Moyenne_B)

print("Concernant A : ")
if moyenne > 0:
        print("La Moyenne des émotions est de "+ str(moyenne) +" c'est donc une prise de parole majoritairement joyeuse")
elif moyenne < 0:
        print("La Moyenne des émotions est de "+ str(moyenne) +" c'est donc une prise de parole majoritairement triste")
else:
        print("La Moyenne des émotions est de "+ str(moyenne) +" c'est donc une prise de parole majoritairement sans emotion")

print("Concernant B : ")
if moyenneB > 0:
        print("La Moyenne des émotions est de "+ str(moyenneB) +" c'est donc une prise de parole majoritairement joyeuse")
elif moyenneB < 0:
        print("La Moyenne des émotions est de "+ str(moyenneB) +" c'est donc une prise de parole majoritairement triste")
else:
        print("La Moyenne des émotions est de "+ str(moyenneB) +" c'est donc une prise de parole majoritairement sans emotion")

# Calcul des pourcentages d'émotions pour chaque locuteur
locuteur_A = [0, 0, 0]
locuteur_B = [0, 0, 0]

for emotion in Moyenne_A:
    if emotion > 0:
        locuteur_A[0] += 1
    elif emotion < 0:
        locuteur_A[1] += 1
    else:
        locuteur_A[2] += 1

for emotion in Moyenne_B:
    if emotion > 0:
        locuteur_B[0] += 1
    elif emotion < 0:
        locuteur_B[1] += 1
    else:
        locuteur_B[2] += 1

# Normalisation des pourcentages
n = len(Moyenne_A)
locuteur_A = [x / n for x in locuteur_A]
n = len(Moyenne_B)
locuteur_B = [x / n for x in locuteur_B]

# Affichage du diagramme en secteurs pour chaque locuteur
labels = ['Positif', 'Négatif', 'Neutre']
x = np.arange(len(labels))

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.2, locuteur_A, 0.2, label='Locuteur A')
rects2 = ax.bar(x, locuteur_B, 0.2, label='Locuteur B')

ax.set_ylabel('Pourcentage')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}%'.format(height * 100),
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha='center', va='bottom')

        
autolabel(rects1)
autolabel(rects2)

plt.show()