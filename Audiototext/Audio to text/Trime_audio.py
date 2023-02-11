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

                                                                Ce programme permet de séquencer un fichier Audio en plusieurs audio
                                                                à savoir un pour chaque prise de parole.
"""


import ffmpeg  

#La liste correspond à la sortie du speaker diarisation

Liste_test = [('1', 0, 2.06), ('0', 2.06, 3.74), ('1', 3.74, 8.72), ('0', 8.72, 11.78), ('1', 11.78, 16.4), ('0', 16.4, 19.88), ('1', 19.88, 25.16), ('0', 25.16, 29.48), ('1', 29.48, 31.58), ('0', 31.58, 35.42), ('1', 35.42, 36.56), ('0', 36.56, 37.28)]


audio_input = ffmpeg.input('Suzuki.wav')

n = 1 #Compteur pour les fichiers audios du locuteur A
m = 1 #Compteur pour les fichiers audios du locuteur B

for i in range(0,len(Liste_test)):
    if Liste_test[i][0] == '1': #On identifie ici le locuteur
        
        start1 = Liste_test[i][1] #On prend le temps de début de la prise de parole
        end = Liste_test[i][2] # On prend le temps de fin de la prise de parole
        duration1 = end - start1
        audio_cut = audio_input.audio.filter('atrim', start=start1, duration=duration1)
        audio_output = ffmpeg.output(audio_cut, 'A/'+'output'+str(n)+'A.wav')
        #ffmpeg.run(audio_output)
        audio_output.run()
        n += 1


    elif Liste_test[i][0] == '0':
        start1 = Liste_test[i][1]
        end = Liste_test[i][2]
        duration1 =end - start1
        audio_cut = audio_input.audio.filter('atrim', start=start1, duration=duration1)
        audio_output = ffmpeg.output(audio_cut, 'B/'+'output'+str(m)+'B.wav')
        #ffmpeg.run(audio_output)
        audio_output.run()
        m += 1 

"""
audio_cut = audio_input.audio.filter('atrim', start=14.72, duration=20.24-14.72)
audio_output = ffmpeg.output(audio_cut, 'output.mp3')
#ffmpeg.run(audio_output)
audio_output.run()


#ffmpeg -i cut.ffconcat -codec copy Conversation2SPaceB.mp3

"""
