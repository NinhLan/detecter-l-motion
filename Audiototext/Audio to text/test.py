#labelling = [(0, 5.9), ('0', 5.9, 7.4), ('1', 7.4, 7.46), ('0', 7.46, 7.82), ('1', 7.82, 10.7), ('0', 10.7, 13.1), ('1', 13.1, 16.1), ('0', 16.1, 20.36)]
#test =('1','2','3')
#print(labelling[1][1])
#print(test[0])
import ffmpeg
test = "Ceci est un test1"+'2'

print(test + 'i')

audio_input = ffmpeg.input('ConversationGroupe.wav')

audio_cut = audio_input.audio.filter('atrim', start=0, duration=5)
audio_output = ffmpeg.output(audio_cut, 'outputfr.wav')
#ffmpeg.run(audio_output)
audio_output.run()
