from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.chroma import DeepChromaProcessor

audio_path = 'wav/dorianD.wav'
proc = DBNBeatTrackingProcessor(fps=100)
act = RNNBeatProcessor()(audio_path)
beats = proc(act)

proc_chord = SequentialProcessor([DeepChromaProcessor(), DeepChromaChordRecognitionProcessor()])
chords = proc_chord(audio_path)

# 打印节拍时间点
print("Detected beats:")
for beat in beats:
    print(beat)

# 打印和弦识别结果
print("\nDetected chords:")
for chord in chords:
    print(chord)
