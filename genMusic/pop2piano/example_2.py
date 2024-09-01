# Example using your own audio file
import time

import librosa
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

audio, sr = librosa.load("/Users/apple/Music/网易云音乐/戴荃 - 悟空 [music].mp3", sr=44100)  # feel free to change the sr to a suitable value.
# audio, sr = librosa.load("/Users/apple/Music/网易云音乐/戴荃 - 悟空 [vocals].mp3", sr=44100)  # feel free to change the sr to a suitable value.
# audio, sr = librosa.load("/Users/apple/Desktop/pythonProject/torchL/genMusic/musicLab/fromNetEase/xunzhang.wav", sr=44100)  # feel free to change the sr to a suitable value.
model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
tokenizer_output = processor.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)["pretty_midi_objects"][0]
tokenizer_output.write("outputs/midi_output_2_"+ str(time.time()) +".mid")
