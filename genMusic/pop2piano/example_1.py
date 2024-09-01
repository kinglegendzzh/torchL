# pip install pretty-midi==0.2.9 essentia==2.1b6.dev1034 librosa scipy resampy
# Example using HuggingFace Dataset
import time

from datasets import load_dataset
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

inputs = processor(
    audio=ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="pt"
)
model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
tokenizer_output = processor.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)["pretty_midi_objects"][0]
tokenizer_output.write("outputs/midi_output"+ str(time.time()) +".mid")