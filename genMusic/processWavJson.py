import librosa
import numpy as np
import madmom
import json
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def processWav(file_path, output_file, overwrite=False):
    """
    Process the WAV file to extract pitch, beats, and chords, then save the structured data.
    """

    def load_audio(file_path):
        start_time = time.time()
        try:
            y, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
        end_time = time.time()
        print(f"Audio loading took {end_time - start_time:.2f} seconds")
        return y, sr

    def detect_pitch(y, sr):
        start_time = time.time()
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_times = [
                (t, float(pitches[index, t]))
                for t in range(pitches.shape[1])
                if (index := magnitudes[:, t].argmax()) and pitches[index, t] > 0
            ]
        except Exception as e:
            print(f"Error detecting pitch: {e}")
            return []
        end_time = time.time()
        print(f"Pitch detection took {end_time - start_time:.2f} seconds")
        return pitch_times

    def detect_beats(y, sr):
        start_time = time.time()
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
        except Exception as e:
            print(f"Error detecting beats: {e}")
            return 0.0, []
        end_time = time.time()
        print(f"Beat detection took {end_time - start_time:.2f} seconds")
        return float(tempo), beat_times

    def detect_chords(file_path):
        start_time = time.time()
        try:
            proc = madmom.features.chords.CNNChordFeatureProcessor()
            chord_rec = madmom.features.chords.CRFChordRecognitionProcessor()
            chords = chord_rec(proc(file_path))
            chords = [(float(chord[0]), float(chord[1]), chord[2]) for chord in chords]
        except Exception as e:
            print(f"Error detecting chords: {e}")
            return []
        end_time = time.time()
        print(f"Chord detection took {end_time - start_time:.2f} seconds")
        return chords

    def style_labeling(pitch_times, beat_times, chords):
        start_time = time.time()
        try:
            styles = []
            if len(pitch_times) > 100:
                styles.append('Rich Melody')
            if len(beat_times) > 50:
                styles.append('Fast Tempo')
            if any('maj' in chord[2] for chord in chords):
                styles.append('Major Chords')
        except Exception as e:
            print(f"Error in style labeling: {e}")
            return []
        end_time = time.time()
        print(f"Style labeling took {end_time - start_time:.2f} seconds")
        return styles

    def generate_description(styles):
        return f"This audio features a {', '.join(styles)}."

    def frequency_to_midi(frequency):
        return 69 + 12 * np.log2(frequency / 440.0)

    def midi_to_note_name(midi_number):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_number // 12) - 1
        note = note_names[midi_number % 12]
        return f"{note}{octave}"

    def convert_pitch_sequence(pitch_sequence):
        note_sequence = []
        for _, frequency in pitch_sequence:
            if frequency > 0:
                midi_note = int(round(frequency_to_midi(frequency)))
                note_name = midi_to_note_name(midi_note)
                note_sequence.append(note_name)
        return note_sequence

    def visualize_pitch_sequence(y, sr, pitch_sequence, note_sequence):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        plt.figure(figsize=(15, 20))
        plt.subplots_adjust(wspace=1, hspace=0.2)

        plt.subplot(311)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.xlabel('Time')
        plt.ylabel('Chroma')

        # plt.subplot(312)
        # librosa.display.waveshow(y, sr=sr)
        # plt.xlabel('Time (seconds)')
        # plt.ylabel('Amplitude')

        plt.subplot(313)
        plt.grid(linewidth=0.5)
        plt.xticks(range(0, len(note_sequence), 50))
        plt.yticks(range(1, 13), note_names)
        note_positions = [i for i, _ in enumerate(note_sequence)]
        plt.scatter(note_positions, [note_names.index(note[:-1]) + 1 for note in note_sequence], marker="s", s=1,
                    color="red")
        plt.xlabel('Time (frames)')
        plt.ylabel('Notes')

        plt.show()

    def load_existing_data(output_file):
        if not os.path.exists(output_file):
            return {}
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def save_data(data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def start():
        existing_data = load_existing_data(output_file)

        if file_path in existing_data:
            if overwrite:
                print(f"Overwriting data for {file_path}.")
            else:
                print(f"Data for {file_path} already exists. Skipping...")
                return

        total_start_time = time.time()

        y, sr = load_audio(file_path)
        if y is None or sr is None:
            print("Skipping processing due to load error.")
            return

        with tqdm(total=100, desc="Processing Audio", bar_format="{l_bar}{bar} [time left: {remaining}]") as pbar:
            pitch_times = detect_pitch(y, sr)
            pbar.update(30)
            tempo, beat_times = detect_beats(y, sr)
            pbar.update(30)
            chords = detect_chords(file_path)
            pbar.update(30)
            styles = style_labeling(pitch_times, beat_times, chords)
            pbar.update(10)

        description = generate_description(styles)
        note_sequence = convert_pitch_sequence(pitch_times)

        data_entry = {
            "file_path": file_path,
            "labels": ",".join(styles),
            "description": description,
            "pitch_sequence": pitch_times,
            "beat_sequence": beat_times,
            "chord_sequence": chords,
            "note_sequence": note_sequence
        }

        existing_data[file_path] = data_entry
        save_data(existing_data, output_file)

        total_end_time = time.time()
        print(f"Total processing took {total_end_time - total_start_time:.2f} seconds")

        visualize_pitch_sequence(y, sr, pitch_times, note_sequence)

    start()


def list_audio_data(output_file):
    def load_existing_data(output_file):
        if not os.path.exists(output_file):
            return {}
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    data = load_existing_data(output_file)
    for key, value in data.items():
        print(f"File Path: {key}")
        print(json.dumps(value, ensure_ascii=False, indent=4))


def delete_audio_data(file_path, output_file):
    def load_existing_data(output_file):
        if not os.path.exists(output_file):
            return {}
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def save_data(data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    data = load_existing_data(output_file)
    if file_path in data:
        del data[file_path]
        save_data(data, output_file)
        print(f"Deleted data for {file_path} from {output_file}")
    else:
        print(f"No data found for {file_path}")


def query_audio_data(file_path, output_file):
    def load_existing_data(output_file):
        if not os.path.exists(output_file):
            return {}
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    data = load_existing_data(output_file)
    if file_path in data:
        return json.dumps(data[file_path], ensure_ascii=False, indent=4)
    else:
        return f"No data found for {file_path}"
