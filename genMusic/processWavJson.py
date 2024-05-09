import librosa
import numpy as np
import madmom
import json
import time
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

"""
代码说明
音频加载和特征提取：
load_audio：加载音频。
detect_pitch：检测音高。
detect_beats：检测节拍。
detect_chords：检测和弦。
音高和音符转换：
frequency_to_midi 和 midi_to_note_name：将频率转换为 MIDI 编号和音符名称。
convert_pitch_sequence：转换音高序列。
节拍分析：
analyze_rhythm：分析节拍强度和小节结构。
数据保存与可视化：
visualize_pitch_sequence：可视化音高、波形、音符序列和节拍强度。
数据管理：
load_existing_data 和 save_data：加载和保存数据。
list_audio_data、delete_audio_data 和 query_audio_data：数据管理接口。
"""
def processWav(file_path, output_file, overwrite=False):
    """
    Process the WAV file to extract pitch, beats, chords, and rhythmic structure, then save the structured data.
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

    def analyze_rhythm(y, sr, beat_times):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_strength = onset_env[beats]
        intervals = np.diff(beat_times)
        avg_interval = np.mean(intervals)

        """
            1. **节拍检测**：
               - 使用 `np.diff` 计算节拍间隔。
               - 根据间隔的分布推测最可能的拍号。
            
            2. **节拍结构分析**：
               - `likely_beats_per_bar` 表示每小节的节拍数，根据间隔的分布推断。
            
            3. **可视化**：
               - 在节奏分析中添加推测的拍号信息。
               - 将 `likely_beats_per_bar` 信息显示在图标题中。
        """
        # Analyze intervals to determine likely time signatures
        counts, bins = np.histogram(intervals, bins=np.linspace(0, max(intervals), num=50))
        peaks, _ = find_peaks(counts, height=np.max(counts) * 0.5)
        likely_beats_per_bar = bins[peaks].astype(int)

        # Ensure common_beat_interval is not zero or very small
        if len(likely_beats_per_bar) > 0 and np.max(likely_beats_per_bar) > 0:
            common_beat_interval = max(likely_beats_per_bar[0], 1)
            bars = np.arange(0, min(len(beat_times), 1000), common_beat_interval)
        else:
            common_beat_interval = 4  # Fallback to 4/4
            bars = np.arange(0, min(len(beat_times), 1000), common_beat_interval)

        rhythm_structure = {
            'tempo': tempo,
            'beats': beat_times,
            'beat_strength': beat_strength.tolist(),
            'bars': [beat_times[i] for i in bars],  # 使用列表推导来索引
            'likely_beats_per_bar': common_beat_interval
        }
        return rhythm_structure

    def visualize_pitch_sequence(y, sr, pitch_sequence, note_sequence, rhythm_structure):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        plt.figure(figsize=(15, 20))
        plt.subplots_adjust(wspace=1, hspace=0.2)

        plt.subplot(411)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.title('Chroma Representation')
        plt.xlabel('Time')
        plt.ylabel('Chroma')

        # plt.subplot(412)
        # librosa.display.waveshow(y, sr=sr)
        # plt.title('Waveform')
        # plt.xlabel('Time (seconds)')
        # plt.ylabel('Amplitude')

        plt.subplot(413)
        plt.grid(True)
        plt.xticks(range(0, len(note_sequence), max(1, len(note_sequence) // 50)))
        plt.yticks(range(1, 13), note_names)
        note_positions = [i for i, _ in enumerate(note_sequence)]
        note_values = [note_names.index(note[:-1]) + 1 for note in note_sequence if note[:-1] in note_names]
        plt.scatter(note_positions, note_values, marker="s", s=1, color="red")
        plt.title('Note Sequence')
        plt.xlabel('Time (frames)')
        plt.ylabel('Notes')

        plt.subplot(414)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        plt.plot(times, onset_env, label='Onset Strength')
        plt.vlines(librosa.frames_to_time(np.arange(len(onset_env))), 0, onset_env.max(), color='r', alpha=0.5,
                   linestyle='--', label='Beats')
        plt.title(f'Rhythm Analysis - Likely {rhythm_structure["likely_beats_per_bar"]}/4 Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Onset Strength')
        plt.legend()

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

        with tqdm(total=100, desc="Processing Audio", bar_format="{l_bar}{bar} [time left: {remaining}]\n") as pbar:
            pitch_times = detect_pitch(y, sr)
            pbar.update(10)
            tempo, beat_times = detect_beats(y, sr)
            pbar.update(10)
            chords = detect_chords(file_path)
            pbar.update(30)
            styles = style_labeling(pitch_times, beat_times, chords)
            pbar.update(10)
            description = generate_description(styles)
            pbar.update(20)
            note_sequence = convert_pitch_sequence(pitch_times)
            pbar.update(20)
            rhythm_structure = analyze_rhythm(y, sr, beat_times)

        data_entry = {
            "file_path": file_path,
            "labels": ",".join(styles),
            "description": description,
            "pitch_sequence": pitch_times,
            "beat_sequence": beat_times,
            "chord_sequence": chords,
            "note_sequence": note_sequence,
            "rhythm_structure": rhythm_structure
        }

        existing_data[file_path] = data_entry
        save_data(existing_data, output_file)

        total_end_time = time.time()
        print(f"Total processing took {total_end_time - total_start_time:.2f} seconds")

        visualize_pitch_sequence(y, sr, pitch_times, note_sequence, rhythm_structure)

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
