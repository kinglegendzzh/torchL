import librosa
import numpy as np
import madmom
import json
import time
import os
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
        # 加载现有数据
        existing_data = load_existing_data(output_file)

        # 检查是否已存在数据
        if file_path in existing_data:
            if overwrite:
                print(f"Overwriting data for {file_path}.")
            else:
                print(f"Data for {file_path} already exists. Skipping...")
                return

        # 开始处理
        total_start_time = time.time()

        y, sr = load_audio(file_path)
        if y is None or sr is None:
            print("Skipping processing due to load error.")
            return

        # 使用 tqdm 显示进度条
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

        # 组织数据
        data_entry = {
            "file_path": file_path,
            "labels": ",".join(styles),
            "description": description,
            "pitch_sequence": pitch_times,
            "beat_sequence": beat_times,
            "chord_sequence": chords
        }

        # 更新数据
        existing_data[file_path] = data_entry

        # 保存数据
        save_data(existing_data, output_file)

        total_end_time = time.time()
        print(f"Total processing took {total_end_time - total_start_time:.2f} seconds")

    start()


def list_audio_data(output_file):
    """
    List all audio data entries in the output file.
    """

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
    """
    Delete the specified audio data entry from the output file.
    """

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
    """
    Query the specified audio data entry from the output file and return as JSON string.
    """

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
