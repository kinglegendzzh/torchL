import librosa
import numpy as np
import madmom
import json
import time
import os


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

    def check_existing_data(file_path, output_file):
        if not os.path.exists(output_file):
            return False
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if file_path in line:
                    return True
        return False

    def remove_existing_data(file_path, output_file):
        if not os.path.exists(output_file):
            return
        lines = []
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                if file_path not in line:
                    f.write(line)

    # 检查是否已存在数据
    if check_existing_data(file_path, output_file):
        if overwrite:
            remove_existing_data(file_path, output_file)
        else:
            print(f"Data for {file_path} already exists. Skipping...")
            return

    # 开始处理
    total_start_time = time.time()

    y, sr = load_audio(file_path)
    if y is None or sr is None:
        print("Skipping processing due to load error.")
        return

    pitch_times = detect_pitch(y, sr)
    tempo, beat_times = detect_beats(y, sr)
    chords = detect_chords(file_path)
    styles = style_labeling(pitch_times, beat_times, chords)
    description = generate_description(styles)

    # 组织数据
    data = {
        "file_path": file_path,
        "labels": ",".join(styles),
        "description": description,
        "pitch_sequence": json.dumps(pitch_times),
        "beat_sequence": json.dumps(beat_times),
        "chord_sequence": json.dumps(chords)
    }

    # 写入文件，使用 UTF-8 编码
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            line = f'{data["file_path"]}|{data["labels"]}|{data["description"]}|{data["pitch_sequence"]}|{data["beat_sequence"]}|{data["chord_sequence"]}\n'
            f.write(line)
    except Exception as e:
        print(f"Error writing to file: {e}")

    total_end_time = time.time()
    print(f"Total processing took {total_end_time - total_start_time:.2f} seconds")


def list_audio_data(output_file):
    """
    List all audio data entries in the output file.
    """
    if not os.path.exists(output_file):
        print("Output file does not exist.")
        return
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        print(line.strip())


def delete_audio_data(file_path, output_file):
    """
    Delete the specified audio data entry from the output file.
    """
    if not os.path.exists(output_file):
        print("Output file does not exist.")
        return
    lines = []
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            if file_path not in line:
                f.write(line)
    print(f"Deleted data for {file_path} from {output_file}")
