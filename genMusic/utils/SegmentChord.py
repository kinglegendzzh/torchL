import genMusic.processWavJson
from utils.ProcessData import save_data, load_existing_data


def segment_chords_by_bars(chords, beat_structure):
    """
    Segment chords based on bar structure derived from beat analysis.

    :param chords: List of tuples (start_time, end_time, chord_label)
    :param beat_structure: Dictionary containing 'bars' which are start times of each bar
    :return: Dictionary where keys are bar indices and values are lists of chords in that bar
    """
    bars = beat_structure.get('bars', [])
    segmented_chords = {i: [] for i in range(len(bars) - 1)}

    chord_index = 0
    num_chords = len(chords)

    for i in range(len(bars) - 1):
        bar_start = bars[i]
        bar_end = bars[i + 1]

        while chord_index < num_chords and chords[chord_index][0] < bar_end:
            chord_start, chord_end, chord_label = chords[chord_index]

            if chord_start >= bar_start:
                segmented_chords[i].append((chord_start, chord_end, chord_label))

            chord_index += 1

    return segmented_chords

# 示例调用
output_file = 'output_data.json'
file_path = 'path/to/your/audio.wav'

# 加载现有数据
existing_data = load_existing_data(output_file)

if file_path in existing_data:
    data_entry = existing_data[file_path]
    chords = data_entry.get('chord_sequence', [])
    rhythm_structure = data_entry.get('rhythm_structure', {})

    segmented_chords = segment_chords_by_bars(chords, rhythm_structure)

    # 打印分割后的和弦
    for bar_index, chords_in_bar in segmented_chords.items():
        print(f"Bar {bar_index}:")
        for chord in chords_in_bar:
            print(f"  {chord}")

    # 可以根据需要将 `segmented_chords` 添加回 `data_entry` 中并保存
    data_entry['segmented_chords'] = segmented_chords
    save_data(existing_data, output_file)
else:
    print(f"No data found for {file_path}")
