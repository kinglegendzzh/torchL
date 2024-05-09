"""
要将音高（频率值）转换为音乐术语（如 C1, A2, D3 等），可以使用 MIDI 音符表示进行转换。然后，将 MIDI 音符转换为标准的音高名称。以下是一个具体的实现步骤和代码示例。

实现步骤
频率转换为 MIDI 音符：
使用公式：midi_note = 69 + 12 * log2(frequency / 440.0)
MIDI 音符转换为音高名称：
建立 MIDI 音符与音高名称的对应关系。

代码解释
频率到 MIDI 音符转换：
frequency_to_midi(frequency) 函数将频率转换为 MIDI 音符编号。
MIDI 音符到音高名称转换：
midi_to_note_name(midi_number) 函数将 MIDI 音符编号转换为标准音高名称。
使用 note_names 列表存储音高名称。
处理音高序列：
convert_pitch_sequence(pitch_sequence) 函数遍历 pitch_sequence，将频率转换为音高名称。
忽略频率为 0 的无效条目。
输出转换结果：
将转换后的音高名称序列打印输出。
"""
import numpy as np

# 音高频率到 MIDI 音符转换函数
def frequency_to_midi(frequency):
    return 69 + 12 * np.log2(frequency / 440.0)

# MIDI 音符到音高名称转换函数
def midi_to_note_name(midi_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"

# 处理音高序列函数
def convert_pitch_sequence(pitch_sequence):
    note_sequence = []
    for _, frequency in pitch_sequence:
        if frequency > 0:  # 忽略无效频率
            midi_note = int(round(frequency_to_midi(frequency)))
            note_name = midi_to_note_name(midi_note)
            note_sequence.append(note_name)
    return note_sequence

# 示例音高序列
pitch_sequence = [
    [5, 425.21435546875],
    [6, 611.1715698242188],
    [7, 575.2137451171875],
    [8, 571.33056640625],
    [9, 259.1099548339844]
]

# 转换音高序列
note_sequence = convert_pitch_sequence(pitch_sequence)

# 输出转换结果
print("Converted Note Sequence:", note_sequence)
