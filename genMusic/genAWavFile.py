import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# 参数
frequency = 440  # 频率，Hz
duration = 1     # 持续时间，秒
amplitude = 0.5  # 振幅
sample_rate = 44100  # 采样率，Hz

# 生成时间序列
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# 生成正弦波信号
signal = amplitude * np.sin(2 * np.pi * frequency * t)

# 保存为WAV文件
signal_int16 = np.int16(signal * 32767)  # 转换为16位整数
write('wav/sine_wave_440Hz.wav', sample_rate, signal_int16)

# 可视化信号
plt.plot(t[:1000], signal[:1000])  # 只画出前1000个样本
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sine Wave 440Hz')
plt.show()
