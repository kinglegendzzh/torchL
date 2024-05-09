import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载音频文件
# filename = 'data/dorianD.mp3'
filename = 'musicLab/fromNetEase/xunzhang.mp3'
y, sr = librosa.load(filename)

# 计算MFCC特征
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# 可视化MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
