torchL AiLab
=======
KL的人工智能实验室

所需库及其配置环境
=======
- python = 3.11
- anaconda3
- pytorch = 2.2.2
- torchaudio = 2.2.2
- torchvision = 0.17.2
- matplotlib = 3.8.0
- numpy = 1.26.4
- madmom = dev
- scipy = 1.11.4
- librosa = 0.10.1
- jupyter = 1.0.0

大体的研究方向
======
2024.5.8 update:
- 目前涉及到的主要内容：
1. 基于librosa识别数字音频提取MFCC特征
2. 基于madmom识别数字音频实现和弦识别及节拍检测
3. 基于on-hot的条件变分自编码器（包括炼丹日志）
4. 变分自编码器（基于原始图像生成采样图像）
5. CNN经典现代网络模型的原理解构
6. 基于LSTM（长时记忆）的循环神经网络

- 预计要进行的长期研究：
1. 基于马尔科夫链的（深度）强化学习研究（Q-Leaning、DQN、HMM）
2. 生成式对抗网络（GAN）的研究
3. 基于注意力机制的Transformer模型的pytorch实现（LLM）