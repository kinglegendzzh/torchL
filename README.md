torchL AILab
=======
KL的人工智能实验室

[TOC]

所需库及其配置环境
=======
- python = 3.11
- c++
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
- spleeter
- aubio

大体的研究方向
======

- [x] 音频识别算法及其衍生应用场景

- [ ] 音乐序列生成式AI

- [x] NLP
  - [x] LSTM
  - [x] Transformer
  - [ ] KAN
  - [ ] Multi-Agent
- [x] 计算机视觉
  - [x] CNN
  - [x] VAE


更新记录
======
**2024.5.30**

- [x] 对音频进行预处理，转储结构化数据，生成“音乐风格识别算法”训练集，预处理动作如下：
  1. 基于spleeter实现对音频进行乐器与人声分离；
  2. 基于librosa和madmom实现对音高、节拍和简单和弦的特征提取；
  3. 基于拍号分割柱式和弦的和声分析方法；

- [x] 基于aubio，对音频进行fft分离（基于 c++的 跨平台支持）

- [ ] 通过LSTM或HMM（隐马尔科夫）模型提高对音频复杂和弦的识别率。

  > 关于chordPrediction项目：https://github.com/kinglegendzzh/chordPrediction

- [x] 发布软件使用指南至B站，传送门：https://www.bilibili.com/video/BV1Ww4m1i7CN

- [x] 研究基于musicpy提取主旋律与和弦，参考资料：https://musicpy.readthedocs.io/zh-cn/latest

- [ ] 使用HMM（隐马尔科夫）模型优化预测算法；

- [ ] 向量化的和弦情绪曲线识别；

- [ ] 前端方面放弃使用pyQT5，转向WebUI开发。

  > AIGC与游戏开发方面：

- [x] ComfyUI本地化部署，工作流原理研究；

- [x] Stable Diffusion v1.5，Stable Diffusion XL、lora等常见大模型的组合，搭建文生图工作流；

- [x] 使用ComfyUI-3D-Pack、TripoSR等搭建2D转3D建模工作流；

- [x] 对Godot游戏引擎，GDScript脚本API 进行熟悉与了解。

- [ ] 构建AI化的3D游戏场景设计工作流；

- [ ] 通过ollama构建本地大语言模型。

  ------

  

**2024.5.8**

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

