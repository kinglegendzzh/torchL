import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import librosa
import numpy as np
import json
import soundfile as sf
from spleeter.separator import Separator
import time
import logging
import psutil
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f'{stage} - Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB')


def separate_audio(file_path, output_dir):
    """
    使用 Spleeter 分离音轨
    """
    logging.info(f'Starting audio separation for {file_path}')
    log_memory_usage('Before separation')

    separator = Separator('spleeter:5stems')
    separator.separate_to_file(file_path, output_dir)
    del separator  # 释放 Spleeter 资源
    gc.collect()  # 强制进行垃圾回收

    log_memory_usage('After separation')
    logging.info(f'Audio separation completed for {file_path}')

def load_audio(file_path):
    logging.info(f'Loading audio from {file_path}')
    y, sr = librosa.load(file_path, sr=None)
    logging.info(f'Loaded audio from {file_path}, sample rate: {sr}, duration: {len(y) / sr:.2f} seconds')
    log_memory_usage('After loading audio')
    return y, sr


def extract_features(y, sr):
    logging.info(f'Extracting features from audio, sample rate: {sr}, duration: {len(y) / sr:.2f} seconds')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    logging.info(f'Extracted MFCC features with shape: {mfcc.shape}')
    log_memory_usage('After extracting features')
    return mfcc


def classify_instruments(features):
    logging.info(f'Classifying instruments with feature shape: {features.shape}')
    # 你可以使用预训练的模型或自己训练的模型来分类乐器
    # 这里假设你已经有一个模型用于分类
    # 示例：model.predict(features)
    pass


def process_audio(file_path, output_dir):
    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0])
    logging.info(f'Processing audio file: {file_path}')
    log_memory_usage('Start processing audio')

    absolute_file_path = os.path.abspath(file_path)
    logging.info(f'Absolute file path: {absolute_file_path}')

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f'Created output directory: {output_dir}')
    else:
        logging.info(f'Output directory already exists: {output_dir}')

    # 生成总轨特征文件
    y, sr = load_audio(file_path)
    features = extract_features(y, sr)
    feature_file = os.path.join(output_dir,
                                f'full_features.json')
    with open(feature_file, 'w') as f:
        json.dump(features.tolist(), f)
    logging.info(f'Saved features to {feature_file}')
    del y, sr, features
    logging.info(f'Cleaned up memory for {feature_file} processing')

    # 分离音轨
    separate_audio(file_path, output_dir)

    # 加载分离后的音轨
    stems = ['vocals', 'drums', 'bass', 'piano', 'other']
    for stem in stems:
        stem_file = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0], f'{stem}.wav')

        # 检查分离出的文件是否存在
        if not os.path.exists(stem_file):
            logging.error(f'Error: {stem_file} does not exist.')
            continue

        y, sr = load_audio(stem_file)

        # 提取特征
        features = extract_features(y, sr)

        # 分类音色
        classify_instruments(features)

        # 生成唯一的特征文件
        feature_file = os.path.join(output_dir,
                                    f'{stem}_features.json')
        with open(feature_file, 'w') as f:
            json.dump(features.tolist(), f)
        logging.info(f'Saved features to {feature_file}')

        # 清理内存中的音频数据
        del y, sr, features
        logging.info(f'Cleaned up memory for {stem} processing')


if __name__ == "__main__":
    # 示例使用
    file_path = 'E:\\apps\\pycharm\\project\\torchL\\ailab\\genMusic\\musicLab\\fromNetEase\\手嶌葵-Young-and-Beautiful.wav'
    output_dir = 'data'
    process_audio(file_path, output_dir)
    logging.info(f'SpleeterTest success end!!')
