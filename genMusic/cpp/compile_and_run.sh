#!/bin/bash

# 设置路径变量
INCLUDE_PATH="/Users/ZZH/AppData/Local/Temp/include"
LIB_PATH="/Users/ZZH/AppData/Local/Temp/lib"
SOURCE_FILE="detectChords.cpp"
OUTPUT_FILE="ChordDetection"
AUDIO_FILE="E:\apps\pycharm\project\torchL\ailab\genMusic\musicLab\fromMe\dorianD.wav"

# 编译
g++ -std=c++11 -I${INCLUDE_PATH} -L${LIB_PATH} -o ${OUTPUT_FILE} ${SOURCE_FILE} -laubio -lsndfile

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# 运行可执行文件
./${OUTPUT_FILE} ${AUDIO_FILE}
