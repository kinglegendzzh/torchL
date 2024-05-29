@echo off
setlocal

:: 设置路径变量
set INCLUDE_PATH=C:\Users\ZZH\AppData\Local\Temp\include
set LIB_PATH=C:\Users\ZZH\AppData\Local\Temp\lib
set SOURCE_FILE=detectChords.cpp
set OUTPUT_FILE=ChordDetection.exe
set AUDIO_FILE=E:\apps\pycharm\project\torchL\ailab\genMusic\musicLab\fromMe\dorianD.wav

:: 编译
g++ -std=c++11 -I%INCLUDE_PATH% -L%LIB_PATH% -o %OUTPUT_FILE% %SOURCE_FILE% -laubio -lsndfile

:: 检查编译是否成功
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)

echo Compilation successful!

:: 运行可执行文件
%OUTPUT_FILE% %AUDIO_FILE%

endlocal
