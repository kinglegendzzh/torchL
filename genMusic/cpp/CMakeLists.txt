cmake_minimum_required(VERSION 3.5)
project(AubioProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加 aubio 库的头文件路径
include_directories("C:/Users/ZZH/AppData/Local/Temp/include")

# 添加 aubio 库的库文件路径
link_directories("C:/Users/ZZH/AppData/Local/Temp/lib")

# 添加可执行文件
add_executable(AubioProject detectChords.cpp)

# add_executable(AubioProject test-fft.c)

# 链接 aubio 库和必要的系统库
target_link_libraries(AubioProject aubio)
