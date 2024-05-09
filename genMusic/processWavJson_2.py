# 示例使用
from genMusic.processWavJson import list_audio_data, processWav, query_audio_data, delete_audio_data

file_path = 'musicLab/fromMe/dorianD.wav'  # 替换为实际的WAV文件路径
output_file = 'data/audio_dataset.json'
processWav(file_path, output_file, overwrite=True)

# 查询和删除示例
# print("Listing all data:")
# list_audio_data(output_file)

print("\nQuery specific data:")
print(query_audio_data(file_path, output_file))
#
# print("\nDeleting specific data:")
# delete_audio_data(file_path, output_file)
#
# print("\nListing all data after deletion:")
# list_audio_data(output_file)
