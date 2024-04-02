import subprocess
import re


def get_gpu_utilization():
    # 调用nvidia-smi命令获取GPU状态
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE)
    # 解码输出结果
    output = result.stdout.decode('utf-8')
    # 使用正则表达式匹配数字（GPU利用率）
    gpu_utilization = re.findall(r'\d+', output)
    return gpu_utilization


gpu_utilization = get_gpu_utilization()
print(f"GPU Utilization: {gpu_utilization}")
