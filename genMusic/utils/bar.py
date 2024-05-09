import time
import sys


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration  - Required  : current iteration (Int)
        total      - Required  : total iterations (Int)
        prefix     - Optional  : prefix string (Str)
        suffix     - Optional  : suffix string (Str)
        length     - Optional  : character length of bar (Int)
        fill       - Optional  : bar fill character (Str)
        print_end  - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


# 示例使用
total = 100
for i in range(total + 1):
    print_progress_bar(i, total, prefix='Progress:', suffix='Complete', length=50)
    time.sleep(0.1)  # 模拟一些工作
