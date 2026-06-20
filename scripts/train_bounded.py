"""
分批训练包装脚本：在指定时长内尽量多训练，时间到自动保存退出。
这样每次 bash 调用都能跑完一段，多次调用累积训练。

用法：python3 train_bounded.py [seconds] [games_per_iter] [lr] [temperature]
"""

import sys
import os
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_num_threads(1)

from train import train


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException()


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 8e-5
    temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 800

    # 设置 SIGALRM，到时抛 TimeoutException
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)

    iters_done = 0
    try:
        # 一次跑 20 轮，跑完再继续；直到 timeout
        while True:
            train(
                num_iterations=20,
                games_per_iter=games,
                num_players=4,
                lr=lr,
                temperature=temp,
                batch_size=256,
                save_every=5,
                log_every=5,
                seed=seed + iters_done,
            )
            iters_done += 20
            print(f"[bounded] 已完成 {iters_done} 轮，继续...", flush=True)
    except TimeoutException:
        print(f"[bounded] 时间到，本次共完成约 {iters_done} 轮", flush=True)
    except KeyboardInterrupt:
        print(f"[bounded] 中断，本次共完成约 {iters_done} 轮", flush=True)


if __name__ == '__main__':
    main()
