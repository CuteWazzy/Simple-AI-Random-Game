"""
一键启动脚本：快速开始训练 / 观战 / 人机对弈。

用法：
  python3 run.py train       # 遗传算法训练（推荐）
  python3 run.py viz         # 可视化训练
  python3 run.py watch       # AI 自对战观战
  python3 run.py human       # 人机对弈
  python3 run.py stats       # 查看模型统计
"""

import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(script_dir, 'scripts')

    if cmd == 'train':
        # 遗传算法训练
        args = sys.argv[2:] if len(sys.argv) > 2 else ['--gens', '30', '--iters', '8', '--workers', '4']
        subprocess.run([sys.executable, os.path.join(scripts_dir, 'genetic_train.py')] + args)

    elif cmd == 'viz':
        # 可视化训练
        args = sys.argv[2:] if len(sys.argv) > 2 else ['--gens', '20', '--iters', '8', '--workers', '4']
        subprocess.run([sys.executable, os.path.join(scripts_dir, 'visualize_train.py')] + args)

    elif cmd == 'watch':
        # AI 自对战
        args = sys.argv[2:] if len(sys.argv) > 2 else ['1', '0', '0.5', '80']
        subprocess.run([sys.executable, os.path.join(scripts_dir, 'watch_compact.py')] + args)

    elif cmd == 'human':
        # 人机对弈
        args = sys.argv[2:] if len(sys.argv) > 2 else ['human', '--seat', '0', '--temp', '0.1']
        subprocess.run([sys.executable, os.path.join(scripts_dir, 'play.py')] + args)

    elif cmd == 'stats':
        # 查看模型统计
        import torch
        sys.path.insert(0, scripts_dir)
        from model import PolicyValueNet

        model_path = os.path.join(script_dir, 'models', 'model.pt')
        if not os.path.exists(model_path):
            print(f"模型不存在: {model_path}")
            return

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"模型: {model_path}")
        print(f"  generation: {ckpt.get('generation', '?')}")
        print(f"  best_name: {ckpt.get('best_name', '?')}")
        print(f"  win_rate: {ckpt.get('win_rate', 0)*100:.1f}%")

        net = PolicyValueNet()
        net.load_state_dict(ckpt['model'])
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  参数量: {n_params}")

    else:
        print(f"未知命令: {cmd}")
        print(__doc__)

if __name__ == '__main__':
    main()
