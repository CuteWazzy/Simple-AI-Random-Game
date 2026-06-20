"""
可视化训练脚本：实时绘制训练曲线 + 锦标赛胜率 + 道具使用统计。

用法：
  python3 visualize_train.py --gens 20 --iters 8 --workers 4

功能：
  - 实时显示 4 个种群的胜率曲线
  - 显示训练 loss 曲线
  - 显示道具使用统计
  - 训练结束保存图表为 PNG
"""

import os
import sys
import time
import random
import argparse
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，保存 PNG
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体支持
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)

from game import (
    Game, PHASE_CHOOSE_TARGET, PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE,
    PHASE_EDIT_POS, PHASE_EDIT_DELTA, PHASE_REWARD_CHOICE,
    PHASE_GAME_OVER, NUM_PHASES,
)
from model import PolicyValueNet, masked_softmax
from numpy_net import NumpyNet, masked_softmax_np, action_mask_np
from train import compute_returns, StepRecord
from genetic_train import (
    PopConfig, DEFAULT_CONFIGS, make_net, load_or_init_net,
    play_with_config, train_one_population, _train_worker,
    play_match_between_nets, ai_pick_action_with_net,
    crossover_nets, mutate_net, tournament,
)


class Visualizer:
    """训练可视化器：定期保存图表 PNG。"""

    def __init__(self, output_dir: str = './models/viz'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.history = {
            'gen': [],
            'win_rates': {p.name: [] for p in DEFAULT_CONFIGS},
            'losses': {p.name: [] for p in DEFAULT_CONFIGS},
            'best_win_rate': [],
            'item_usage': {p.name: [] for p in DEFAULT_CONFIGS},
        }
        self.best_win_rate = -1.0
        self.best_name = None

    def update(self, gen: int, win_rates: Dict, train_stats: Dict,
               item_usage: Dict = None):
        """更新历史数据。"""
        self.history['gen'].append(gen)
        for name, wr in win_rates.items():
            self.history['win_rates'][name].append(wr)
        for name in self.history['losses']:
            stats = train_stats.get(name, {})
            self.history['losses'][name].append(stats.get('avg_loss', 0.0))
        if item_usage:
            for name, usage in item_usage.items():
                self.history['item_usage'][name].append(usage)

        # 更新最佳
        current_best = max(win_rates.values()) if win_rates else 0
        if current_best > self.best_win_rate:
            self.best_win_rate = current_best
            self.best_name = max(win_rates, key=win_rates.get)
        self.history['best_win_rate'].append(self.best_win_rate)

    def save_plots(self, suffix: str = ''):
        """保存所有图表。"""
        self._plot_win_rates(suffix)
        self._plot_losses(suffix)
        self._plot_best(suffix)
        if any(self.history['item_usage'][name] for name in self.history['item_usage']):
            self._plot_item_usage(suffix)

    def _plot_win_rates(self, suffix: str):
        """4 个种群的胜率曲线。"""
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        gens = self.history['gen']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, (name, wrs) in enumerate(self.history['win_rates'].items()):
            if wrs:
                ax.plot(gens[-len(wrs):], wrs, label=name, color=colors[i % 4],
                        linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('代数')
        ax.set_ylabel('胜率')
        ax.set_title('种群胜率演进' + suffix)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.7, max(max(w) if w else 0 for w in self.history['win_rates'].values()) + 0.05))
        path = os.path.join(self.output_dir, f'win_rates{suffix}.png')
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"  -> 保存 {path}", flush=True)

    def _plot_losses(self, suffix: str):
        """训练 loss 曲线。"""
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        gens = self.history['gen']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, (name, losses) in enumerate(self.history['losses'].items()):
            if losses:
                ax.plot(gens[-len(losses):], losses, label=name, color=colors[i % 4],
                        linewidth=2, alpha=0.8)
        ax.set_xlabel('代数')
        ax.set_ylabel('Loss')
        ax.set_title('训练 Loss 演进' + suffix)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, f'losses{suffix}.png')
        plt.savefig(path, dpi=100)
        plt.close()

    def _plot_best(self, suffix: str):
        """历史最佳胜率曲线。"""
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        gens = self.history['gen']
        best = self.history['best_win_rate']
        ax.plot(gens, best, color='#9b59b6', linewidth=3, label='历史最佳')
        ax.fill_between(gens, 0, best, alpha=0.2, color='#9b59b6')
        ax.set_xlabel('代数')
        ax.set_ylabel('胜率')
        ax.set_title(f'历史最佳胜率（当前: {self.best_name or "-"} {self.best_win_rate*100:.1f}%）' + suffix)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.8, max(best) + 0.05) if best else 0.8)
        path = os.path.join(self.output_dir, f'best_win_rate{suffix}.png')
        plt.savefig(path, dpi=100)
        plt.close()

    def _plot_item_usage(self, suffix: str):
        """道具使用统计。"""
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        gens = self.history['gen']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, (name, usages) in enumerate(self.history['item_usage'].items()):
            if usages:
                ax.plot(gens[-len(usages):], usages, label=name, color=colors[i % 4],
                        linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('代数')
        ax.set_ylabel('道具使用次数')
        ax.set_title('道具使用统计' + suffix)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, f'item_usage{suffix}.png')
        plt.savefig(path, dpi=100)
        plt.close()


def count_item_usage(net: PolicyValueNet, config: PopConfig,
                     num_games: int = 10, seed: int = 0) -> int:
    """统计某网络在 num_games 局中的道具使用次数。"""
    rng = random.Random(seed)
    np_net = NumpyNet(net, activation=config.activation)
    total_uses = 0
    for _ in range(num_games):
        recs, _ = play_with_config(net, config, num_players=4, rng=rng, np_net=np_net)
        total_uses += sum(1 for r in recs if r.phase == PHASE_ITEM_CHOICE and r.action_idx != 0)
    return total_uses


def evolve_with_viz(
    num_generations: int = 10,
    iters_per_gen: int = 8,
    base_ckpt_path: str = './models/genetic_model.pt',
    output_path: str = './models/genetic_model.pt',
    num_workers: int = 4,
    seed: int = 42,
    viz_every: int = 1,
):
    """带可视化的进化训练。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_gen = 0
    best_net = None
    best_win_rate = -1.0
    best_name = None

    # 续训
    if os.path.exists(output_path):
        try:
            old_ckpt = torch.load(output_path, map_location='cpu', weights_only=False)
            start_gen = old_ckpt.get('generation', 0)
            best_win_rate = old_ckpt.get('win_rate', -1.0)
            best_name = old_ckpt.get('best_name', None)
            if 'model' in old_ckpt:
                best_net = PolicyValueNet()
                best_net.load_state_dict(old_ckpt['model'])
                best_net.eval()
            if best_name:
                print(f"[续训] 从第 {start_gen} 代继续，历史最佳: [{best_name}] 胜率={best_win_rate*100:.1f}%", flush=True)
            base_ckpt_path = output_path
        except Exception as e:
            print(f"[续训] 加载失败：{e}，从头开始", flush=True)

    print(f"\n{'='*70}")
    print(f"  可视化遗传算法训练")
    print(f"  种群数: {len(DEFAULT_CONFIGS)} | 起始代: {start_gen+1} | 目标代: {start_gen+num_generations}")
    print(f"  每代训练: {iters_per_gen} 轮 | 并行 worker: {num_workers}")
    print(f"  可视化输出: ./models/viz/")
    print(f"{'='*70}\n")

    # 初始化可视化器
    viz = Visualizer()
    viz.best_win_rate = best_win_rate
    viz.best_name = best_name

    # 初始化种群
    populations = []
    for config in DEFAULT_CONFIGS:
        net = load_or_init_net(config, base_ckpt_path)
        populations.append({
            'name': config.name,
            'config': config,
            'net': net,
            'win_rate': 0.0,
            'history': [],
            'train_stats': {},
        })
        print(f"  初始化种群 [{config.name}] 激活={config.activation} T={config.temperature}", flush=True)

    for gen in range(start_gen, start_gen + num_generations):
        gen_start = time.time()
        print(f"\n{'='*70}")
        print(f"  第 {gen+1} 代 (目标 {start_gen + num_generations})")
        print(f"{'='*70}")

        # 阶段 1：多进程并行训练
        print(f"\n[阶段 1] 多进程并行训练 ({iters_per_gen} 轮/种群)...")
        train_start = time.time()

        worker_args = []
        for pop in populations:
            cfg_copy = deepcopy(pop['config'])
            init_state = {k: v.clone() for k, v in pop['net'].state_dict().items()}
            base_seed_i = seed + gen * 1000 + (hash(pop['name']) & 0xFFFF)
            worker_args.append((cfg_copy, init_state, iters_per_gen, base_seed_i))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_train_worker, arg) for arg in worker_args]
            results = {}
            for future in as_completed(futures):
                name, state_dict, stats, cfg_dict = future.result()
                results[name] = (state_dict, stats, cfg_dict)

        for pop in populations:
            if pop['name'] in results:
                state_dict, stats, cfg_dict = results[pop['name']]
                pop['net'].load_state_dict(state_dict)
                pop['config'].temperature = cfg_dict['temperature']
                pop['train_stats'] = stats
                print(f"  [{pop['name']}] loss={stats['avg_loss']:.4f} T={pop['config'].temperature:.3f}", flush=True)

        train_time = time.time() - train_start
        print(f"  训练耗时: {train_time:.1f}s")

        # 阶段 2：锦标赛评估
        print(f"\n[阶段 2] 锦标赛评估...")
        tourney_start = time.time()
        win_rates = tournament(populations, num_matches_per_pair=2,
                               seed=seed + gen * 100)
        tourney_time = time.time() - tourney_start

        for pop in populations:
            pop['win_rate'] = win_rates.get(pop['name'], 0.0)
            pop['history'].append(pop['win_rate'])
        print(f"  锦标赛耗时: {tourney_time:.1f}s")
        print(f"  胜率:")
        for pop in sorted(populations, key=lambda x: -x['win_rate']):
            print(f"    {pop['name']}: {pop['win_rate']*100:.1f}%", flush=True)

        # 阶段 3：统计道具使用
        print(f"\n[阶段 3] 统计道具使用...")
        item_usage = {}
        for pop in populations:
            usage = count_item_usage(pop['net'], pop['config'],
                                      num_games=5, seed=seed + gen * 100)
            item_usage[pop['name']] = usage
            print(f"  [{pop['name']}] 5 局道具使用: {usage} 次", flush=True)

        # 阶段 4：遗传操作
        if gen < start_gen + num_generations - 1:
            print(f"\n[阶段 4] 遗传操作...")
            sorted_pops = sorted(populations, key=lambda x: -x['win_rate'])
            elite = sorted_pops[0]
            print(f"  精英: [{elite['name']}] 胜率={elite['win_rate']*100:.1f}%")

            top2 = sorted_pops[:2]
            new_populations = [elite]
            for i in range(1, len(populations)):
                alpha = random.random()
                child_net = crossover_nets(top2[0]['net'], top2[1]['net'], alpha=alpha)
                child_net = mutate_net(child_net, mutation_rate=0.1,
                                       mutation_std=0.03,
                                       rng=random.Random(seed + gen * 100 + i))
                new_populations.append({
                    'name': sorted_pops[i]['name'],
                    'config': sorted_pops[i]['config'],
                    'net': child_net,
                    'win_rate': 0.0,
                    'history': sorted_pops[i]['history'],
                    'train_stats': {},
                })
            populations = new_populations
            print(f"  产生 {len(new_populations)-1} 个后代")

        # 更新最佳
        for pop in populations:
            if pop['win_rate'] > best_win_rate:
                best_win_rate = pop['win_rate']
                best_net = deepcopy(pop['net'])
                best_name = pop['name']
        if best_net is None:
            best_net = deepcopy(sorted(populations, key=lambda x: -x['win_rate'])[0]['net'])

        # 保存
        ckpt = {
            'model': best_net.state_dict(),
            'iteration': (gen + 1) * iters_per_gen,
            'win_rate': best_win_rate,
            'best_name': best_name,
            'generation': gen + 1,
            'populations_history': {p['name']: p['history'] for p in populations},
        }
        torch.save(ckpt, output_path)

        # 更新可视化
        train_stats_dict = {p['name']: p['train_stats'] for p in populations}
        viz.update(gen + 1, win_rates, train_stats_dict, item_usage)

        # 每代保存图表
        if (gen + 1) % viz_every == 0 or gen + 1 == start_gen + num_generations:
            print(f"\n[可视化] 保存图表...")
            viz.save_plots(suffix=f'_gen{gen+1}')

        gen_time = time.time() - gen_start
        print(f"\n  -> 第 {gen+1} 代完成，耗时 {gen_time:.1f}s")
        print(f"  -> 历史最佳: [{best_name}] 胜率={best_win_rate*100:.1f}%")

    # 保存最终图表
    viz.save_plots(suffix='_final')
    print(f"\n{'='*70}")
    print(f"  可视化训练完成！")
    print(f"  最佳种群: [{best_name}] 胜率={best_win_rate*100:.1f}%")
    print(f"  图表保存至: ./models/viz/")
    print(f"{'='*70}")

    return best_net, best_name, best_win_rate


def main():
    parser = argparse.ArgumentParser(description='可视化遗传算法训练')
    parser.add_argument('--gens', type=int, default=10, help='进化代数')
    parser.add_argument('--iters', type=int, default=8, help='每代训练轮数')
    parser.add_argument('--workers', type=int, default=4, help='并行 worker 数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--viz-every', type=int, default=1, help='每几代保存图表')
    args = parser.parse_args()

    evolve_with_viz(
        num_generations=args.gens,
        iters_per_gen=args.iters,
        num_workers=args.workers,
        seed=args.seed,
        viz_every=args.viz_every,
    )


if __name__ == '__main__':
    main()
