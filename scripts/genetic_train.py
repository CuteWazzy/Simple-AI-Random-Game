"""
遗传算法 + 多配置并行训练框架。

设计：
  * 维护 N 个"种群"（subpopulation），每个种群有独立的配置：
    - 不同的温度调度
    - 不同的奖励权重（w_hit, w_def, w_item, w_self_aim）
    - 不同的激活函数（ReLU / Tanh / GELU / ELU）
  * 每代（generation）：
    1. 每个种群内部用自博弈 + REINFORCE 训练若干轮
    2. 跨种群对战评估（round-robin 锦标赛）
    3. 选择 top-K 种群作为"父代"
    4. 通过参数交叉（线性插值）和变异（高斯噪声）产生下一代
    5. 保留精英（elitism）

  * 多线程：用 ThreadPoolExecutor 并行训练多个种群
    PyTorch 在 CPU 上单线程推理最快，所以每个线程 set_num_threads(1)

参数保存到 ./models/genetic_model.pt
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
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 每个进程/线程单线程推理
torch.set_num_threads(1)

from game import (
    Game, PHASE_CHOOSE_TARGET, PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE,
    PHASE_EDIT_POS, PHASE_EDIT_DELTA, PHASE_REWARD_CHOICE,
    PHASE_GAME_OVER, NUM_PHASES,
)
from model import PolicyValueNet, masked_softmax
from numpy_net import NumpyNet, masked_softmax_np, action_mask_np
from train import (
    play_one_game as _play_one_game_orig,
    compute_returns, StepRecord,
)

# ---------- 种群配置 ----------

@dataclass
class PopConfig:
    """每个种群的配置。"""
    name: str
    temperature: float
    temperature_decay: float
    min_temperature: float
    eps_greedy: float
    lr: float
    # 奖励权重
    w_hp: float = 0.05
    w_opp: float = 0.03
    w_item: float = 0.05
    w_hit: float = 0.50
    w_def: float = 0.30
    w_useless: float = 0.0
    w_self_aim: float = 0.40
    # 激活函数：'relu' / 'tanh' / 'gelu' / 'elu'
    activation: str = 'relu'
    # 网络结构
    hidden_size: int = 128
    # 训练参数
    games_per_iter: int = 16
    batch_size: int = 128
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 1.0


# 4 个初始种群，配置差异明显
# ★ 高温训练：所有种群温度调高 0.4-0.6
# ★ 重新加入 w_useless 惩罚（小值，避免模式坍塌但抑制无效使用）
DEFAULT_CONFIGS = [
    PopConfig(
        name='A_relu_hot',
        temperature=1.8, temperature_decay=0.997, min_temperature=0.9,  # ★ 高温
        eps_greedy=0.20, lr=4e-4,
        w_hit=0.50, w_def=0.30, w_item=0.05, w_self_aim=0.40,
        w_useless=-0.03,
        activation='relu', hidden_size=128,
        games_per_iter=12, batch_size=128,
    ),
    PopConfig(
        name='B_gelu_balanced',
        temperature=1.4, temperature_decay=0.997, min_temperature=0.7,  # ★ 高温
        eps_greedy=0.15, lr=3e-4,
        w_hit=0.40, w_def=0.40, w_item=0.08, w_self_aim=0.50,
        w_useless=-0.02,
        activation='gelu', hidden_size=128,
        games_per_iter=12, batch_size=128,
    ),
    PopConfig(
        name='C_tanh_conservative',
        temperature=1.2, temperature_decay=0.998, min_temperature=0.6,  # ★ 高温
        eps_greedy=0.12, lr=2e-4,
        w_hit=0.30, w_def=0.50, w_item=0.10, w_self_aim=0.60,
        w_useless=-0.04,
        activation='tanh', hidden_size=128,
        games_per_iter=12, batch_size=128,
    ),
    PopConfig(
        name='D_elu_aggressive',
        temperature=2.0, temperature_decay=0.995, min_temperature=1.0,  # ★ 超高温
        eps_greedy=0.25, lr=5e-4,
        w_hit=0.70, w_def=0.20, w_item=0.03, w_self_aim=0.30,
        w_useless=-0.01,
        activation='elu', hidden_size=128,
        games_per_iter=12, batch_size=128,
    ),
]


# ---------- 网络工厂（支持不同激活函数） ----------

def make_net(config: PopConfig) -> PolicyValueNet:
    """根据 config 创建网络，替换激活函数。"""
    net = PolicyValueNet(hidden_size=config.hidden_size)
    # 替换 trunk 中的 ReLU
    if config.activation != 'relu':
        act = {
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
        }[config.activation]
        # trunk 是 Sequential，奇数索引是激活层
        new_layers = []
        for i, layer in enumerate(net.trunk):
            if i % 2 == 1:  # 激活层位置
                new_layers.append(act)
            else:
                new_layers.append(layer)
        net.trunk = nn.Sequential(*new_layers)
    return net


def load_or_init_net(config: PopConfig, base_ckpt_path: Optional[str] = None) -> PolicyValueNet:
    """初始化网络。如果 base_ckpt_path 提供且存在，加载基础参数
    （忽略 trunk 维度差异，仅加载匹配的层）。
    """
    net = make_net(config)
    if base_ckpt_path and os.path.exists(base_ckpt_path):
        try:
            ckpt = torch.load(base_ckpt_path, map_location='cpu', weights_only=False)
            model_state = ckpt['model']
            # 仅加载形状匹配的参数
            own_state = net.state_dict()
            loaded = 0
            for k, v in model_state.items():
                if k in own_state and own_state[k].shape == v.shape:
                    own_state[k] = v
                    loaded += 1
            net.load_state_dict(own_state)
            print(f"  [{config.name}] 加载基础参数 {loaded}/{len(own_state)} 个", flush=True)
        except Exception as e:
            print(f"  [{config.name}] 加载失败：{e}", flush=True)
    return net


# ---------- 单种群训练（继承自 train.py，但使用 config 的权重） ----------

def play_with_config(net, config: PopConfig,
                     num_players: int = 4,
                     rng: Optional[random.Random] = None,
                     device: torch.device = torch.device('cpu'),
                     np_net: Optional[NumpyNet] = None) -> Tuple[List[StepRecord], int]:
    """用 config 的奖励权重玩一局。

    如果提供 np_net（NumpyNet 实例），用 numpy 推理（快 2.6x）；
    否则用 torch net 推理。
    """
    if rng is None:
        rng = random.Random()
    g = Game(num_players, rng)
    records: List[StepRecord] = []

    while not g.is_done():
        phase = g.phase
        valid = g.get_valid_actions()
        if not valid:
            break
        dm = g.get_decision_maker()
        state_vec = g.encode_state()

        # ★ 优化：优先用 numpy 推理（快 2.6x）
        if np_net is not None:
            state_np = np.array(state_vec, dtype=np.float32)
            out_np = np_net(state_np)
            head_name = PolicyValueNet.head_for_phase(phase)
            logits_np = out_np[head_name][0]  # shape: (K,)
            mask_np = action_mask_np(phase, valid, num_players)  # ★ 纯 numpy，比 torch 快 17x
            probs_np = masked_softmax_np(logits_np / max(config.temperature, 1e-3), mask_np)
            value_np = float(out_np['value'][0])

            # 采样动作
            if rng.random() < config.eps_greedy:
                ai = rng.randrange(len(valid))
                head_idx = PolicyValueNet.action_to_head_index(phase, valid[ai])
            else:
                valid_idx = [PolicyValueNet.action_to_head_index(phase, a) for a in valid]
                valid_p = probs_np[valid_idx]
                s = valid_p.sum()
                if s <= 0:
                    head_idx = rng.choice(valid_idx)
                else:
                    valid_p = valid_p / s
                    ai = rng.choices(range(len(valid_idx)), weights=valid_p, k=1)[0]
                    head_idx = valid_idx[ai]

            action = PolicyValueNet.head_index_to_action(phase, head_idx)
            # 用 numpy 算 log_prob 和 entropy，转回 torch tensor
            log_prob = torch.tensor(float(np.log(probs_np[head_idx] + 1e-10)))
            value = torch.tensor(value_np)
            entropy = torch.tensor(float(-(probs_np * np.log(probs_np + 1e-10)).sum()))
        else:
            # torch 推理（兼容旧代码）
            state_t = torch.tensor([state_vec], dtype=torch.float32, device=device)
            with torch.no_grad():
                out = net(state_t)
            head_name = PolicyValueNet.head_for_phase(phase)
            logits = out[head_name][0]
            mask = PolicyValueNet.action_mask(phase, valid, num_players)
            probs = masked_softmax(logits / max(config.temperature, 1e-3), mask)
            probs_np = probs.cpu().numpy()

            if rng.random() < config.eps_greedy:
                ai = rng.randrange(len(valid))
                head_idx = PolicyValueNet.action_to_head_index(phase, valid[ai])
            else:
                valid_idx = [PolicyValueNet.action_to_head_index(phase, a) for a in valid]
                valid_p = probs_np[valid_idx]
                s = valid_p.sum()
                if s <= 0:
                    head_idx = rng.choice(valid_idx)
                else:
                    valid_p = valid_p / s
                    ai = rng.choices(range(len(valid_idx)), weights=valid_p, k=1)[0]
                    head_idx = valid_idx[ai]

            action = PolicyValueNet.head_index_to_action(phase, head_idx)
            log_prob = torch.log(probs[head_idx] + 1e-10)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            value = out['value'][0]

        # 快照
        dm_hp_before = g.hp[dm] + g.extra_hp[dm]
        dm_flip_before = g.flip[dm]
        dm_edit_before = g.edit[dm]
        dm_reroll_before = g.reroll[dm]
        opp_hp_before = sum(g.hp[j] + g.extra_hp[j] for j in range(g.num_players)
                            if j != dm and g.is_alive(j))
        dm_hit_delta_before = g.item_hit_delta[dm]
        dm_dmg_avoided_before = g.item_dmg_avoided[dm]
        dm_useless_before = g.item_useless_count[dm]
        dm_self_aim_hurt_before = g.item_self_aim_hurt[dm]

        rec = StepRecord(
            state=state_vec, phase=phase, valid_actions=valid,
            action_idx=head_idx, log_prob=log_prob, value=value,
            entropy=entropy, decision_maker=dm, reward_shaping=0.0,
        )

        g.step(action)

        # 复合 shaping
        dm_hp_after = g.hp[dm] + g.extra_hp[dm]
        hp_delta_self = dm_hp_after - dm_hp_before
        opp_hp_after = sum(g.hp[j] + g.extra_hp[j] for j in range(g.num_players)
                           if j != dm and g.is_alive(j))
        opp_hp_delta = opp_hp_after - opp_hp_before
        item_delta = ((g.flip[dm] - dm_flip_before)
                      + (g.edit[dm] - dm_edit_before)
                      + (g.reroll[dm] - dm_reroll_before))
        hit_delta_via_item = g.item_hit_delta[dm] - dm_hit_delta_before
        dmg_avoided_via_item = g.item_dmg_avoided[dm] - dm_dmg_avoided_before
        useless_via_item = g.item_useless_count[dm] - dm_useless_before
        self_aim_hurt_via_item = g.item_self_aim_hurt[dm] - dm_self_aim_hurt_before

        rec.reward_shaping = (
            config.w_hp * hp_delta_self
            + config.w_opp * (-opp_hp_delta)
            + config.w_item * item_delta
            + config.w_hit * hit_delta_via_item
            + config.w_def * dmg_avoided_via_item
            + config.w_useless * useless_via_item
            + config.w_self_aim * self_aim_hurt_via_item
        )

        records.append(rec)

    return records, g.winner


def train_one_population(net: PolicyValueNet, config: PopConfig,
                          num_iters: int, base_seed: int,
                          device: torch.device = torch.device('cpu')) -> Tuple[PolicyValueNet, Dict]:
    """训练一个种群 num_iters 轮。返回训练后的网络和统计。"""
    optimizer = Adam(net.parameters(), lr=config.lr)
    rng = random.Random(base_seed)
    stats = {'iters': 0, 'avg_loss': 0.0, 'avg_p_loss': 0.0, 'avg_v_loss': 0.0, 'avg_ent': 0.0}

    total_loss = 0.0
    total_p_loss = 0.0
    total_v_loss = 0.0
    total_ent = 0.0
    n_batches_total = 0

    for it in range(num_iters):
        net.eval()
        # ★ 创建 NumpyNet 用于数据收集加速（比 torch forward 快 2.6x）
        np_net = NumpyNet(net, activation=config.activation)
        all_records: List[StepRecord] = []
        all_returns: List[float] = []

        for gi in range(config.games_per_iter):
            recs, winner = play_with_config(net, config, num_players=4, rng=rng,
                                            device=device, np_net=np_net)
            returns = compute_returns(recs, winner, gamma=0.99)
            all_records.extend(recs)
            all_returns.extend(returns)

        net.train()
        n = len(all_records)
        if n == 0:
            continue
        idxs = list(range(n))
        rng.shuffle(idxs)

        for bstart in range(0, n, config.batch_size):
            bend = min(bstart + config.batch_size, n)
            batch_idx = idxs[bstart:bend]
            states = torch.tensor(
                [all_records[i].state for i in batch_idx],
                dtype=torch.float32, device=device)
            returns_t = torch.tensor(
                [all_returns[i] for i in batch_idx],
                dtype=torch.float32, device=device)

            out = net(states)
            phases = [all_records[i].phase for i in batch_idx]
            action_idxs = [all_records[i].action_idx for i in batch_idx]
            valid_actions_list = [all_records[i].valid_actions for i in batch_idx]

            log_probs = torch.zeros(len(batch_idx), device=device)
            entropies = torch.zeros(len(batch_idx), device=device)
            values = out['value']

            for k, i in enumerate(batch_idx):
                phase = phases[k]
                head_name = PolicyValueNet.head_for_phase(phase)
                logits = out[head_name][k]
                mask = PolicyValueNet.action_mask(phase, valid_actions_list[k], 4)
                probs = masked_softmax(logits, mask)
                a_idx = action_idxs[k]
                log_probs[k] = torch.log(probs[a_idx] + 1e-10)
                entropies[k] = -(probs * torch.log(probs + 1e-10)).sum()

            advantages = returns_t - values.detach()
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns_t)
            entropy_loss = -entropies.mean()
            loss = policy_loss + config.value_coeff * value_loss + config.entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            total_ent += -entropy_loss.item()
            n_batches_total += 1

        # 温度衰减
        config.temperature = max(config.min_temperature, config.temperature * config.temperature_decay)
        stats['iters'] += 1

    if n_batches_total > 0:
        stats['avg_loss'] = total_loss / n_batches_total
        stats['avg_p_loss'] = total_p_loss / n_batches_total
        stats['avg_v_loss'] = total_v_loss / n_batches_total
        stats['avg_ent'] = total_ent / n_batches_total

    return net, stats


# ---------- 多进程 worker（必须在顶层，可 pickle） ----------

def _train_worker(args):
    """子进程 worker：接收配置和初始参数，返回训练后的参数。"""
    config, init_state_dict, num_iters, base_seed = args
    # 子进程内设置单线程
    torch.set_num_threads(1)
    # 重建网络
    net = make_net(config)
    if init_state_dict is not None:
        own = net.state_dict()
        for k, v in init_state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        net.load_state_dict(own)
    # 训练
    net, stats = train_one_population(net, config, num_iters, base_seed)
    return config.name, net.state_dict(), stats, config.__dict__


# ---------- 锦标赛评估 ----------

def ai_pick_action_with_net(net: PolicyValueNet, g: Game,
                             temperature: float = 0.3,
                             device: torch.device = torch.device('cpu'),
                             rng: Optional[random.Random] = None) -> int:
    if rng is None:
        rng = random.Random()
    phase = g.phase
    valid = g.get_valid_actions()
    if not valid:
        return 0
    state = torch.tensor([g.encode_state()], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = net(state)
    head_name = PolicyValueNet.head_for_phase(phase)
    logits = out[head_name][0]
    mask = PolicyValueNet.action_mask(phase, valid, g.num_players)
    probs = masked_softmax(logits / max(temperature, 1e-3), mask)
    if temperature <= 1e-3:
        head_idx = int(probs.argmax().item())
    else:
        valid_idx = [PolicyValueNet.action_to_head_index(phase, a) for a in valid]
        valid_p = probs[valid_idx].cpu().numpy()
        s = valid_p.sum()
        if s <= 0:
            head_idx = rng.choice(valid_idx)
        else:
            valid_p = valid_p / s
            pick = rng.choices(range(len(valid_idx)), weights=valid_p, k=1)[0]
            head_idx = valid_idx[pick]
    return PolicyValueNet.head_index_to_action(phase, head_idx)


def play_match_between_nets(net_a: PolicyValueNet, net_b: PolicyValueNet,
                            seat_a: int, num_players: int = 4,
                            rng: Optional[random.Random] = None,
                            temperature: float = 0.3,
                            max_steps: int = 1000,
                            other_nets: dict = None) -> int:
    """让 net_a 在 seat_a 位置，其它位置用 net_b（或 other_nets 指定）。
    other_nets: {seat: net} 指定其它座位的网络。
    返回 winner 索引。
    """
    if rng is None:
        rng = random.Random()
    g = Game(num_players, rng)
    steps = 0
    while not g.is_done() and steps < max_steps:
        dm = g.get_decision_maker()
        if dm == seat_a:
            net = net_a
        elif other_nets and dm in other_nets:
            net = other_nets[dm]
        else:
            net = net_b
        a = ai_pick_action_with_net(net, g, temperature=temperature, rng=rng)
        g.step(a)
        steps += 1
    return g.winner


def tournament(populations: List[Dict],
               num_matches_per_pair: int = 3,
               seed: int = 0) -> Dict[str, float]:
    """增强锦标赛：多种对抗模式统计胜率。

    模式 1: 1v1 双座位轮换（每个 pair 对战 num_matches_per_pair × 2 局）
    模式 2: 1v1 多座位（0vs1, 0vs2, 0vs3, 1vs2, 1vs3, 2vs3）
    模式 3: 4 种群混战（4 个不同种群同台，每个种群占一个座位）
    模式 4: 多温度对抗（低温 0.2 + 中温 0.5）

    每个种群总对抗局数：约 num_matches_per_pair × 6（1v1）+ 4（混战）× 2（温度）= 较多
    """
    rng = random.Random(seed)
    win_counts = {p['name']: 0 for p in populations}
    total_counts = {p['name']: 0 for p in populations}

    # ---------- 模式 1+2: 1v1 多座位多温度对抗 ----------
    seat_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    temperatures = [0.2, 0.5]  # 低温 + 中温

    for i, p1 in enumerate(populations):
        for j, p2 in enumerate(populations):
            if i >= j:
                continue
            for seat_a, seat_b in seat_pairs:
                for temp in temperatures:
                    for m in range(num_matches_per_pair // 2 + 1):
                        # p1 在 seat_a，p2 在 seat_b
                        rng_match = random.Random(rng.random())
                        # 构造 other_nets：其它座位用 p2
                        other_nets = {s: p2['net'] for s in range(4) if s != seat_a}
                        other_nets[seat_b] = p2['net']
                        w = play_match_between_nets(p1['net'], p2['net'], seat_a,
                                                     num_players=4, rng=rng_match,
                                                     temperature=temp,
                                                     other_nets=other_nets)
                        if w == seat_a:
                            win_counts[p1['name']] += 1
                        elif w == seat_b:
                            win_counts[p2['name']] += 1
                        total_counts[p1['name']] += 1
                        total_counts[p2['name']] += 1

                        # 换座位：p2 在 seat_a，p1 在 seat_b
                        rng_match = random.Random(rng.random())
                        other_nets = {s: p1['net'] for s in range(4) if s != seat_a}
                        other_nets[seat_b] = p1['net']
                        w = play_match_between_nets(p2['net'], p1['net'], seat_a,
                                                     num_players=4, rng=rng_match,
                                                     temperature=temp,
                                                     other_nets=other_nets)
                        if w == seat_a:
                            win_counts[p2['name']] += 1
                        elif w == seat_b:
                            win_counts[p1['name']] += 1
                        total_counts[p2['name']] += 1
                        total_counts[p1['name']] += 1

    # ---------- 模式 3: 4 种群混战 ----------
    # 4 个种群各占一个座位，循环座位排列（只取部分排列加快速度）
    import itertools
    # 只取 4 种排列（而非 24 种）加快速度
    selected_perms = list(itertools.permutations(range(4)))[:4]
    for perm in selected_perms:
        seat_to_pop = [0] * 4
        for pop_idx, seat in enumerate(perm):
            seat_to_pop[seat] = pop_idx
        for temp in temperatures:
            rng_match = random.Random(rng.random())
            g = Game(4, rng_match)
            steps = 0
            while not g.is_done() and steps < 1000:
                dm = g.get_decision_maker()
                pop_idx = seat_to_pop[dm]
                net = populations[pop_idx]['net']
                a = ai_pick_action_with_net(net, g, temperature=temp, rng=rng_match)
                g.step(a)
                steps += 1
            if g.winner >= 0:
                winner_pop = seat_to_pop[g.winner]
                win_counts[populations[winner_pop]['name']] += 1
            for p in populations:
                total_counts[p['name']] += 1

    win_rates = {name: win_counts[name] / max(1, total_counts[name])
                 for name in win_counts}
    return win_rates


# ---------- 遗传操作 ----------

def crossover_nets(net_a: PolicyValueNet, net_b: PolicyValueNet,
                   alpha: float = 0.5) -> PolicyValueNet:
    """线性交叉：child = alpha * a + (1-alpha) * b。"""
    child = PolicyValueNet(hidden_size=net_a.trunk[0].out_features)
    sd_a = net_a.state_dict()
    sd_b = net_b.state_dict()
    sd_child = child.state_dict()
    for k in sd_child:
        if k in sd_a and k in sd_b and sd_a[k].shape == sd_b[k].shape:
            sd_child[k] = alpha * sd_a[k] + (1 - alpha) * sd_b[k]
    child.load_state_dict(sd_child)
    return child


def mutate_net(net: PolicyValueNet, mutation_rate: float = 0.1,
               mutation_std: float = 0.05,
               rng: Optional[random.Random] = None) -> PolicyValueNet:
    """高斯变异：以 mutation_rate 概率给每个参数加高斯噪声。"""
    if rng is None:
        rng = random.Random()
    sd = net.state_dict()
    for k in sd:
        if sd[k].dtype.is_floating_point:
            mask = torch.rand_like(sd[k]) < mutation_rate
            noise = torch.randn_like(sd[k]) * mutation_std
            sd[k] = sd[k] + mask * noise
    net.load_state_dict(sd)
    return net


# ---------- 主进化循环 ----------

def evolve(
    num_generations: int = 10,
    iters_per_gen: int = 20,
    base_ckpt_path: str = './models/model.pt',
    output_path: str = './models/genetic_model.pt',
    num_workers: int = 4,
    seed: int = 42,
    resume: bool = True,
):
    """主进化循环。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_gen = 0
    best_net = None
    best_win_rate = -1.0
    best_name = None

    # 续训：尝试加载已有 genetic_model.pt
    if resume and os.path.exists(output_path):
        try:
            old_ckpt = torch.load(output_path, map_location='cpu', weights_only=False)
            start_gen = old_ckpt.get('generation', 0)
            best_win_rate = old_ckpt.get('win_rate', -1.0)
            best_name = old_ckpt.get('best_name', None)
            # ★ 修复：加载历史最佳网络参数
            if 'model' in old_ckpt:
                best_net = PolicyValueNet()
                best_net.load_state_dict(old_ckpt['model'])
                best_net.eval()
            if best_name:
                print(f"[续训] 从第 {start_gen} 代继续，历史最佳: [{best_name}] 胜率={best_win_rate*100:.1f}%", flush=True)
            # 用历史最佳作为基础参数
            base_ckpt_path = output_path
        except Exception as e:
            print(f"[续训] 加载失败：{e}，从头开始", flush=True)

    print(f"\n{'='*70}")
    print(f"  遗传算法 + 多配置并行训练")
    print(f"  种群数: {len(DEFAULT_CONFIGS)} | 起始代: {start_gen+1} | 目标代: {start_gen+num_generations}")
    print(f"  每代训练: {iters_per_gen} 轮 | 并行 worker: {num_workers}")
    print(f"{'='*70}\n")

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
        })
        print(f"  初始化种群 [{config.name}] 激活={config.activation} T={config.temperature} lr={config.lr}", flush=True)

    for gen in range(start_gen, start_gen + num_generations):
        gen_start = time.time()
        print(f"\n{'='*70}")
        print(f"  第 {gen+1} 代 (目标 {start_gen + num_generations})")
        print(f"{'='*70}")

        # 阶段 1：多进程并行训练每个种群（真并行，绕过 GIL）
        print(f"\n[阶段 1] 多进程并行训练所有种群 ({iters_per_gen} 轮/种群)...")
        train_start = time.time()

        # 准备每个种群的参数：用当前种群的网络参数作为初始化
        worker_args = []
        for pop in populations:
            # 深拷贝 config，避免被子进程修改
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

        # 把训练后的参数加载回主进程的网络
        for pop in populations:
            if pop['name'] in results:
                state_dict, stats, cfg_dict = results[pop['name']]
                # 更新网络参数
                pop['net'].load_state_dict(state_dict)
                # 更新 config（温度衰减在子进程内发生）
                pop['config'].temperature = cfg_dict['temperature']
                pop['train_stats'] = stats
                print(f"  [{pop['name']}] loss={stats['avg_loss']:.4f} (p={stats['avg_p_loss']:.4f} "
                      f"v={stats['avg_v_loss']:.4f} ent={stats['avg_ent']:.4f}) "
                      f"T={pop['config'].temperature:.3f}", flush=True)

        train_time = time.time() - train_start
        print(f"  训练耗时: {train_time:.1f}s")

        # 阶段 2：锦标赛评估
        print(f"\n[阶段 2] 锦标赛评估（每个种群互相对战）...")
        tourney_start = time.time()
        win_rates = tournament(populations, num_matches_per_pair=3,
                               seed=seed + gen * 100)
        tourney_time = time.time() - tourney_start

        for pop in populations:
            pop['win_rate'] = win_rates.get(pop['name'], 0.0)
            pop['history'].append(pop['win_rate'])
        print(f"  锦标赛耗时: {tourney_time:.1f}s")
        print(f"  胜率:")
        for pop in sorted(populations, key=lambda x: -x['win_rate']):
            print(f"    {pop['name']}: {pop['win_rate']*100:.1f}%", flush=True)

        # 阶段 3：选择 + 交叉 + 变异
        if gen < num_generations - 1:
            print(f"\n[阶段 3] 遗传操作（选择/交叉/变异）...")
            # 按胜率排序
            sorted_pops = sorted(populations, key=lambda x: -x['win_rate'])
            elite = sorted_pops[0]
            print(f"  精英: [{elite['name']}] 胜率={elite['win_rate']*100:.1f}%")

            # 保留精英 + top-2 交叉产生后代 + 变异
            # 简化：保留 top-1 精英，其余用 top-2 交叉 + 变异
            new_populations = [elite]  # 精英直接保留

            top2 = sorted_pops[:2]
            for i in range(1, len(populations)):
                # 交叉 top-2
                alpha = random.random()
                child_net = crossover_nets(top2[0]['net'], top2[1]['net'], alpha=alpha)
                # 变异
                child_net = mutate_net(child_net,
                                       mutation_rate=0.1,
                                       mutation_std=0.03,
                                       rng=random.Random(seed + gen * 100 + i))
                # 配置随机选 top-2 之一
                child_config = deepcopy(random.choice(top2)['config'])
                child_config.name = sorted_pops[i]['name'] + f'_gen{gen+1}'
                new_populations.append({
                    'name': sorted_pops[i]['name'],  # 保留原名
                    'config': sorted_pops[i]['config'],  # 保留原配置
                    'net': child_net,
                    'win_rate': 0.0,
                    'history': sorted_pops[i]['history'],
                })

            populations = new_populations
            print(f"  产生 {len(new_populations)-1} 个后代（交叉+变异）")

        # 更新最佳
        for pop in populations:
            if pop['win_rate'] > best_win_rate:
                best_win_rate = pop['win_rate']
                best_net = deepcopy(pop['net'])
                best_name = pop['name']

        # 如果 best_net 还是 None（第一次没匹配），用当前精英
        if best_net is None:
            best_net = deepcopy(sorted(populations, key=lambda x: -x['win_rate'])[0]['net'])

        # 保存最佳
        ckpt = {
            'model': best_net.state_dict(),
            'iteration': (gen + 1) * iters_per_gen,
            'win_rate': best_win_rate,
            'best_name': best_name,
            'generation': gen + 1,
            'populations_history': {p['name']: p['history'] for p in populations},
        }
        torch.save(ckpt, output_path)
        gen_time = time.time() - gen_start
        print(f"\n  -> 第 {gen+1} 代完成，耗时 {gen_time:.1f}s")
        print(f"  -> 历史最佳: [{best_name}] 胜率={best_win_rate*100:.1f}%")
        print(f"  -> 已保存 {output_path}", flush=True)

    # 训练结束（不再覆盖保存，循环内已保存最终状态）
    print(f"\n{'='*70}")
    print(f"  进化训练完成！")
    print(f"  最佳种群: [{best_name}] 胜率={best_win_rate*100:.1f}%")
    print(f"  总训练轮数: {(start_gen + num_generations) * iters_per_gen}")
    print(f"  最终代数: {start_gen + num_generations}")
    print(f"  模型保存至: {output_path}")
    print(f"{'='*70}")

    return best_net, best_name, best_win_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gens', type=int, default=10,
                        help='进化代数')
    parser.add_argument('--iters', type=int, default=20,
                        help='每代每种群训练轮数')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行 worker 数')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    evolve(
        num_generations=args.gens,
        iters_per_gen=args.iters,
        num_workers=args.workers,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
