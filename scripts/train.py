"""
自博弈训练脚本。

算法：REINFORCE 带基线（policy gradient with value baseline），加 entropy 正则。
- 共享一个网络为所有玩家做决策（self-play）
- 每局结束后，赢家步奖励 +1，输家步奖励 -1，并加少量 shaping（造成/受到伤害）
- 优势 = return - value
- Loss = -log_pi * advantage + c_v * value_loss - c_e * entropy

参数保存到 ./models/model.pt
每轮训练后自动覆盖，并保留最近 N 个 checkpoint。
"""

import os
import sys
import time
import random
import argparse
from collections import deque
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# 单线程推理在 CPU 上对小网络快得多（避免多线程调度开销）
torch.set_num_threads(1)

# 确保能 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import (
    Game, PHASE_CHOOSE_TARGET, PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE,
    PHASE_EDIT_POS, PHASE_EDIT_DELTA, PHASE_REWARD_CHOICE,
    PHASE_GAME_OVER, NUM_PHASES,
)
from model import (
    PolicyValueNet, masked_softmax,
    PolicyValueNet as PVN,
)

# ---------- 路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = "./models"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
MODEL_PATH = os.path.join(DOWNLOAD_DIR, "model.pt")
WORKLOG_PATH = "./worklog.md"


# ---------- 单局自博弈 ----------
class StepRecord:
    __slots__ = ('state', 'phase', 'valid_actions', 'action_idx',
                 'log_prob', 'value', 'entropy', 'decision_maker',
                 'reward_shaping')
    def __init__(self, state, phase, valid_actions, action_idx,
                 log_prob, value, entropy, decision_maker, reward_shaping=0.0):
        self.state = state
        self.phase = phase
        self.valid_actions = valid_actions
        self.action_idx = action_idx
        self.log_prob = log_prob
        self.value = value
        self.entropy = entropy
        self.decision_maker = decision_maker
        self.reward_shaping = reward_shaping


def play_one_game(net: PolicyValueNet, num_players: int = 4,
                  rng: random.Random = None, temperature: float = 1.0,
                  device: torch.device = torch.device('cpu'),
                  eps_greedy: float = 0.0) -> Tuple[List[StepRecord], int]:
    """用当前策略玩一局，返回所有决策步与赢家索引。"""
    if rng is None:
        rng = random.Random()
    g = Game(num_players, rng)
    records: List[StepRecord] = []

    while not g.is_done():
        phase = g.phase
        valid = g.get_valid_actions()
        if not valid:
            # 不应该发生，但防御性处理
            break
        dm = g.get_decision_maker()
        state_vec = g.encode_state()
        state_t = torch.tensor([state_vec], dtype=torch.float32, device=device)

        with torch.no_grad():
            out = net(state_t)
        head_name = PolicyValueNet.head_for_phase(phase)
        logits = out[head_name][0]  # shape: [K]
        mask = PolicyValueNet.action_mask(phase, valid, num_players)
        probs = masked_softmax(logits / max(temperature, 1e-3), mask)
        probs_np = probs.cpu().numpy()

        # epsilon-greedy
        if rng.random() < eps_greedy:
            ai = rng.randrange(len(valid))
            head_idx = PolicyValueNet.action_to_head_index(phase, valid[ai])
        else:
            # 从合法动作里按 probs 采样
            valid_idx = [PolicyValueNet.action_to_head_index(phase, a) for a in valid]
            valid_p = probs_np[valid_idx]
            s = valid_p.sum()
            if s <= 0:
                ai = rng.randrange(len(valid))
                head_idx = valid_idx[ai]
            else:
                valid_p = valid_p / s
                ai = rng.choices(range(len(valid_idx)), weights=valid_p, k=1)[0]
                head_idx = valid_idx[ai]

        action = PolicyValueNet.head_index_to_action(phase, head_idx)
        log_prob = torch.log(probs[head_idx] + 1e-10)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        value = out['value'][0]

        # 决策者本步前的 HP / 道具快照
        dm_hp_before = g.hp[dm] + g.extra_hp[dm]
        dm_flip_before = g.flip[dm]
        dm_edit_before = g.edit[dm]
        dm_reroll_before = g.reroll[dm]
        # 对手 HP 总和（用于衡量"对敌人造成的整体压制"）
        opp_hp_before = sum(g.hp[j] + g.extra_hp[j] for j in range(g.num_players)
                            if j != dm and g.is_alive(j))
        # 本步前的道具追踪快照
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

        # ---- 复合奖励 shaping：血量 + 道具 + 道具命中/防御 + 无效惩罚 + 反向防御 ----
        # 1) 决策者 HP 变化：HP 增加正奖励，HP 减少负奖励（强化自我保护）
        dm_hp_after = g.hp[dm] + g.extra_hp[dm]
        hp_delta_self = dm_hp_after - dm_hp_before
        # 2) 对手总 HP 变化：对手 HP 减少正奖励（造成伤害被认可）
        opp_hp_after = sum(g.hp[j] + g.extra_hp[j] for j in range(g.num_players)
                           if j != dm and g.is_alive(j))
        opp_hp_delta = opp_hp_after - opp_hp_before
        # 3) 道具净增量：获得 +，消耗 -
        item_delta = ((g.flip[dm] - dm_flip_before)
                      + (g.edit[dm] - dm_edit_before)
                      + (g.reroll[dm] - dm_reroll_before))
        # 4) 本步使用道具后造成的"额外命中数"
        hit_delta_via_item = g.item_hit_delta[dm] - dm_hit_delta_before
        # 5) 本步使用道具后"减少的伤害"
        dmg_avoided_via_item = g.item_dmg_avoided[dm] - dm_dmg_avoided_before
        # 6) 无效道具使用次数（扣分）
        useless_via_item = g.item_useless_count[dm] - dm_useless_before
        # 7) ★ 修复盲区：防御者用 flip 让自瞄者反伤的命中数
        self_aim_hurt_via_item = g.item_self_aim_hurt[dm] - dm_self_aim_hurt_before

        # 复合函数：
        #   w_hp           * ΔHP_self         自己 HP 变化（核心）
        #   w_opp          * (-ΔHP_opp)        对手 HP 减少（造成伤害）
        #   w_item         * Δitems            道具净增量（鼓励积累，但不鼓励浪费）
        #   w_hit          * hit_via_item      道具导致的额外命中
        #   w_def          * dmg_avoided       道具导致的伤害避免
        #   w_useless      * useless_via_item  无效道具使用扣分
        #   w_self_aim     * self_aim_hurt     ★ 反向防御奖励（让自瞄者反伤）
        w_hp = 0.05
        w_opp = 0.03
        w_item = 0.05       # ★ 调高：从 0.02 → 0.05（强鼓励积累道具）
        w_hit = 0.50       # ★ 调高：从 0.30 → 0.50（强鼓励道具命中）
        w_def = 0.30       # ★ 调高：从 0.20 → 0.30（强鼓励防御性道具）
        w_useless = 0.0    # ★ 设为 0：完全移除无效惩罚，让 AI 重新学
        w_self_aim = 0.40  # ★ 调高：从 0.25 → 0.40（强鼓励反向防御）
        rec.reward_shaping = (
            w_hp * hp_delta_self
            + w_opp * (-opp_hp_delta)
            + w_item * item_delta
            + w_hit * hit_delta_via_item
            + w_def * dmg_avoided_via_item
            + w_useless * useless_via_item
            + w_self_aim * self_aim_hurt_via_item
        )

        records.append(rec)

    return records, g.winner


# ---------- 计算回报 ----------
def compute_returns(records: List[StepRecord], winner: int,
                    gamma: float = 0.99,
                    win_reward: float = 1.0,
                    lose_reward: float = -1.0) -> List[float]:
    """计算每步的 return。
    每步的"终局信号"基于该步的决策者最终是否获胜，按 gamma^(n-1-i) 折扣。
    再加上 shaping 的折扣累计。
    """
    n = len(records)
    # 先算 shaping 的折扣累计
    shaping_returns = [0.0] * n
    running = 0.0
    for i in reversed(range(n)):
        running = records[i].reward_shaping + gamma * running
        shaping_returns[i] = running
    # 每步的 return = shaping_return + gamma^distance * outcome(dm_i)
    returns = [0.0] * n
    for i in range(n):
        outcome = win_reward if records[i].decision_maker == winner else lose_reward
        distance = n - 1 - i
        returns[i] = shaping_returns[i] + (gamma ** distance) * outcome
    return returns


# ---------- 训练循环 ----------
def train(
    num_iterations: int = 200,
    games_per_iter: int = 64,
    num_players: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 1.0,
    temperature: float = 1.0,
    temperature_decay: float = 0.995,
    min_temperature: float = 0.5,
    eps_greedy: float = 0.15,  # ★ 调高：从 0.02 → 0.15 强制探索
    save_every: int = 5,
    log_every: int = 1,
    seed: int = 42,
    device: torch.device = torch.device('cpu'),
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    net = PolicyValueNet().to(device)
    optimizer = Adam(net.parameters(), lr=lr)

    # 加载已有 checkpoint 继续训练
    start_iter = 0
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_iter = ckpt.get('iteration', 0)
            print(f"[续训] 从第 {start_iter} 轮继续，加载 {MODEL_PATH}")
        except Exception as e:
            print(f"[续训] 加载失败：{e}，从头开始")

    rng = random.Random(seed)

    win_counts = [0] * num_players
    recent_returns = deque(maxlen=200)
    recent_lengths = deque(maxlen=200)

    for it in range(start_iter, start_iter + num_iterations):
        iter_start = time.time()
        net.eval()
        all_records: List[StepRecord] = []
        all_returns: List[float] = []

        for gi in range(games_per_iter):
            recs, winner = play_one_game(
                net, num_players=num_players, rng=rng,
                temperature=temperature, device=device,
                eps_greedy=eps_greedy,
            )
            if winner >= 0:
                win_counts[winner] += 1
            returns = compute_returns(recs, winner, gamma=gamma)
            all_records.extend(recs)
            all_returns.extend(returns)
            recent_lengths.append(len(recs))

        # 训练
        net.train()
        n = len(all_records)
        idxs = list(range(n))
        rng.shuffle(idxs)

        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        n_batches = 0

        for bstart in range(0, n, batch_size):
            bend = min(bstart + batch_size, n)
            batch_idx = idxs[bstart:bend]
            states = torch.tensor(
                [all_records[i].state for i in batch_idx],
                dtype=torch.float32, device=device)
            returns_t = torch.tensor(
                [all_returns[i] for i in batch_idx],
                dtype=torch.float32, device=device)

            out = net(states)
            # 把每一步的 phase 收集起来
            phases = [all_records[i].phase for i in batch_idx]
            action_idxs = [all_records[i].action_idx for i in batch_idx]
            valid_actions_list = [all_records[i].valid_actions for i in batch_idx]
            dms = [all_records[i].decision_maker for i in batch_idx]

            # 逐 phase 计算 log_prob（不能直接整 batch 用同一个 head）
            # 简化：按 phase 分组，但为了向量化，我们用 mask 直接选
            log_probs = torch.zeros(len(batch_idx), device=device)
            entropies = torch.zeros(len(batch_idx), device=device)
            values = out['value']  # [B]

            for k, i in enumerate(batch_idx):
                phase = phases[k]
                head_name = PolicyValueNet.head_for_phase(phase)
                logits = out[head_name][k]
                mask = PolicyValueNet.action_mask(phase, valid_actions_list[k], num_players)
                probs = masked_softmax(logits, mask)
                a_idx = action_idxs[k]
                log_probs[k] = torch.log(probs[a_idx] + 1e-10)
                entropies[k] = -(probs * torch.log(probs + 1e-10)).sum()

            advantages = returns_t - values.detach()
            # 标准化优势
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns_t)
            entropy_loss = -entropies.mean()
            loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            total_ent += -entropy_loss.item()
            n_batches += 1

        if n_batches > 0:
            total_loss /= n_batches
            total_p_loss /= n_batches
            total_v_loss /= n_batches
            total_ent /= n_batches

        # 温度衰减
        temperature = max(min_temperature, temperature * temperature_decay)

        iter_time = time.time() - iter_start

        if (it + 1) % log_every == 0:
            total_wins = sum(win_counts) or 1
            win_dist = " ".join(f"P{i+1}:{win_counts[i]}/{total_wins}"
                                for i in range(num_players))
            print(f"[Iter {it+1}/{start_iter+num_iterations}] "
                  f"games={games_per_iter} steps/game={sum(recent_lengths)/max(1,len(recent_lengths)):.1f} "
                  f"loss={total_loss:.4f} (p={total_p_loss:.4f} v={total_v_loss:.4f} ent={total_ent:.4f}) "
                  f"T={temperature:.3f} time={iter_time:.1f}s")
            print(f"    胜率分布: {win_dist}")

        if (it + 1) % save_every == 0 or it + 1 == start_iter + num_iterations:
            ckpt = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': it + 1,
                'num_players': num_players,
                'temperature': temperature,
            }
            torch.save(ckpt, MODEL_PATH)
            # 保留最近 3 个备份
            for k in range(2, 5):
                old_path = os.path.join(DOWNLOAD_DIR, f"model_iter_{it+1-k*save_every}.pt")
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except Exception:
                        pass
            # 也存一个带 iter 号的快照
            snap_path = os.path.join(DOWNLOAD_DIR, f"model_iter_{it+1}.pt")
            torch.save(ckpt, snap_path)
            # 清理老快照
            snaps = sorted([f for f in os.listdir(DOWNLOAD_DIR)
                            if f.startswith("model_iter_") and f.endswith(".pt")])
            for old in snaps[:-3]:
                try:
                    os.remove(os.path.join(DOWNLOAD_DIR, old))
                except Exception:
                    pass
            print(f"    -> 已保存 {MODEL_PATH}")

    # 写 worklog
    try:
        with open(WORKLOG_PATH, 'a', encoding='utf-8') as f:
            f.write("\n---\n")
            f.write("Task ID: ai-training\n")
            f.write("Agent: main\n")
            f.write(f"Task: 训练数字对战游戏的神经网络 AI\n")
            f.write("\nWork Log:\n")
            f.write(f"- 完成 {num_iterations} 轮自博弈训练，每轮 {games_per_iter} 局\n")
            f.write(f"- 最终温度 {temperature:.3f}\n")
            f.write(f"- 参数保存至 {MODEL_PATH}\n")
            f.write("\nStage Summary:\n")
            f.write(f"- 训练了 4 玩家游戏，参数量 {sum(p.numel() for p in net.parameters())}\n")
            f.write(f"- 胜率分布: {win_counts}\n")
    except Exception as e:
        print(f"worklog 写入失败：{e}")

    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--games', type=int, default=64,
                        help='每轮自博弈局数')
    parser.add_argument('--players', type=int, default=4,
                        help='玩家数')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(
        num_iterations=args.iters,
        games_per_iter=args.games,
        num_players=args.players,
        lr=args.lr,
        temperature=args.temperature,
        batch_size=args.batch,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
