"""
人机对弈 / AI 观战脚本。

用法：
  python3 play.py watch          # AI vs AI 自对战，打印每步
  python3 play.py watch --n 100  # 跑 100 局统计胜率
  python3 play.py human          # 你和 3 个 AI 同台竞技
  python3 play.py human --players 4 --seat 0   # 你坐 1 号位

参数加载自 ./models/model.pt
"""

import os
import sys
import argparse
import random
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_num_threads(1)

from game import (
    Game, PHASE_CHOOSE_TARGET, PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE,
    PHASE_EDIT_POS, PHASE_EDIT_DELTA, PHASE_REWARD_CHOICE,
    PHASE_GAME_OVER, PHASE_NAMES,
)
from model import PolicyValueNet, masked_softmax

MODEL_PATH = "./models/model.pt"


def load_net(device: torch.device = torch.device('cpu')) -> PolicyValueNet:
    net = PolicyValueNet().to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        it = ckpt.get('iteration', '?')
        print(f"[已加载模型] 第 {it} 轮训练参数", file=sys.stderr)
    else:
        print(f"[警告] 未找到 {MODEL_PATH}，使用随机初始化网络", file=sys.stderr)
    net.eval()
    return net


def ai_pick_action(net: PolicyValueNet, g: Game,
                   temperature: float = 0.5,
                   device: torch.device = torch.device('cpu'),
                   rng: Optional[random.Random] = None) -> int:
    """让 AI 选一个动作。返回游戏 action（非 head 索引）。"""
    if rng is None:
        rng = random.Random()
    phase = g.phase
    valid = g.get_valid_actions()
    if not valid:
        return valid[0] if valid else 0
    state = torch.tensor([g.encode_state()], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = net(state)
    head_name = PolicyValueNet.head_for_phase(phase)
    logits = out[head_name][0]
    mask = PolicyValueNet.action_mask(phase, valid, g.num_players)
    probs = masked_softmax(logits / max(temperature, 1e-3), mask)
    if temperature <= 1e-3:
        # 贪心
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


# ---------- 渲染 ----------
def render(g: Game, file=sys.stdout):
    print(f"\n=== 回合 {g.turn_count + 1} ===", file=file)
    print(f"阶段: {PHASE_NAMES[g.phase]}", file=file)
    if g.phase != PHASE_CHOOSE_TARGET:
        print(f"当前玩家: P{g.current_player + 1}  目标: P{g.target + 1 if g.target >= 0 else '-'}", file=file)
        print(f"随机数: {g.ran} (位数={g.digits})  命中={g.nhurt}  flip+{g.nflip} edit+{g.nedit} reroll+{g.nreroll}", file=file)
    else:
        print(f"当前玩家: P{g.current_player + 1}（请选择攻击目标）", file=file)
    for i in range(g.num_players):
        alive = '存活' if g.is_alive(i) else '死亡'
        cur = ' <- 当前' if i == g.current_player else ''
        tgt = ' <- 目标' if i == g.target else ''
        iu = ' <- 道具决策者' if i == g.item_user and g.phase in (PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE, PHASE_EDIT_POS, PHASE_EDIT_DELTA) else ''
        print(f"  P{i+1}: HP={g.hp[i]:2d}+{g.extra_hp[i]:2d}  道具 f={g.flip[i]} e={g.edit[i]} r={g.reroll[i]}  [{alive}]{cur}{tgt}{iu}", file=file)


# ---------- 观战模式 ----------
def watch(net: PolicyValueNet, num_players: int = 4,
          num_games: int = 1, seed: int = 0, verbose: bool = True,
          temperature: float = 0.3):
    rng = random.Random(seed)
    wins = [0] * num_players
    total_steps = 0
    for gi in range(num_games):
        g = Game(num_players, random.Random(rng.random()))
        if verbose:
            print(f"\n>>>>> 第 {gi+1}/{num_games} 局 <<<<<")
        while not g.is_done():
            if verbose:
                render(g)
            a = ai_pick_action(net, g, temperature=temperature, rng=rng)
            if verbose:
                head_name = PolicyValueNet.head_for_phase(g.phase)
                print(f"  -> AI(P{g.get_decision_maker()+1}) 在 {PHASE_NAMES[g.phase]} 选择 action={a}", file=sys.stdout)
            g.step(a)
            if g.last_damage > 0 and verbose:
                tname = 'P' + str(g.last_damage_target + 1) if g.last_damage_target >= 0 else '无'
                aname = 'P' + str(g.last_attacker + 1) if g.last_attacker >= 0 else '无'
                print(f"  !! {aname} 对 {tname} 造成 {g.last_damage} 点伤害", file=sys.stdout)
            total_steps += 1
            if total_steps > 500000:
                break
        if verbose:
            print(f"\n===== 游戏结束 =====")
            render(g)
        if g.winner >= 0:
            wins[g.winner] += 1
        if verbose:
            print(f"赢家: P{g.winner + 1 if g.winner >= 0 else '无'}")
    print(f"\n=== 总计 {num_games} 局 ===")
    for i in range(num_players):
        print(f"  P{i+1}: {wins[i]} 胜 ({wins[i]/num_games*100:.1f}%)")
    return wins


# ---------- 人机模式 ----------
def human_play(net: PolicyValueNet, num_players: int = 4, human_seat: int = 0,
               seed: int = 0, ai_temperature: float = 0.3):
    rng = random.Random(seed)
    g = Game(num_players, random.Random(rng.random()))
    print(f"\n你扮演 P{human_seat + 1}，其余 {num_players - 1} 个玩家由 AI 控制。\n")
    while not g.is_done():
        render(g)
        dm = g.get_decision_maker()
        if dm == human_seat:
            # 人类回合
            valid = g.get_valid_actions()
            print(f"\n[你的回合] 阶段={PHASE_NAMES[g.phase]} 合法动作: {valid}")
            if g.phase == PHASE_CHOOSE_TARGET:
                print("  选择目标玩家编号 (1-indexed):")
            elif g.phase == PHASE_ITEM_DECISION:
                print("  是否使用道具？0=否 1=是:")
            elif g.phase == PHASE_ITEM_CHOICE:
                print("  选择道具: 0=结束 1=flip 2=edit 3=reroll:")
            elif g.phase == PHASE_EDIT_POS:
                print(f"  选择位置 (0..{len(str(g.ran))-1}):")
            elif g.phase == PHASE_EDIT_DELTA:
                print("  选择变化量: 0=+1 1=-1:")
            elif g.phase == PHASE_REWARD_CHOICE:
                print("  选择奖励: 1=flip 2=edit 3=reroll:")
            try:
                a = int(input("  > ").strip())
            except (EOFError, ValueError):
                print("输入结束，退出。")
                return
            # 转换 1-indexed 目标
            if g.phase == PHASE_CHOOSE_TARGET:
                a = a - 1
            if a not in valid:
                print(f"  非法动作 {a}，跳过。")
                continue
            g.step(a)
        else:
            # AI 回合
            a = ai_pick_action(net, g, temperature=ai_temperature, rng=rng)
            print(f"\n  [AI P{dm+1}] 在 {PHASE_NAMES[g.phase]} 选择 action={a}")
            g.step(a)
        if g.last_damage > 0:
            tname = 'P' + str(g.last_damage_target + 1) if g.last_damage_target >= 0 else '无'
            aname = 'P' + str(g.last_attacker + 1) if g.last_attacker >= 0 else '无'
            print(f"  !! {aname} 对 {tname} 造成 {g.last_damage} 点伤害")
    print(f"\n===== 游戏结束 =====")
    render(g)
    if g.winner == human_seat:
        print("你赢了！")
    elif g.winner >= 0:
        print(f"AI P{g.winner+1} 获胜。")
    else:
        print("没有玩家存活。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['watch', 'human'], default='watch',
                        nargs='?')
    parser.add_argument('--n', type=int, default=1, help='观战局数')
    parser.add_argument('--players', type=int, default=4)
    parser.add_argument('--seat', type=int, default=0, help='人类座位（0-indexed）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temp', type=float, default=0.3,
                        help='AI 温度，越低越贪心')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    net = load_net()
    if args.mode == 'watch':
        watch(net, num_players=args.players, num_games=args.n,
              seed=args.seed, verbose=not args.quiet, temperature=args.temp)
    else:
        human_play(net, num_players=args.players, human_seat=args.seat,
                   seed=args.seed, ai_temperature=args.temp)


if __name__ == '__main__':
    main()
