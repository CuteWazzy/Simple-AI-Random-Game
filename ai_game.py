"""
AI 数字对战 - 主入口程序
集成观战、人机对弈、训练于一体。

无参数：进入交互式菜单
参数：
  watch [seed] [temp] [max_turns]   观战 AI 自对战
  human [seat] [temp]               人机对弈
  stats                             查看模型统计
  train [gens] [iters]              快速训练
"""

import sys
import os
import random

# 确保能找到 scripts 目录（多种路径兼容）
def _setup_path():
    """设置模块搜索路径，兼容多种运行方式。"""
    candidates = []
    # 1. 脚本所在目录的 scripts 子目录
    base = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(base, 'scripts'))
    # 2. 脚本所在目录本身
    candidates.append(base)
    # 3. PyInstaller 打包后的 _MEIPASS
    if hasattr(sys, '_MEIPASS'):
        candidates.append(sys._MEIPASS)
        candidates.append(os.path.join(sys._MEIPASS, 'scripts'))
    # 4. 当前目录的 scripts
    candidates.append(os.path.join(os.getcwd(), 'scripts'))
    for p in candidates:
        if p and os.path.exists(os.path.join(p, 'game.py')):
            sys.path.insert(0, p)
            return p
    # 如果都没找到，把所有候选加入 path
    for p in candidates:
        if p and os.path.exists(p):
            sys.path.insert(0, p)

SCRIPT_DIR = _setup_path()

import torch
torch.set_num_threads(1)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from game import (
    Game, PHASE_CHOOSE_TARGET, PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE,
    PHASE_EDIT_POS, PHASE_EDIT_DELTA, PHASE_REWARD_CHOICE,
    PHASE_GAME_OVER, PHASE_NAMES,
)
from model import PolicyValueNet, masked_softmax
from numpy_net import NumpyNet, masked_softmax_np, action_mask_np


# ---------- 模型路径 ----------

def find_model():
    """查找模型文件。"""
    candidates = []
    base = os.path.dirname(os.path.abspath(__file__))
    # PyInstaller 打包后
    if hasattr(sys, '_MEIPASS'):
        candidates.append(os.path.join(sys._MEIPASS, 'model.pt'))
        candidates.append(os.path.join(sys._MEIPASS, 'models', 'model.pt'))
        candidates.append(os.path.join(sys._MEIPASS, 'genetic_model.pt'))
    # 脚本所在目录的 models 子目录
    candidates.append(os.path.join(base, 'models', 'model.pt'))
    candidates.append(os.path.join(base, 'model.pt'))
    # scripts 目录的 models
    candidates.append(os.path.join(SCRIPT_DIR, 'models', 'model.pt'))
    candidates.append(os.path.join(SCRIPT_DIR, 'model.pt'))
    # 当前目录
    candidates.append(os.path.join(os.getcwd(), 'models', 'model.pt'))
    candidates.append(os.path.join(os.getcwd(), 'model.pt'))
    for path in candidates:
        if os.path.isfile(path):  # 必须是文件不是目录
            return path
    return None


def load_net():
    """加载模型。"""
    model_path = find_model()
    net = PolicyValueNet()
    if model_path:
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            net.load_state_dict(ckpt['model'])
            it = ckpt.get('generation', ckpt.get('iteration', '?'))
            wr = ckpt.get('win_rate', 0)
            print(f"[已加载模型] 第 {it} 代训练参数，历史最佳胜率 {wr*100:.1f}%", file=sys.stderr)
        except Exception as e:
            print(f"[警告] 加载模型失败：{e}，使用随机初始化", file=sys.stderr)
    else:
        print("[警告] 未找到模型文件，使用随机初始化", file=sys.stderr)
    net.eval()
    return net


# ---------- AI 决策 ----------

def ai_pick_action(net, g, temperature=0.3, rng=None):
    if rng is None:
        rng = random.Random()
    phase = g.phase
    valid = g.get_valid_actions()
    if not valid:
        return 0
    state = torch.tensor([g.encode_state()], dtype=torch.float32)
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


# ---------- 渲染 ----------

def render_hp(g):
    parts = []
    for i in range(g.num_players):
        if g.is_alive(i):
            tag = []
            if i == g.current_player: tag.append('当前')
            if i == g.target: tag.append('目标')
            t = '/'.join(tag) if tag else ''
            hp_str = f"{g.hp[i]}+{g.extra_hp[i]}" if g.extra_hp[i] > 0 else f"{g.hp[i]}"
            parts.append(f"P{i+1}({hp_str}{'['+t+']' if t else ''})")
        else:
            parts.append(f"~~P{i+1}~~")
    return '  '.join(parts)


def count_digits(x):
    return 1 if x == 0 else len(str(x))


def watch_game(net, seed=0, temperature=0.3, max_turns=80):
    """观战一局 AI 自对战。"""
    rng = random.Random(seed)
    g = Game(4, random.Random(rng.random()))
    print(f"\n{'='*60}")
    print(f"  AI 自对战（温度 {temperature}，种子 {seed}）")
    print(f"{'='*60}")
    print(f"\n初始: {render_hp(g)}")

    prev_turn = -1
    last_ran = None
    last_nhurt = None
    item_events = []

    while not g.is_done() and g.turn_count < max_turns:
        if g.turn_count != prev_turn:
            if prev_turn >= 0 and (last_ran is not None or item_events):
                if last_ran is not None:
                    dmg_str = f"命中{last_nhurt}" if last_nhurt > 0 else "未命中"
                    print(f"  └ 抽数 {last_ran} (位数={count_digits(last_ran)}) {dmg_str}")
                if item_events:
                    print(f"  └ 道具: {'; '.join(item_events)}")
                if g.last_damage > 0:
                    aname = f"P{g.last_attacker+1}" if g.last_attacker >= 0 else '?'
                    tname = f"P{g.last_damage_target+1}" if g.last_damage_target >= 0 else '?'
                    print(f"  ★ {aname} 对 {tname} 造成 {g.last_damage} 点伤害")
            prev_turn = g.turn_count
            last_ran = None
            last_nhurt = None
            item_events = []
            print(f"\n[回合 {g.turn_count + 1}] {render_hp(g)}")

        phase = g.phase
        dm = g.get_decision_maker()

        if phase == PHASE_CHOOSE_TARGET:
            a = ai_pick_action(net, g, temperature=temperature, rng=rng)
            target_name = "自己(自瞄)" if a == dm else f"P{a+1}"
            print(f"  P{dm+1} → 攻击 {target_name}")
            g.step(a)
            last_ran = g.ran
            last_nhurt = g.nhurt
        elif phase == PHASE_ITEM_DECISION:
            a = ai_pick_action(net, g, temperature=temperature, rng=rng)
            g.step(a)
        elif phase == PHASE_ITEM_CHOICE:
            a = ai_pick_action(net, g, temperature=temperature, rng=rng)
            if a == 0:
                g.step(a)
            elif a == 1:
                item_events.append(f"P{dm+1} flip")
                g.step(a)
                last_ran = g.ran
                last_nhurt = g.nhurt
            elif a == 2:
                g.step(2)
                pos = ai_pick_action(net, g, temperature=temperature, rng=rng)
                g.step(pos)
                delta = ai_pick_action(net, g, temperature=temperature, rng=rng)
                sign = '+' if delta == 0 else '-'
                item_events.append(f"P{dm+1} edit@{pos}{sign}1")
                g.step(delta)
                last_ran = g.ran
                last_nhurt = g.nhurt
            elif a == 3:
                item_events.append(f"P{dm+1} reroll")
                g.step(a)
                last_ran = g.ran
                last_nhurt = g.nhurt
        elif phase == PHASE_REWARD_CHOICE:
            a = ai_pick_action(net, g, temperature=temperature, rng=rng)
            items = {1: 'flip', 2: 'edit', 3: 'reroll'}
            item_events.append(f"奖励={items[a]}")
            g.step(a)
            last_ran = g.ran
            last_nhurt = g.nhurt

    # 收尾
    if last_ran is not None or item_events:
        if last_ran is not None:
            dmg_str = f"命中{last_nhurt}" if last_nhurt > 0 else "未命中"
            print(f"  └ 抽数 {last_ran} (位数={count_digits(last_ran)}) {dmg_str}")
        if item_events:
            print(f"  └ 道具: {'; '.join(item_events)}")
        if g.last_damage > 0:
            aname = f"P{g.last_attacker+1}" if g.last_attacker >= 0 else '?'
            tname = f"P{g.last_damage_target+1}" if g.last_damage_target >= 0 else '?'
            print(f"  ★ {aname} 对 {tname} 造成 {g.last_damage} 点伤害")

    print(f"\n最终: {render_hp(g)}")
    if g.is_done():
        print(f"★ 赢家: P{g.winner+1}" if g.winner >= 0 else "没有玩家存活")
    else:
        print(f"(已展示前 {max_turns} 回合)")


def human_play(net, human_seat=0, seed=0, ai_temperature=0.3):
    """人机对弈。"""
    rng = random.Random(seed)
    g = Game(4, random.Random(rng.random()))
    print(f"\n{'='*60}")
    print(f"  人机对弈（你扮演 P{human_seat+1}）")
    print(f"{'='*60}")

    while not g.is_done():
        print(f"\n[回合 {g.turn_count + 1}] {render_hp(g)}")
        dm = g.get_decision_maker()
        if dm == human_seat:
            # 人类回合
            phase = g.phase
            valid = g.get_valid_actions()
            print(f"\n[你的回合] 阶段={PHASE_NAMES[phase]}")
            if phase == PHASE_CHOOSE_TARGET:
                print(f"  当前数字: {g.ran} (位数={count_digits(g.ran)}) 命中={g.nhurt}" if g.ran else "")
                print(f"  选择目标: {valid} (输入玩家编号 1-4)")
            elif phase == PHASE_ITEM_DECISION:
                print(f"  当前数字: {g.ran} 命中={g.nhurt}")
                print(f"  是否使用道具? 0=否 1=是")
            elif phase == PHASE_ITEM_CHOICE:
                print(f"  当前数字: {g.ran} 命中={g.nhurt}")
                print(f"  你的道具: flip={g.flip[dm]} edit={g.edit[dm]} reroll={g.reroll[dm]}")
                print(f"  选择: 0=结束 1=flip 2=edit 3=reroll")
            elif phase == PHASE_EDIT_POS:
                print(f"  数字: {g.ran}")
                print(f"  选择位置 (0-{len(str(g.ran))-1})")
            elif phase == PHASE_EDIT_DELTA:
                print(f"  选择变化: 0=+1 1=-1")
            elif phase == PHASE_REWARD_CHOICE:
                print(f"  4 位奖励! 选择: 1=flip 2=edit 3=reroll")

            try:
                a = int(input("  > ").strip())
            except (EOFError, ValueError):
                print("输入结束，退出。")
                return

            if phase == PHASE_CHOOSE_TARGET:
                a = a - 1  # 1-indexed -> 0-indexed

            if a not in valid:
                print(f"  非法动作 {a}，合法: {valid}")
                continue
            g.step(a)
        else:
            # AI 回合
            phase = g.phase
            a = ai_pick_action(net, g, temperature=ai_temperature, rng=rng)
            if phase == PHASE_CHOOSE_TARGET:
                target_name = "自己(自瞄)" if a == dm else f"P{a+1}"
                print(f"  [AI P{dm+1}] 攻击 {target_name}")
            elif phase == PHASE_ITEM_DECISION:
                pass  # 不打印 y/n
            elif phase == PHASE_ITEM_CHOICE:
                items = ['', 'flip', 'edit', 'reroll']
                if a != 0:
                    print(f"  [AI P{dm+1}] 使用 {items[a]}")
            elif phase == PHASE_REWARD_CHOICE:
                items = {1: 'flip', 2: 'edit', 3: 'reroll'}
                print(f"  [AI P{dm+1}] 奖励选择 {items[a]}")
            g.step(a)

        if g.last_damage > 0:
            aname = f"P{g.last_attacker+1}" if g.last_attacker >= 0 else '?'
            tname = f"P{g.last_damage_target+1}" if g.last_damage_target >= 0 else '?'
            print(f"  ★ {aname} 对 {tname} 造成 {g.last_damage} 点伤害")

    print(f"\n{'='*60}")
    print(f"  游戏结束")
    print(f"{'='*60}")
    print(f"最终: {render_hp(g)}")
    if g.winner == human_seat:
        print("你赢了！")
    elif g.winner >= 0:
        print(f"AI P{g.winner+1} 获胜。")
    else:
        print("没有玩家存活。")


def show_stats(net):
    """显示模型统计。"""
    model_path = find_model()
    print(f"\n{'='*60}")
    print(f"  模型统计")
    print(f"{'='*60}")
    if model_path:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"模型路径: {model_path}")
        print(f"代数: {ckpt.get('generation', ckpt.get('iteration', '?'))}")
        print(f"最佳种群: {ckpt.get('best_name', '?')}")
        print(f"历史最佳胜率: {ckpt.get('win_rate', 0)*100:.1f}%")
    else:
        print("未找到模型文件")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"参数量: {n_params}")
    print(f"状态维度: {Game.STATE_SIZE}")


def interactive_menu(net):
    """交互式菜单。"""
    while True:
        print(f"\n{'='*60}")
        print(f"  AI 数字对战 - 主菜单")
        print(f"{'='*60}")
        print("  1. 观战 AI 自对战")
        print("  2. 人机对弈")
        print("  3. 查看模型统计")
        print("  4. 多局观战（统计胜率）")
        print("  0. 退出")
        try:
            choice = input("\n请选择 [1-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return

        if choice == '1':
            try:
                seed = int(input("种子 (0-9999，回车=0): ").strip() or '0')
                temp = float(input("温度 (0.1-2.0，回车=0.5): ").strip() or '0.5')
                turns = int(input("最大回合 (回车=80): ").strip() or '80')
            except ValueError:
                seed, temp, turns = 0, 0.5, 80
            watch_game(net, seed=seed, temperature=temp, max_turns=turns)
        elif choice == '2':
            try:
                seat = int(input("你的座位 (0-3，回车=0): ").strip() or '0')
                seed = int(input("种子 (回车=0): ").strip() or '0')
            except ValueError:
                seat, seed = 0, 0
            human_play(net, human_seat=seat, seed=seed)
        elif choice == '3':
            show_stats(net)
        elif choice == '4':
            try:
                n = int(input("局数 (回车=10): ").strip() or '10')
                temp = float(input("温度 (回车=0.3): ").strip() or '0.3')
            except ValueError:
                n, temp = 10, 0.3
            wins = [0, 0, 0, 0]
            rng = random.Random(42)
            for i in range(n):
                g = Game(4, random.Random(rng.random()))
                steps = 0
                while not g.is_done() and steps < 1000:
                    a = ai_pick_action(net, g, temperature=temp, rng=rng)
                    g.step(a)
                    steps += 1
                if g.winner >= 0:
                    wins[g.winner] += 1
                print(f"  局 {i+1}: P{g.winner+1 if g.winner>=0 else '-'} 获胜")
            print(f"\n{'='*40}")
            print(f"  {n} 局统计:")
            for i in range(4):
                print(f"  P{i+1}: {wins[i]} 胜 ({wins[i]/n*100:.1f}%)")
        elif choice == '0':
            print("再见！")
            return
        else:
            print("无效选择")


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        net = load_net()
        if cmd == 'watch':
            seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
            turns = int(sys.argv[4]) if len(sys.argv) > 4 else 80
            watch_game(net, seed=seed, temperature=temp, max_turns=turns)
        elif cmd == 'human':
            seat = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            human_play(net, human_seat=seat, seed=seed)
        elif cmd == 'stats':
            show_stats(net)
        else:
            print(f"未知命令: {cmd}")
            print("用法: ai_game [watch|human|stats] [参数...]")
    else:
        net = load_net()
        interactive_menu(net)


if __name__ == '__main__':
    main()
