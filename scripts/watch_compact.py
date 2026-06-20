"""
简洁观战模式：只显示每回合的关键事件（抽数、道具使用、伤害、奖励）。
用法：python3 watch_compact.py [num_games] [seed]
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(1)

from game import (
    Game, PHASE_CHOOSE_TARGET, PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE,
    PHASE_EDIT_POS, PHASE_EDIT_DELTA, PHASE_REWARD_CHOICE, PHASE_GAME_OVER,
)
from model import PolicyValueNet, masked_softmax
from play import load_net, ai_pick_action


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


def watch_compact(net, num_games=1, seed=0, temperature=0.3, max_turns=30):
    rng = random.Random(seed)
    wins = [0] * 4
    for gi in range(num_games):
        g = Game(4, random.Random(rng.random()))
        print(f"\n{'='*60}")
        print(f"  第 {gi+1} 局")
        print(f"{'='*60}")
        print("初始:", render_hp(g))

        turn = 0
        prev_turn = -1
        step_in_turn = 0
        last_ran = None
        last_nhurt = None
        item_events = []

        while not g.is_done() and turn < max_turns:
            if g.turn_count != prev_turn:
                if prev_turn >= 0 and (last_ran is not None or item_events):
                    # 打印上一回合总结（即使没造成伤害也要打印道具事件）
                    if last_ran is not None:
                        dmg_str = f"命中{last_nhurt}" if last_nhurt > 0 else "未命中"
                        print(f"  └ 抽数 {last_ran} (位数={count_digits(last_ran)}) {dmg_str}")
                    if item_events:
                        print(f"  └ 道具: {'; '.join(item_events)}")
                prev_turn = g.turn_count
                turn += 1
                step_in_turn = 0
                last_ran = None
                last_nhurt = None
                item_events = []
                print(f"\n[回合 {g.turn_count + 1}] {render_hp(g)}")

            phase = g.phase
            dm = g.get_decision_maker()
            valid = g.get_valid_actions()

            if phase == PHASE_CHOOSE_TARGET:
                a = ai_pick_action(net, g, temperature=temperature, rng=rng)
                print(f"  P{dm+1} → 攻击 P{a+1}")
                g.step(a)
                last_ran = g.ran
                last_nhurt = g.nhurt

            elif phase == PHASE_ITEM_DECISION:
                a = ai_pick_action(net, g, temperature=temperature, rng=rng)
                g.step(a)
                # 不打印 y/n，只在真正使用道具时记录

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
                    # edit: 先选 pos 再选 delta
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
                # 接下来 step 会触发伤害结算
                g.step(a)
                # 打印伤害
                if g.last_damage > 0:
                    aname = f"P{g.last_attacker+1}" if g.last_attacker >= 0 else '?'
                    tname = f"P{g.last_damage_target+1}" if g.last_damage_target >= 0 else '?'
                    print(f"  └ 抽数 {last_ran} (位数={count_digits(last_ran)}) 命中{last_nhurt}")
                    if item_events:
                        print(f"  └ 道具: {'; '.join(item_events)}")
                    print(f"  ★ {aname} 对 {tname} 造成 {g.last_damage} 点伤害")
                    last_ran = None
                    item_events = []

            else:
                # 不应该到这里
                g.step(ai_pick_action(net, g, temperature=temperature, rng=rng))

        # 收尾：如果还有未打印的事件
        if last_ran is not None or item_events:
            if last_ran is not None:
                dmg_str = f"命中{last_nhurt}" if last_nhurt > 0 else "未命中"
                print(f"  └ 抽数 {last_ran} (位数={count_digits(last_ran)}) {dmg_str}")
            if item_events:
                print(f"  └ 道具: {'; '.join(item_events)}")

        # 如果游戏因 max_turns 截断，可能有未结算的回合事件
        if not g.is_done() and (last_ran is not None or item_events):
            pass  # 上面已经打印了

        print("\n最终:", render_hp(g))
        if g.is_done():
            wins[g.winner] += 1
            print(f"★ 赢家: P{g.winner+1}")
        else:
            print(f"(已展示前 {max_turns} 回合)")

    if num_games > 1:
        print(f"\n{'='*60}")
        print(f"  总计 {num_games} 局")
        print(f"{'='*60}")
        for i in range(4):
            bar = '█' * int(wins[i] / num_games * 40)
            print(f"  P{i+1}: {wins[i]:3d} 胜 ({wins[i]/num_games*100:5.1f}%) {bar}")


def count_digits(x):
    return 1 if x == 0 else (len(str(x)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    max_turns = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    net = load_net()
    watch_compact(net, num_games=n, seed=seed, temperature=temp, max_turns=max_turns)


if __name__ == '__main__':
    main()
