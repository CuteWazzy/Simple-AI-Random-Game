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

# Windows 编码兜底：确保中文能正常输出
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
    # 设置控制台代码页为 UTF-8
    try:
        os.system('chcp 65001 > nul 2>&1')
    except Exception:
        pass

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


def find_writable_model_path():
    """找一个可写的模型保存路径。"""
    candidates = []
    base = os.path.dirname(os.path.abspath(sys.argv[0]))
    candidates.append(os.path.join(base, 'models', 'genetic_model.pt'))
    candidates.append(os.path.join(base, 'genetic_model.pt'))
    candidates.append(os.path.join(os.getcwd(), 'models', 'genetic_model.pt'))
    candidates.append(os.path.join(os.getcwd(), 'genetic_model.pt'))
    home = os.path.expanduser('~')
    candidates.append(os.path.join(home, 'ai_game_models', 'genetic_model.pt'))
    for path in candidates:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            test_file = path + '.tmp'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return path
        except Exception:
            continue
    return os.path.join(os.getcwd(), 'genetic_model.pt')


def run_genetic_train(num_generations=10, iters_per_gen=8, num_workers=4):
    """运行遗传算法训练。"""
    from genetic_train import (
        DEFAULT_CONFIGS, load_or_init_net, _train_worker,
        crossover_nets, mutate_net, tournament,
    )
    from copy import deepcopy
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    output_path = find_writable_model_path()
    print(f"\n[训练] 模型将保存到: {output_path}")
    print(f"[训练] {num_generations} 代 × {iters_per_gen} 轮 × 4 种群 = {num_generations*iters_per_gen*4} 局自博弈")
    print()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_gen = 0
    best_net = None
    best_win_rate = -1.0
    best_name = None

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
        except Exception as e:
            print(f"[续训] 加载失败：{e}，从头开始", flush=True)

    base_ckpt = output_path if os.path.exists(output_path) else find_model()
    populations = []
    for config in DEFAULT_CONFIGS:
        net = load_or_init_net(config, base_ckpt)
        populations.append({
            'name': config.name, 'config': config, 'net': net,
            'win_rate': 0.0, 'history': [], 'train_stats': {},
        })
        print(f"  初始化种群 [{config.name}] 激活={config.activation} T={config.temperature}", flush=True)

    for gen in range(start_gen, start_gen + num_generations):
        gen_start = time.time()
        print(f"\n{'='*60}")
        print(f"  第 {gen+1} 代 (目标 {start_gen + num_generations})")
        print(f"{'='*60}")

        print(f"\n[阶段 1] 多进程并行训练 ({iters_per_gen} 轮/种群)...")
        train_start = time.time()

        worker_args = []
        for pop in populations:
            cfg_copy = deepcopy(pop['config'])
            init_state = {k: v.clone() for k, v in pop['net'].state_dict().items()}
            base_seed_i = seed + gen * 1000 + (hash(pop['name']) & 0xFFFF)
            worker_args.append((cfg_copy, init_state, iters_per_gen, base_seed_i))

        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_train_worker, arg) for arg in worker_args]
                results = {}
                for future in as_completed(futures):
                    name, state_dict, stats, cfg_dict = future.result()
                    results[name] = (state_dict, stats, cfg_dict)
        except Exception as e:
            print(f"  [错误] 多进程训练失败: {e}")
            print(f"  [提示] 在 exe 模式下多进程可能受限，建议用源码版训练")
            return

        for pop in populations:
            if pop['name'] in results:
                state_dict, stats, cfg_dict = results[pop['name']]
                pop['net'].load_state_dict(state_dict)
                pop['config'].temperature = cfg_dict['temperature']
                pop['train_stats'] = stats
                print(f"  [{pop['name']}] loss={stats['avg_loss']:.4f} T={pop['config'].temperature:.3f}", flush=True)

        train_time = time.time() - train_start
        print(f"  训练耗时: {train_time:.1f}s")

        print(f"\n[阶段 2] 锦标赛评估...")
        win_rates = tournament(populations, num_matches_per_pair=2, seed=seed + gen * 100)
        for pop in populations:
            pop['win_rate'] = win_rates.get(pop['name'], 0.0)
            pop['history'].append(pop['win_rate'])
        print(f"  胜率:")
        for pop in sorted(populations, key=lambda x: -x['win_rate']):
            print(f"    {pop['name']}: {pop['win_rate']*100:.1f}%", flush=True)

        if gen < start_gen + num_generations - 1:
            print(f"\n[阶段 3] 遗传操作...")
            sorted_pops = sorted(populations, key=lambda x: -x['win_rate'])
            elite = sorted_pops[0]
            print(f"  精英: [{elite['name']}] 胜率={elite['win_rate']*100:.1f}%")
            top2 = sorted_pops[:2]
            new_populations = [elite]
            for i in range(1, len(populations)):
                alpha = random.random()
                child_net = crossover_nets(top2[0]['net'], top2[1]['net'], alpha=alpha)
                child_net = mutate_net(child_net, mutation_rate=0.1, mutation_std=0.03,
                                       rng=random.Random(seed + gen * 100 + i))
                new_populations.append({
                    'name': sorted_pops[i]['name'], 'config': sorted_pops[i]['config'],
                    'net': child_net, 'win_rate': 0.0, 'history': sorted_pops[i]['history'],
                    'train_stats': {},
                })
            populations = new_populations
            print(f"  产生 {len(new_populations)-1} 个后代")

        for pop in populations:
            if pop['win_rate'] > best_win_rate:
                best_win_rate = pop['win_rate']
                best_net = deepcopy(pop['net'])
                best_name = pop['name']
        if best_net is None:
            best_net = deepcopy(sorted(populations, key=lambda x: -x['win_rate'])[0]['net'])

        try:
            ckpt = {
                'model': best_net.state_dict(),
                'iteration': (gen + 1) * iters_per_gen,
                'win_rate': best_win_rate, 'best_name': best_name,
                'generation': gen + 1,
                'populations_history': {p['name']: p['history'] for p in populations},
            }
            torch.save(ckpt, output_path)
            print(f"\n  -> 已保存到 {output_path}")
        except Exception as e:
            print(f"\n  [警告] 保存失败: {e}")

        gen_time = time.time() - gen_start
        print(f"  -> 第 {gen+1} 代完成，耗时 {gen_time:.1f}s")
        print(f"  -> 历史最佳: [{best_name}] 胜率={best_win_rate*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"  训练完成！")
    print(f"  最佳种群: [{best_name}] 胜率={best_win_rate*100:.1f}%")
    print(f"  模型保存至: {output_path}")
    print(f"{'='*60}")


def run_single_train(num_iters=100, games_per_iter=24):
    """单种群训练。"""
    from train import train as train_fn
    output_path = find_writable_model_path()
    print(f"\n[训练] 模型将保存到: {output_path}")
    print(f"[训练] {num_iters} 轮 × {games_per_iter} 局/轮 = {num_iters*games_per_iter} 局自博弈")
    print()
    try:
        train_fn(num_iterations=num_iters, games_per_iter=games_per_iter,
                 num_players=4, lr=1e-3, temperature=1.0, batch_size=256, seed=42)
    except Exception as e:
        print(f"\n[错误] 训练失败: {e}")
        import traceback
        traceback.print_exc()


def train_menu():
    """训练子菜单。"""
    while True:
        print(f"\n{'='*60}")
        print(f"  训练子菜单")
        print(f"{'='*60}")
        print("  1. 遗传算法训练（推荐，4 种群并行）")
        print("  2. 单种群训练（简单）")
        print("  3. 查看当前模型统计")
        print("  0. 返回主菜单")
        try:
            choice = input("\n请选择 [0-3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return

        if choice == '1':
            try:
                gens = int(input("进化代数 (回车=10): ").strip() or '10')
                iters = int(input("每代训练轮数 (回车=8): ").strip() or '8')
                workers = int(input("并行进程数 (回车=4): ").strip() or '4')
            except ValueError:
                gens, iters, workers = 10, 8, 4
            run_genetic_train(gens, iters, workers)
        elif choice == '2':
            try:
                iters = int(input("训练轮数 (回车=100): ").strip() or '100')
                games = int(input("每轮局数 (回车=24): ").strip() or '24')
            except ValueError:
                iters, games = 100, 24
            run_single_train(iters, games)
        elif choice == '3':
            net = load_net()
            show_stats(net)
        elif choice == '0':
            return


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
        print("  5. 训练 AI（遗传算法/单种群）")
        print("  0. 退出")
        try:
            choice = input("\n请选择 [0-5]: ").strip()
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
        elif choice == '5':
            train_menu()
        elif choice == '0':
            print("再见！")
            return
        else:
            print("无效选择")


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train':
            gens = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            iters = int(sys.argv[3]) if len(sys.argv) > 3 else 8
            workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
            run_genetic_train(gens, iters, workers)
            return
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
            print("用法: ai_game [watch|human|stats|train] [参数...]")
            print("  watch [seed] [temp] [turns]  观战")
            print("  human [seat] [seed]          人机对弈")
            print("  stats                        模型统计")
            print("  train [gens] [iters] [workers]  训练")
    else:
        net = load_net()
        interactive_menu(net)


if __name__ == '__main__':
    main()
