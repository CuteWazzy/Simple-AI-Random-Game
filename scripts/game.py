"""
游戏引擎：忠实移植自用户提供的 C++ 代码。

核心机制：
  * 玩家初始 HP=10，另有 extra_HP（护盾），伤害先扣护盾再扣 HP
  * 每回合当前玩家选择攻击目标，然后随机抽一个 0~999999 的数
  * 根据数字计算：命中数 nhurt、可获道具 nflip/nedit/nreroll
  * 进入道具阶段：所有存活玩家轮流（从当前玩家开始）决定是否使用道具
      - 当前玩家（抽数者）可用 flip / edit / reroll
      - 其它玩家只能用 flip（用来翻转抽数者的数字，破坏命中）
  * 道具阶段结束后，根据数字位数触发奖励：
      4 位 -> 选一个道具；3 位 -> 回 2 血；2 位 -> 回 4 血；1 位 -> +10 护盾
      5 位 -> 命中数翻倍（无奖励）；6 位 -> 无特殊效果
  * 命中 >0：对目标造成 nhurt 伤害
  * 命中 == 0 且位数 >=5 且不是自瞄：当前玩家自己扣 1 血
  * 当前玩家获得 nflip+nedit+nreroll 个道具

状态机阶段：
  CHOOSE_TARGET  -> 当前玩家选目标
  ITEM_DECISION  -> 道具用户回答 y/n
  ITEM_CHOICE    -> 道具用户选具体道具（0=退出，1=flip，2=edit，3=reroll）
  EDIT_POS       -> 选 edit 的位置（0..len-1）
  EDIT_DELTA     -> 选 edit 的变化量（0=+1，1=-1）
  REWARD_CHOICE  -> 4 位时选奖励道具（1/2/3）
  GAME_OVER      -> 游戏结束
"""

import random
import math
from typing import List, Optional

# ---------- 数字处理函数（与 C++ 完全等价） ----------

HIT_PAIRS = frozenset({69, 78, 91})


def flip_num(x: int) -> int:
    """反转数字的位数。如 123 -> 321，120 -> 21。"""
    s = str(x)
    return int(s[::-1])


def edit_num(x: int, pos: int, delta: int) -> int:
    """把 x 第 pos 位（从左 0 起）加上 delta（mod 10）。"""
    s = str(x)
    if pos < 0 or pos >= len(s):
        return x
    d = (int(s[pos]) + delta) % 10
    s = s[:pos] + str(d) + s[pos + 1:]
    return int(s)


def calc_flip(x: int) -> int:
    """x 位数趋势变化次数 <=1 时返回 1，否则 0。"""
    s = str(x)
    if len(s) <= 1:
        return 1
    trend = 0  # 0=初始, 1=上升, -1=下降
    changes = 0
    for i in range(1, len(s)):
        if s[i] > s[i - 1]:
            if trend == -1:
                changes += 1
            trend = 1
        elif s[i] < s[i - 1]:
            if trend == 1:
                changes += 1
            trend = -1
    return 1 if changes <= 1 else 0


_EDIT_CALC = {
    (2,): 0,
    (3,): 0,
    (4,): 1,
    (5,): 1,
    (6,): 3,
    (2, 2): 1,
    (2, 3): 1,
    (2, 4): 3,
    (3, 3): 2,
    (2, 2, 2): 3,
}


def calc_edit(x: int) -> int:
    """根据重复数字的 pattern 返回 edit 道具数。"""
    s = str(x)
    counts = [0] * 10
    for c in s:
        counts[int(c)] += 1
    ms = tuple(sorted(c for c in counts if c >= 2))
    return _EDIT_CALC.get(ms, 0)


def calc_reroll(x: int) -> int:
    """每个数字超出 2 次的部分求和。"""
    s = str(x)
    counts = [0] * 10
    for c in s:
        counts[int(c)] += 1
    return sum(max(0, c - 2) for c in counts)


def calc_hurt(x: int) -> int:
    """统计所有 (i,j) 对组成的两位数是否在命中表 {69,78,91} 中。"""
    s = str(x)
    hits = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            pair = int(s[i]) * 10 + int(s[j])
            if pair in HIT_PAIRS:
                hits += 1
    return hits


def count_digits(x: int) -> int:
    return 1 if x == 0 else (int(math.log10(x)) + 1)


# ---------- 状态机阶段常量 ----------

PHASE_CHOOSE_TARGET = 0
PHASE_ITEM_DECISION = 1
PHASE_ITEM_CHOICE = 2
PHASE_EDIT_POS = 3
PHASE_EDIT_DELTA = 4
PHASE_REWARD_CHOICE = 5
PHASE_GAME_OVER = 6
NUM_PHASES = 7

PHASE_NAMES = [
    'CHOOSE_TARGET', 'ITEM_DECISION', 'ITEM_CHOICE',
    'EDIT_POS', 'EDIT_DELTA', 'REWARD_CHOICE', 'GAME_OVER'
]


class Game:
    """一局多玩家游戏的状态机。step(action) 推进一步。"""

    def __init__(self, num_players: int = 4, rng: Optional[random.Random] = None):
        assert 2 <= num_players <= 10
        self.num_players = num_players
        self.rng = rng or random.Random()
        self.reset()

    # -------- 初始化 --------
    def reset(self):
        n = self.num_players
        self.hp = [10] * n
        self.extra_hp = [0] * n
        self.flip = [0] * n
        self.edit = [0] * n
        self.reroll = [0] * n
        self.current_player = 0
        self.target = -1
        self.ran = 0
        self.nhurt = 0
        self.nflip = 0
        self.nedit = 0
        self.nreroll = 0
        self.digits = 0
        self.phase = PHASE_CHOOSE_TARGET
        self.item_user = 0
        self.item_user_queue: List[int] = []
        self.someone_used_in_pass = False
        self.item_use_count = 0
        self.edit_pos_buffer = 0
        self.winner = -1
        self.turn_count = 0
        self.last_damage = 0  # 用于奖励 shaping
        self.last_damage_target = -1
        self.last_attacker = -1  # 最近一次造成伤害的攻击者（current_player）
        # ---- 道具使用追踪（用于奖励 shaping）----
        # 本回合内每位玩家使用道具的累计次数（在 _next_turn 时清零）
        self.item_used_this_turn = [0] * 10  # index by player id
        # 本回合内"由道具使用导致的命中变化"（nhurt 变化量），按道具用户累计
        # 当某玩家用 flip/edit/reroll 改变数字导致 nhurt 变化时，记到该玩家名下
        self.item_hit_delta = [0] * 10  # 道具导致的命中数增加
        # 本回合内"由道具使用避免的自扣血"，按道具用户累计
        # 当某玩家用 flip 把"5位+未命中"改成"非5位+未命中"或"命中"时，记奖励
        self.item_dmg_avoided = [0] * 10
        # ★ 新增：本回合内"无效道具使用"次数（用了道具但没产生任何正面效果）
        # 用于 shaping 扣分，避免 AI 浪费道具
        self.item_useless_count = [0] * 10
        # ★ 新增（修复盲区）：本回合内"防御者让自瞄者反伤"的命中数
        # 当 current_player 自瞄时，防御者用 flip 把数字反转成命中数字，
        # 让自瞄者自伤。这种"反向防御"应当给防御者奖励。
        # 字段名：item_self_aim_hurt[defender] = 累计让自瞄者自伤的命中数
        self.item_self_aim_hurt = [0] * 10
        return self

    # -------- 工具查询 --------
    def is_alive(self, i: int) -> bool:
        return self.hp[i] + self.extra_hp[i] > 0

    def alive_count(self) -> int:
        return sum(1 for i in range(self.num_players) if self.is_alive(i))

    def total_hp(self, i: int) -> int:
        return self.hp[i] + self.extra_hp[i]

    def _self_damage_risk(self, player: int) -> int:
        """返回 player 当前是否面临"未命中且位数>=5且非自瞄"的自扣血风险。
        返回 1 表示有风险（即将自扣 1 血），0 表示无风险。
        判断依据：player 是不是当前攻击者？如果是，且 target != player，
        且 nhurt == 0 且 digits >= 5，则有风险。
        """
        if player != self.current_player:
            return 0
        if self.target == self.current_player:
            return 0  # 自瞄不会自扣
        if self.nhurt > 0:
            return 0  # 已命中不扣
        if self.digits < 5:
            return 0
        return 1

    def _record_item_use(self, j: int, old_nhurt: int,
                         old_risk: int, new_risk: int):
        """记录一次道具使用，更新追踪指标。

        判断"有效使用"的标准（任一满足即有效）：
          - 攻击者（j == current_player）：
              * nhurt 增加（精准 edit/reroll 创造命中）
              * 或自扣血风险减少（消除了 5 位+未中+非自瞄的风险）
              * 或位数变成 1/2/3/4（获得位数奖励）
              * 或位数变成 5 且 nhurt > 0（命中翻倍收益）
          - 防御者（j != current_player）：
              * nhurt 减少（破坏攻击者命中，保护目标）
              * ★ 修复盲区：当 current_player 自瞄时，nhurt 增加
                让自瞄者反伤——这是"反向防御"，给防御者奖励
        满足以上条件则计为有效；否则计为无效使用，扣分。
        """
        self.item_use_count += 1
        self.someone_used_in_pass = True
        self.item_used_this_turn[j] += 1

        # 判断是否产生正面效果
        useful = False

        # ---- 攻击者逻辑 ----
        if j == self.current_player:
            delta = self.nhurt - old_nhurt
            if delta > 0:
                self.item_hit_delta[j] += delta
                useful = True
            # 风险减少
            risk_reduction = old_risk - new_risk
            if risk_reduction > 0:
                self.item_dmg_avoided[j] += risk_reduction
                useful = True
            # 位数变成奖励位数
            if self.digits in (1, 2, 3, 4):
                useful = True
            # 5 位且有命中（命中翻倍收益）
            if self.digits == 5 and self.nhurt > 0:
                useful = True

        # ---- 防御者逻辑 ----
        if j != self.current_player:
            # 情况 1：防御者破坏攻击者命中（保护目标）
            delta = old_nhurt - self.nhurt
            if delta > 0:
                self.item_dmg_avoided[j] += delta
                useful = True
            # ★ 情况 2（修复盲区）：如果攻击者在自瞄，
            # 防御者用 flip 让数字变成命中——攻击者会自伤
            # 这种"反向防御"应当被奖励
            if self.target == self.current_player and self.nhurt > old_nhurt:
                # 自瞄者命中数增加 = 自瞄者自伤 = 防御者贡献
                hurt_delta = self.nhurt - old_nhurt
                if hurt_delta > 0:
                    self.item_self_aim_hurt[j] += hurt_delta
                    useful = True

        # ---- 无效使用判定 ----
        if not useful:
            self.item_useless_count[j] += 1

    # -------- 内部推进 --------
    def _roll(self):
        self.ran = self.rng.randint(0, 999999)
        self._recompute_stats()
        # 道具阶段：所有存活玩家从当前玩家开始轮询
        alive = [j for k in range(self.num_players)
                 for j in [(self.current_player + k) % self.num_players]
                 if self.is_alive(j)]
        self.item_user = alive[0]
        self.item_user_queue = alive[1:]
        self.someone_used_in_pass = False
        self.item_use_count = 0
        self.phase = PHASE_ITEM_DECISION

    def _recompute_stats(self):
        self.nhurt = calc_hurt(self.ran)
        self.nflip = calc_flip(self.ran)
        self.nedit = calc_edit(self.ran)
        self.nreroll = calc_reroll(self.ran)
        self.digits = count_digits(self.ran)
        if self.digits == 5:
            self.nhurt *= 2

    def _advance_item_user(self):
        if self.item_user_queue:
            self.item_user = self.item_user_queue.pop(0)
            self.phase = PHASE_ITEM_DECISION
        else:
            # 本轮结束
            if self.someone_used_in_pass and self.item_use_count < 50:
                alive = [j for k in range(self.num_players)
                         for j in [(self.current_player + k) % self.num_players]
                         if self.is_alive(j)]
                self.item_user = alive[0]
                self.item_user_queue = alive[1:]
                self.someone_used_in_pass = False
                self.phase = PHASE_ITEM_DECISION
            else:
                self._go_to_reward_or_apply()

    def _go_to_reward_or_apply(self):
        if self.digits == 4:
            self.phase = PHASE_REWARD_CHOICE
        else:
            self._apply_damage_and_items()
            self._next_turn()

    def _apply_damage_and_items(self):
        self.last_damage = 0
        self.last_damage_target = -1
        self.last_attacker = self.current_player
        # 奖励
        if self.digits == 3:
            heal = min(2, 10 - self.hp[self.current_player])
            self.hp[self.current_player] += heal
        elif self.digits == 2:
            heal = min(4, 10 - self.hp[self.current_player])
            self.hp[self.current_player] += heal
        elif self.digits == 1:
            self.extra_hp[self.current_player] += 10
        # 伤害
        if self.nhurt > 0:
            actual = self._deal_damage(self.target, self.nhurt)
            self.last_damage = actual
            self.last_damage_target = self.target
        elif self.digits >= 5 and self.target != self.current_player:
            actual = self._deal_damage(self.current_player, 1)
            self.last_damage = actual
            self.last_damage_target = self.current_player
        # 道具
        self.flip[self.current_player] += self.nflip
        self.edit[self.current_player] += self.nedit
        self.reroll[self.current_player] += self.nreroll

    def _deal_damage(self, target: int, dmg: int) -> int:
        actual = 0
        if self.extra_hp[target] >= dmg:
            self.extra_hp[target] -= dmg
            actual = dmg
        else:
            actual = self.extra_hp[target]
            dmg -= self.extra_hp[target]
            self.extra_hp[target] = 0
            base_dmg = min(dmg, self.hp[target])
            self.hp[target] -= base_dmg
            actual += base_dmg
        return actual

    def _next_turn(self):
        self.turn_count += 1
        # 重置 last_damage，避免渲染时显示上一回合的旧伤害
        self.last_damage = 0
        self.last_damage_target = -1
        self.last_attacker = -1
        # 清零本回合道具使用追踪
        for i in range(len(self.item_used_this_turn)):
            self.item_used_this_turn[i] = 0
            self.item_hit_delta[i] = 0
            self.item_dmg_avoided[i] = 0
            self.item_useless_count[i] = 0
            self.item_self_aim_hurt[i] = 0
        alive = [i for i in range(self.num_players) if self.is_alive(i)]
        if len(alive) <= 1:
            self.phase = PHASE_GAME_OVER
            self.winner = alive[0] if alive else -1
            return
        next_p = (self.current_player + 1) % self.num_players
        while not self.is_alive(next_p):
            next_p = (next_p + 1) % self.num_players
        self.current_player = next_p
        self.target = -1
        self.ran = 0
        self.phase = PHASE_CHOOSE_TARGET

    # -------- 对外接口 --------
    def get_valid_actions(self) -> List[int]:
        """返回当前阶段合法动作列表。"""
        if self.phase == PHASE_CHOOSE_TARGET:
            # 允许自瞄（i == current_player 也合法）。
            # 自瞄时：未命中不会自扣 1 血（因为 target == current_player）
            # 这是合法的防御策略，AI 应该学会在残血时使用
            return [i for i in range(self.num_players)
                    if self.is_alive(i)]
        elif self.phase == PHASE_ITEM_DECISION:
            return [0, 1]  # 0=n 1=y
        elif self.phase == PHASE_ITEM_CHOICE:
            j = self.item_user
            valid = [0]
            if self.flip[j] > 0:
                valid.append(1)
            if self.edit[j] > 0 and j == self.current_player:
                valid.append(2)
            if self.reroll[j] > 0 and j == self.current_player:
                valid.append(3)
            return valid
        elif self.phase == PHASE_EDIT_POS:
            return list(range(len(str(self.ran))))
        elif self.phase == PHASE_EDIT_DELTA:
            return [0, 1]
        elif self.phase == PHASE_REWARD_CHOICE:
            return [1, 2, 3]
        return []

    def get_decision_maker(self) -> int:
        if self.phase in (PHASE_CHOOSE_TARGET, PHASE_REWARD_CHOICE):
            return self.current_player
        if self.phase in (PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE,
                          PHASE_EDIT_POS, PHASE_EDIT_DELTA):
            return self.item_user
        return -1

    def step(self, action: int):
        if self.phase == PHASE_CHOOSE_TARGET:
            self.target = action
            self._roll()
        elif self.phase == PHASE_ITEM_DECISION:
            j = self.item_user
            if action == 1:
                can_use = (self.flip[j] > 0) or (
                    j == self.current_player and
                    (self.edit[j] > 0 or self.reroll[j] > 0))
                if can_use:
                    self.phase = PHASE_ITEM_CHOICE
                else:
                    self._advance_item_user()
            else:
                self._advance_item_user()
        elif self.phase == PHASE_ITEM_CHOICE:
            j = self.item_user
            if action == 0:
                self._advance_item_user()
            elif action == 1 and self.flip[j] > 0:
                # 使用 flip：记录使用前后 nhurt 和"自扣血风险"
                old_nhurt = self.nhurt
                old_self_dmg_risk = self._self_damage_risk(j)
                self.flip[j] -= 1
                self.ran = flip_num(self.ran)
                self._recompute_stats()
                new_self_dmg_risk = self._self_damage_risk(j)
                self._record_item_use(j, old_nhurt, old_self_dmg_risk, new_self_dmg_risk)
            elif action == 2 and self.edit[j] > 0 and j == self.current_player:
                self.phase = PHASE_EDIT_POS
            elif action == 3 and self.reroll[j] > 0 and j == self.current_player:
                old_nhurt = self.nhurt
                old_self_dmg_risk = self._self_damage_risk(j)
                self.reroll[j] -= 1
                self.ran = self.rng.randint(0, 999999)
                self._recompute_stats()
                new_self_dmg_risk = self._self_damage_risk(j)
                self._record_item_use(j, old_nhurt, old_self_dmg_risk, new_self_dmg_risk)
            # invalid -> stay
        elif self.phase == PHASE_EDIT_POS:
            s = str(self.ran)
            if 0 <= action < len(s):
                self.edit_pos_buffer = action
                self.phase = PHASE_EDIT_DELTA
        elif self.phase == PHASE_EDIT_DELTA:
            j = self.item_user
            delta = 1 if action == 0 else -1
            old_nhurt = self.nhurt
            old_self_dmg_risk = self._self_damage_risk(j)
            self.edit[j] -= 1
            self.ran = edit_num(self.ran, self.edit_pos_buffer, delta)
            self._recompute_stats()
            new_self_dmg_risk = self._self_damage_risk(j)
            self._record_item_use(j, old_nhurt, old_self_dmg_risk, new_self_dmg_risk)
            self.phase = PHASE_ITEM_CHOICE
        elif self.phase == PHASE_REWARD_CHOICE:
            i = self.current_player
            if action == 1:
                self.flip[i] += 1
            elif action == 2:
                self.edit[i] += 1
            elif action == 3:
                self.reroll[i] += 1
            self._apply_damage_and_items()
            self._next_turn()

    # -------- 状态编码 --------
    def encode_state(self, max_players: int = 10) -> List[float]:
        """编码为神经网络输入特征。长度 = max_players*9 + 全局特征。"""
        feats: List[float] = []
        for i in range(max_players):
            if i < self.num_players:
                feats.extend([
                    self.hp[i] / 10.0,
                    min(self.extra_hp[i], 10) / 10.0,
                    min(self.flip[i], 5) / 5.0,
                    min(self.edit[i], 5) / 5.0,
                    min(self.reroll[i], 5) / 5.0,
                    1.0 if self.is_alive(i) else 0.0,
                    1.0 if i == self.current_player else 0.0,
                    1.0 if i == self.target else 0.0,
                    1.0 if i == self.item_user else 0.0,
                ])
            else:
                feats.extend([0.0] * 9)
        # 全局特征
        dm = self.get_decision_maker()
        feats.append(self.alive_count() / 10.0)
        # phase one-hot
        for p in range(NUM_PHASES):
            feats.append(1.0 if p == self.phase else 0.0)
        feats.append(self.ran / 999999.0)
        # digits one-hot (1..6)
        for d in range(1, 7):
            feats.append(1.0 if d == self.digits else 0.0)
        feats.append(min(self.nhurt, 20) / 20.0)
        feats.append(min(self.nflip, 3) / 3.0)
        feats.append(min(self.nedit, 3) / 3.0)
        feats.append(min(self.nreroll, 3) / 3.0)
        feats.append(min(self.item_use_count, 50) / 50.0)
        feats.append(self.turn_count / 100.0)
        # decision maker 标识 one-hot（用于让网络知道"我"是谁）
        for i in range(max_players):
            feats.append(1.0 if i == dm else 0.0)
        return feats

    STATE_SIZE = 10 * 9 + 1 + 7 + 1 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 10  # = 120

    def is_done(self) -> bool:
        return self.phase == PHASE_GAME_OVER


# ---------- 快速自测 ----------
if __name__ == '__main__':
    g = Game(4, random.Random(42))
    steps = 0
    while not g.is_done() and steps < 500:
        valid = g.get_valid_actions()
        a = random.choice(valid)
        g.step(a)
        steps += 1
    print(f"游戏结束，{steps} 步，赢家：玩家 {g.winner + 1 if g.winner >= 0 else '无'}")
    for i in range(g.num_players):
        print(f"  P{i+1}: HP={g.hp[i]} extra={g.extra_hp[i]} "
              f"flip={g.flip[i]} edit={g.edit[i]} reroll={g.reroll[i]} "
              f"alive={g.is_alive(i)}")
    print(f"特征维度: {len(g.encode_state())} (期望 {Game.STATE_SIZE})")
