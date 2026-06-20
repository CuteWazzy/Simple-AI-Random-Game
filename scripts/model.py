"""
神经网络策略-价值模型。

设计：
  * 共享 MLP 主干：input(121) -> 128 -> 128
  * 多个动作头：target / item_decision / item_choice / edit_pos / edit_delta / reward_choice
  * 价值头：标量 V(s)

forward 返回一个 dict，包含所有头的 logits 与 value。
训练时根据当前 phase 选用对应的头，并按 mask 屏蔽非法动作。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from game import (
    PHASE_CHOOSE_TARGET, PHASE_ITEM_DECISION, PHASE_ITEM_CHOICE,
    PHASE_EDIT_POS, PHASE_EDIT_DELTA, PHASE_REWARD_CHOICE,
    Game,
)

MAX_PLAYERS = 10


class PolicyValueNet(nn.Module):
    def __init__(self, state_size: int = Game.STATE_SIZE,
                 hidden_size: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # 各动作头
        self.head_target = nn.Linear(hidden_size, MAX_PLAYERS)
        self.head_item_decision = nn.Linear(hidden_size, 2)        # 0=n 1=y
        self.head_item_choice = nn.Linear(hidden_size, 4)          # 0/1/2/3
        self.head_edit_pos = nn.Linear(hidden_size, 6)             # 0..5
        self.head_edit_delta = nn.Linear(hidden_size, 2)           # 0=+1 1=-1
        self.head_reward_choice = nn.Linear(hidden_size, 3)        # 1/2/3 -> 索引 0/1/2
        self.head_value = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return {
            'target': self.head_target(h),
            'item_decision': self.head_item_decision(h),
            'item_choice': self.head_item_choice(h),
            'edit_pos': self.head_edit_pos(h),
            'edit_delta': self.head_edit_delta(h),
            'reward_choice': self.head_reward_choice(h),
            'value': self.head_value(h).squeeze(-1),
        }

    # ---- 工具：根据 phase 取出该步使用的 head 输出与 mask ----
    @staticmethod
    def head_for_phase(phase: int) -> str:
        return {
            PHASE_CHOOSE_TARGET: 'target',
            PHASE_ITEM_DECISION: 'item_decision',
            PHASE_ITEM_CHOICE: 'item_choice',
            PHASE_EDIT_POS: 'edit_pos',
            PHASE_EDIT_DELTA: 'edit_delta',
            PHASE_REWARD_CHOICE: 'reward_choice',
        }[phase]

    @staticmethod
    def action_mask(phase: int, valid_actions, num_players: int = MAX_PLAYERS) -> torch.Tensor:
        """构造布尔 mask，True 表示该动作合法。"""
        if phase == PHASE_CHOOSE_TARGET:
            mask = torch.zeros(MAX_PLAYERS, dtype=torch.bool)
            for a in valid_actions:
                mask[a] = True
            return mask
        elif phase == PHASE_ITEM_DECISION:
            mask = torch.zeros(2, dtype=torch.bool)
            for a in valid_actions:
                mask[a] = True
            return mask
        elif phase == PHASE_ITEM_CHOICE:
            mask = torch.zeros(4, dtype=torch.bool)
            for a in valid_actions:
                mask[a] = True
            return mask
        elif phase == PHASE_EDIT_POS:
            mask = torch.zeros(6, dtype=torch.bool)
            for a in valid_actions:
                mask[a] = True
            return mask
        elif phase == PHASE_EDIT_DELTA:
            return torch.ones(2, dtype=torch.bool)
        elif phase == PHASE_REWARD_CHOICE:
            # valid_actions 是 [1,2,3]，需要映射到索引 [0,1,2]
            mask = torch.zeros(3, dtype=torch.bool)
            for a in valid_actions:
                mask[a - 1] = True
            return mask
        return torch.zeros(0, dtype=torch.bool)

    @staticmethod
    def action_to_head_index(phase: int, action: int) -> int:
        """把游戏 action 映射到 head 输出的索引。REWARD_CHOICE 的 1/2/3 -> 0/1/2。"""
        if phase == PHASE_REWARD_CHOICE:
            return action - 1
        return action

    @staticmethod
    def head_index_to_action(phase: int, idx: int) -> int:
        if phase == PHASE_REWARD_CHOICE:
            return idx + 1
        return idx


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """数值稳定的 masked softmax。"""
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask, neg_inf)
    return F.softmax(masked, dim=-1)


if __name__ == '__main__':
    net = PolicyValueNet()
    print(net)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"参数总数: {n_params}")
    # 测试前向
    import random
    g = Game(4, random.Random(0))
    state = torch.tensor([g.encode_state()], dtype=torch.float32)
    out = net(state)
    for k, v in out.items():
        print(f"  {k}: shape={tuple(v.shape)}")
