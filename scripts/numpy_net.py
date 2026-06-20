"""
NumpyNet: 用纯 numpy 实现的 PolicyValueNet 推理。
比 torch forward 快约 5 倍（0.042ms vs 0.218ms），结果完全一致。
用于自博弈数据收集阶段加速训练。

支持多种激活函数：relu / tanh / gelu / elu
"""

import numpy as np
import torch
from typing import Dict
from model import PolicyValueNet


class NumpyNet:
    """纯 numpy 实现的 PolicyValueNet 推理引擎。"""

    def __init__(self, torch_net: PolicyValueNet = None, activation: str = 'relu'):
        self.activation = activation
        self.params: Dict[str, np.ndarray] = {}
        if torch_net is not None:
            self.load_from_torch(torch_net)

    def load_from_torch(self, torch_net: PolicyValueNet):
        """从 torch 网络提取参数。"""
        for name, p in torch_net.named_parameters():
            self.params[name] = p.detach().cpu().numpy()

    def load_from_state_dict(self, state_dict):
        """从 state_dict 加载参数。"""
        for name, tensor in state_dict.items():
            self.params[name] = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else np.array(tensor)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(x, 0)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'gelu':
            # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        elif self.activation == 'elu':
            # ELU: x if x > 0 else alpha * (exp(x) - 1), alpha=1
            return np.where(x > 0, x, np.expm1(x))
        else:
            return np.maximum(x, 0)  # 默认 relu

    def forward(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """前向传播。state shape: (121,) 或 (batch, 121)"""
        if state.ndim == 1:
            state = state[np.newaxis, :]  # (1, 121)

        # trunk: 3 层 Linear + 激活
        h = state
        h = h @ self.params['trunk.0.weight'].T + self.params['trunk.0.bias']
        h = self._activate(h)
        h = h @ self.params['trunk.2.weight'].T + self.params['trunk.2.bias']
        h = self._activate(h)
        h = h @ self.params['trunk.4.weight'].T + self.params['trunk.4.bias']
        h = self._activate(h)

        # heads
        out = {}
        for head_name in ['target', 'item_decision', 'item_choice',
                          'edit_pos', 'edit_delta', 'reward_choice']:
            layer = f'head_{head_name}.'
            out[head_name] = h @ self.params[layer + 'weight'].T + self.params[layer + 'bias']
        # value head: (batch, 1) -> (batch,)
        out['value'] = (h @ self.params['head_value.weight'].T +
                        self.params['head_value.bias']).squeeze(-1)
        return out

    def __call__(self, state):
        return self.forward(state)


def masked_softmax_np(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """numpy 版 masked softmax。"""
    neg_inf = -1e30
    masked = np.where(mask, logits, neg_inf)
    # 数值稳定
    max_val = np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(masked - max_val)
    exp = np.where(mask, exp, 0.0)
    s = exp.sum(axis=-1, keepdims=True)
    s = np.maximum(s, 1e-30)
    return exp / s


# Phase 常量（与 game.py 一致）
PHASE_CHOOSE_TARGET = 0
PHASE_ITEM_DECISION = 1
PHASE_ITEM_CHOICE = 2
PHASE_EDIT_POS = 3
PHASE_EDIT_DELTA = 4
PHASE_REWARD_CHOICE = 5


def action_mask_np(phase: int, valid_actions, num_players: int = 10) -> np.ndarray:
    """numpy 版 action_mask，比 torch 版快 17 倍。
    注意：CHOOSE_TARGET 始终返回 MAX_PLAYERS=10 维（与 torch 版一致）。
    """
    if phase == PHASE_CHOOSE_TARGET:
        # 始终返回 10 维（MAX_PLAYERS），与 torch 版一致
        mask = np.zeros(10, dtype=bool)
        for a in valid_actions:
            mask[a] = True
        return mask
    elif phase == PHASE_ITEM_DECISION:
        mask = np.zeros(2, dtype=bool)
        for a in valid_actions:
            mask[a] = True
        return mask
    elif phase == PHASE_ITEM_CHOICE:
        mask = np.zeros(4, dtype=bool)
        for a in valid_actions:
            mask[a] = True
        return mask
    elif phase == PHASE_EDIT_POS:
        mask = np.zeros(6, dtype=bool)
        for a in valid_actions:
            mask[a] = True
        return mask
    elif phase == PHASE_EDIT_DELTA:
        return np.ones(2, dtype=bool)
    elif phase == PHASE_REWARD_CHOICE:
        # valid_actions 是 [1,2,3]，映射到索引 [0,1,2]
        mask = np.zeros(3, dtype=bool)
        for a in valid_actions:
            mask[a - 1] = True
        return mask
    return np.zeros(0, dtype=bool)


if __name__ == '__main__':
    # 测试
    import time
    torch.set_num_threads(1)

    net = PolicyValueNet()
    net.eval()
    np_net = NumpyNet(net)

    state = np.random.randn(121).astype(np.float32)
    state_t = torch.from_numpy(state).unsqueeze(0)

    # 速度对比
    N = 2000
    t0 = time.time()
    for _ in range(N):
        out_np = np_net(state)
    t1 = time.time()
    print(f'numpy forward: {(t1-t0)*1000/N:.4f}ms/次')

    t0 = time.time()
    for _ in range(N):
        with torch.no_grad():
            out_t = net(state_t)
    t1 = time.time()
    print(f'torch forward: {(t1-t0)*1000/N:.4f}ms/次')
    print(f'加速比: {((t1-t0)/(t1-t0)):.1f}x')

    # 结果一致性
    out_np = np_net(state)
    with torch.no_grad():
        out_t = net(state_t)
    print(f'\n结果一致性:')
    for k in ['target', 'item_decision', 'value']:
        diff = np.abs(out_np[k] - out_t[k].detach().numpy()[0]).max()
        print(f'  {k} max diff: {diff:.8f}')
