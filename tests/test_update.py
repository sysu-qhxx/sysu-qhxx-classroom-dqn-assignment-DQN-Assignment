import torch
from alg.agent import DQNAgent
import numpy as np

def test_update_logic_exists():
    """检查学生是否实现了 update 函数"""
    import inspect
    src = inspect.getsource(DQNAgent.update)
    assert "pass" not in src, "请实现 DQNAgent.update() 函数逻辑"

def test_buffer_sampling():
    agent = DQNAgent(4, 2)
    for _ in range(70):
        agent.buffer.push(np.zeros(4), 0, 1, np.zeros(4), False)
    agent.update()  # 不报错即可
