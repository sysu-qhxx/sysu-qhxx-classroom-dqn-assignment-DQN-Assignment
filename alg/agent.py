import torch
import torch.optim as optim
import torch.nn.functional as F
from alg.network import QNetwork
from alg.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.action_dim = action_dim

    def select_action(self, state, epsilon=0.1):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
            return q_values.argmax().item()

    # -----------------------------
    # TODO: 学生需要完成这个函数
    # -----------------------------
    def update(self):
        """Perform one DQN update step."""
        if len(self.buffer) < self.batch_size:
            return None

        # 从经验池采样
        transitions = self.buffer.sample(self.batch_size)
        batch = self._batchify(transitions)

        # TODO: 完成 DQN 的 Q-learning 更新逻辑
        # 目标:
        #   target = r + gamma * max(Q_target(next_state))
        #   loss = MSE(Q(state, action), target)
        # 提示: 用 torch.gather() 按动作索引 Q 值

        pass  # ← 学生在此处实现更新逻辑

    def _batchify(self, transitions):
        states = torch.tensor([t.state for t in transitions], dtype=torch.float32).to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor([t.next_state for t in transitions], dtype=torch.float32).to(self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
