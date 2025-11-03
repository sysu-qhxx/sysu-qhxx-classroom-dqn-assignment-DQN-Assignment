import gymnasium as gym
import torch
from alg.agent import DQNAgent

def train():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    episodes = 200
    epsilon = 0.1
    update_target_every = 10

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        if ep % update_target_every == 0:
            agent.update_target()
        print(f"Episode {ep}: reward={total_reward}")

    torch.save(agent.q_net.state_dict(), "dqn_model.pt")
    env.close()

if __name__ == "__main__":
    train()
