import gymnasium as gym
import torch
from alg.network import QNetwork

def evaluate(model_path="dqn_model.pt", episodes=5):
    env = gym.make("CartPole-v1", render_mode=None)
    model = QNetwork(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    total_reward = 0
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = q_values.argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
        total_reward += ep_reward
        print(f"Episode {ep} reward: {ep_reward}")
    print(f"Average reward: {total_reward / episodes}")

if __name__ == "__main__":
    evaluate()
