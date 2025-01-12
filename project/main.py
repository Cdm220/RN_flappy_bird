import gymnasium as gym
import flappy_bird_gymnasium
from agents.dqn_agent import DQNAgent
from preprocessing.state_processor import normalize_state
from utils.training_utils import save_checkpoint, load_checkpoint
from config import Config


def train():
    env = gym.make("FlappyBird-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    best_reward = float('-inf')
    best_score = 0

    for episode in range(Config.EPISODES):
        state, _ = env.reset()
        state = normalize_state(state)
        total_reward = 0
        steps = 0
        score = 0

        while True:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_state(next_state)
            agent.memory.push(state, action, reward, next_state, terminated)

            state = next_state
            if reward > 0.1:
                score += 1
                reward = 10.0
            total_reward += reward
            steps += 1

            agent.train()

            if terminated or truncated:
                break

        if total_reward > best_reward:
            best_reward = total_reward
            best_score = score
            save_checkpoint(agent, episode, best_reward)

        if episode % 250 == 0:
            print(f"Episode {episode + 1}/{Config.EPISODES}, Steps: {steps}, Score: {score}, "
                  f"Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Best Reward: {best_reward:.2f}, Best Score: {best_score}")

    env.close()


def test():
    env = gym.make("FlappyBird-v0", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Load the best checkpoint
    load_checkpoint(agent, "flappy_bird_dqn.pth")
    agent.model.eval()

    state, _ = env.reset()
    total_reward = 0
    steps = 0
    score = 0

    while True:
        state = normalize_state(state)
        action = agent.act(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)

        state = next_state
        if reward > 0.1:
            score += 1
            reward = 10.0
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"Test Results - Steps: {steps}, Total Reward: {total_reward:.2f}, Score: {score}")
    env.close()
    return score


if __name__ == "__main__":
    # train()
    best_score = 0
    avg_score = 0
    for _ in range(10):
        score = test()
        best_score = max(best_score, score)
        avg_score += score

    print(f"Best Score: {best_score}, Average Score: {avg_score / 10}")