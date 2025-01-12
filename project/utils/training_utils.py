import torch


def save_checkpoint(agent, episode, best_reward):

    torch.save({
        'episode': episode,
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'best_reward': best_reward
    }, 'flappy_bird_dqn.pth')


def load_checkpoint(agent, filename):
    checkpoint = torch.load(filename)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
