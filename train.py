import numpy as np
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from env import Game2048Env
from agent import DQNAgent

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train DQN on 3x3 2048")
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from (e.g., checkpoint.pth)')
    args = parser.parse_args()
    
    env = Game2048Env()
    agent = DQNAgent()
    
    # Hyperparameters
    num_episodes = args.episodes
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 10000
    target_update_freq = 50
    
    # Setup TensorBoard tracing to monitor logic
    writer = SummaryWriter('runs/3x3_2048_DQN')
    
    best_score = -float('inf')
    epsilon = epsilon_start
    start_episode = 1
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume)
            start_episode = checkpoint['episode'] + 1
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            agent.target_network.load_state_dict(checkpoint['target_model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epsilon = checkpoint['epsilon']
            best_score = checkpoint.get('best_score', -float('inf'))
            print(f"Resuming from episode {start_episode} with epsilon {epsilon:.4f}")
        else:
            print(f"No checkpoint found at '{args.resume}', starting from scratch.")

    print("Starting Training...")
    print(f"Logging to TensorBoard (view with: tensorboard --logdir=runs)")
    print("-" * 50)
    
    for episode in range(start_episode, num_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        loss_history = []
        invalid_count = 0
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                done = True
                break
                
            action = agent.select_action(state, epsilon, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            # Record the valid actions of the NEXT state to train the target properly
            next_valid_actions = env.get_valid_actions()
            next_valid_mask = [1 if a in next_valid_actions else 0 for a in range(agent.num_actions)]
            
            agent.memory.push(state, action, reward, next_state, done, next_valid_mask)
            
            if not info['valid_move']:
                invalid_count += 1
                if invalid_count >= 5:
                    done = True
            else:
                invalid_count = 0
                
            loss = agent.train_step()
            if loss > 0:
                loss_history.append(loss)
                
            state = next_state
            episode_reward += reward
            
        # Epsilon decay
        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) / epsilon_decay_steps
            epsilon = max(epsilon_end, epsilon) # bound
            
        # Update Target Network
        if episode % target_update_freq == 0:
            agent.update_target_network()
            
        # Track highest tile this episode
        highest_tile = info['highest_tile']
        
        # Logging
        avg_loss = np.mean(loss_history) if loss_history else 0
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Highest_Tile/Episode', highest_tile, episode)
        writer.add_scalar('Loss/Average', avg_loss, episode)
        writer.add_scalar('Hyperparameters/Epsilon', epsilon, episode)
        
        if episode % 100 == 0:
            print(f"Episode: {episode:4d} | Reward: {episode_reward:7.1f} | "
                  f"Highest Tile: {highest_tile:4d} | Epsilon: {epsilon:.2f} | Loss: {avg_loss:.4f}")
            
        if episode_reward > best_score:
            best_score = episode_reward
            torch.save(agent.q_network.state_dict(), 'best_model.pth')
            
        # Save a full checkpoint periodically
        if episode % 100 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'target_model_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': epsilon,
                'best_score': best_score
            }, 'checkpoint.pth')
            
    # Save a final checkpoint
    torch.save({
        'episode': num_episodes,
        'model_state_dict': agent.q_network.state_dict(),
        'target_model_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': epsilon,
        'best_score': best_score
    }, 'checkpoint.pth')
    
    print("-" * 50)
    print("Training finished! Best model saved to 'best_model.pth'")
    writer.close()

if __name__ == '__main__':
    main()
