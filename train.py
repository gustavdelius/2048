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
    parser.add_argument('--epsilon-start', type=float, default=None, help='Starting epsilon (overrides checkpoint default)')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='Final minimum epsilon value')
    parser.add_argument('--epsilon-decay', type=int, default=10000, help='Number of episodes to decay epsilon over')
    parser.add_argument('--exploration-tile', type=int, default=2, help='Tile value to start using epsilon exploration')
    args = parser.parse_args()
    
    env = Game2048Env()
    agent = DQNAgent()
    
    # Hyperparameters
    num_episodes = args.episodes
    epsilon_start_default = 1.0 # Only used if starting fresh and no override given
    epsilon_end = args.epsilon_end
    epsilon_decay_steps = args.epsilon_decay
    target_update_freq = 50
    exploration_tile = args.exploration_tile
    
    # Setup TensorBoard tracing to monitor logic
    writer = SummaryWriter('runs/3x3_2048_DQN')
    
    # Track the best *average* score over 100 episodes instead of single episode record
    best_score = -float('inf')
    epsilon = args.epsilon_start if args.epsilon_start is not None else epsilon_start_default
    start_episode = 1
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, weights_only=False)
            start_episode = checkpoint['episode'] + 1
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            agent.target_network.load_state_dict(checkpoint['target_model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
            if args.epsilon_start is not None:
                epsilon = args.epsilon_start
            else:
                epsilon = checkpoint.get('epsilon', 0.05)
            best_score = checkpoint.get('best_score', -float('inf'))
            print(f"Resuming from episode {start_episode} with epsilon {epsilon:.4f}")
        else:
            print(f"No checkpoint found at '{args.resume}', starting from scratch.")

    print("Starting Training...")
    print(f"Logging to TensorBoard (view with: tensorboard --logdir=runs)")
    print("-" * 50)
    
    recent_rewards = []
    recent_highest_tiles = []
    recent_avg_losses = []
    
    for episode in range(start_episode, num_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        loss_history = []
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                done = True
                break
                
            # Allow more exploration late in the game (e.g., after the exploration_tile is created)
            # Early in the game, reduce random moves to increase chances of reaching later stages
            current_epsilon = epsilon if np.max(env.board) >= exploration_tile else 0.001
            
            action = agent.select_action(state, current_epsilon, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            # Record the valid actions of the NEXT state to train the target properly
            next_valid_actions = env.get_valid_actions()
            next_valid_mask = [1 if a in next_valid_actions else 0 for a in range(agent.num_actions)]
            
            agent.memory.push(state, action, reward, next_state, done, next_valid_mask)
                
            loss = agent.train_step()
            if loss > 0:
                loss_history.append(loss)
                
            state = next_state
            episode_reward += reward
            
        # Epsilon decay
        if epsilon > epsilon_end:
            # We use the initial epsilon of the run to calculate the step decay rate
            start_eps_for_decay = args.epsilon_start if args.epsilon_start is not None else epsilon_start_default
            epsilon -= (start_eps_for_decay - epsilon_end) / epsilon_decay_steps
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
        
        recent_rewards.append(episode_reward)
        recent_highest_tiles.append(highest_tile)
        recent_avg_losses.append(avg_loss)
        
        if episode % 100 == 0:
            avg_100_reward = np.mean(recent_rewards)
            max_100_tile = int(np.max(recent_highest_tiles))
            max_tile_count = recent_highest_tiles.count(max_100_tile)
            avg_100_loss = np.mean(recent_avg_losses)
            
            print(f"Episode: {episode:4d} | Avg Reward: {avg_100_reward:7.1f} | "
                  f"Max Tile: {max_100_tile:4d} ({max_tile_count:2d}) | Epsilon: {epsilon:.2f} | Avg Loss: {avg_100_loss:.4f}")
            
            # Save best model based on 100-episode average performance, not single luck
            # Also save if best_model.pth doesn't exist yet
            if avg_100_reward > best_score or not os.path.exists('best_model.pth'):
                if avg_100_reward > best_score:
                    print(f"*** New best 100-episode average reward: {avg_100_reward:.1f}! Model saved. ***")
                else:
                    print(f"*** 'best_model.pth' missing, saving current model. ***")
                best_score = avg_100_reward
                torch.save(agent.q_network.state_dict(), 'best_model.pth')
                
            recent_rewards = []
            recent_highest_tiles = []
            recent_avg_losses = []
            
            # Save a full checkpoint periodically
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
