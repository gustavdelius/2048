import argparse
import numpy as np
import torch
from collections import Counter
import time

from env import Game2048Env
from agent import DQNAgent

def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN on 3x3 2048")
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model to evaluate')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--epsilon', type=float, default=0.0, help='Epsilon for exploration during evaluation')
    args = parser.parse_args()
    
    env = Game2048Env()
    agent = DQNAgent()
    
    print(f"Loading model '{args.model}'...")
    try:
        # Load the model directly if it's just the state dict (like best_model.pth)
        # or extract from checkpoint (like checkpoint.pth)
        checkpoint = torch.load(args.model, weights_only=False, map_location=agent.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        else:
            agent.q_network.load_state_dict(checkpoint)
        agent.q_network.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Starting evaluation for {args.episodes} episodes with epsilon {args.epsilon}...")
    print("-" * 50)
    
    rewards = []
    highest_tiles = []
    steps = []
    
    start_time = time.time()
    
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            action = agent.select_action(state, args.epsilon, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
        rewards.append(episode_reward)
        highest_tiles.append(info['highest_tile'])
        steps.append(step_count)
        
        if episode % 10 == 0 or episode == args.episodes:
            print(f"Progress: {episode}/{args.episodes} episodes completed.")
            
    end_time = time.time()
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Total episodes: {args.episodes}")
    print(f"Time taken: {end_time - start_time:.2f} seconds ({(end_time - start_time) / args.episodes * 1000:.1f} ms/ep)")
    print(f"Average Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    
    print("\nMax Tile Distribution:")
    max_tile_counts = Counter(highest_tiles)
    for tile in sorted(max_tile_counts.keys(), reverse=True):
        count = max_tile_counts[tile]
        percentage = (count / args.episodes) * 100
        print(f"  {tile:4d}: {count:4d} episodes ({percentage:5.1f}%)")
    
    print("=" * 50)

if __name__ == '__main__':
    main()
