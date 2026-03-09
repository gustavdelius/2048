# 2048 DQN Agent

This repository contains a Deep Q-Network (DQN) agent trained to play a 3x3 version of the popular game [2048](https://play2048.co/). The project includes the custom game environment, the PyTorch-based RL agent, a training loop, and a Flask web interface to visualize the agent's gameplay.

## Project Structure

- **`env.py`**: Custom 3x3 2048 environment implementation. Handles game logic (sliding, merging, invalid moves, and score tracking).
- **`agent.py`**: PyTorch implementation of the Deep Q-Network (DQN). Uses an embedding layer to represent tile states and two exact copies of the network (policy and target) for stable training.
- **`train.py`**: The main training loop. Implements epsilon-greedy exploration, experience replay, and periodic target network updates. Includes TensorBoard logging.
- **`evaluate.py`**: Script to evaluate model performance across multiple games by tracking average reward, average steps, and recording tile distributions.
- **`app.py`**: A Flask web application that serves a web-based 2048 UI. You can manually play the game or ask the pre-trained agent (`best_model.pth`) to play for you.
- **`best_model.pth`** Saved PyTorch model weights for the trained agent.
- **`dqn_math.md` & `dqn_explanation.md`**: Detailed markdown documents explaining the theoretical background and mathematical derivations behind the DQN agent for this environment.
- **Visualization & Debugging Scripts**: Scripts such as `visualise_embedding.py`, `analyze_embeddings.py`, and `debug_*.py` are used for debugging the reward function, calculating training loss, and visualizing learned tile embeddings.

## Installation

1. Prepare your Python environment (e.g., venv):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install the necessary dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install flask numpy tensorboard
   ```

## Usage

### 1. Training the Agent

To start training the DQN agent from scratch, run `train.py`. The script will output training progress to the console and log metrics to TensorBoard.

```bash
python train.py --episodes 50000
```

Available arguments:
- `--episodes`: Total number of episodes to train (default: `5000`).
- `--resume`: Path to a checkpoint file to resume training (e.g., `--resume checkpoint.pth`).
- `--epsilon-start`: Starting exploration rate.
- `--epsilon-end`: Final minimum exploration rate (default: `0.05`).
- `--epsilon-decay`: Number of episodes to decay epsilon over.
- `--exploration-tile`: Tile value to start using epsilon exploration (default: `2`).

To monitor training using TensorBoard:
```bash
tensorboard --logdir=runs
```

### 2. Evaluating the Agent

Once you have a model trained (e.g., `best_model.pth`), you can evaluate its performance over a set number of games using `evaluate.py`. 

```bash
python evaluate.py --episodes 100
```

Available arguments:
- `--model`: Path to a specific model to evaluate (default: `best_model.pth`)
- `--episodes`: Number of episodes to run (default: `100`)
- `--epsilon`: Exploration rate during evaluation (default: `0.01`)

### 3. Running the Web Interface

Once you have a `best_model.pth` trained, you can run the Flask app to see the agent in action.

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000`. Navigate to this URL in your browser to:
- Play the 3x3 2048 game manually.
- Click the AI move button to have the pre-trained DQN agent pick the best move.

### 4. Understanding the Math

For a deep dive into the reinforcement learning theory and mathematical derivations behind this specific implementation, refer to:

- `dqn_math.md`
- `dqn_explanation.md`
- `embedding_analysis.md`
