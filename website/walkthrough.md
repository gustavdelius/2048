# Educational 3x3 2048 Reinforcement Learning

The objective was to write all necessary code for training an RL agent to play a 3x3 version of 2048, prioritizing an easy-to-understand educational design, monitoring tools, and quadratic reward scaling. Below is a breakdown of the implementation.

## 1. Virtual Environment & Dependencies

To avoid clashing with your system packages, all dependencies were installed in a local Python virtual environment (`venv`). 
* If you run any scripts manually, ensure you use `venv/bin/python`.
* Dependencies used: `torch`, `numpy`, `tensorboard`, `matplotlib`.

## 2. Environment (`env.py`)

A custom, concise imitation of an OpenAI Gym environment is implemented in [env.py](file:///home/gustav/Git/2048/env.py).
* The state is returned as the Base-2 logarithm ($\log_2$) of the tile values (e.g., $2 \rightarrow 1$, $16 \rightarrow 4$). This prevents the neural network from struggling with exponentially scaling inputs.
* The reward uses the requested **quadratic** scaling formula. Merging two tiles of value $V$ to create $2V$ gives a reward of $(2V)^2$.
* The board is modeled as a 3x3 NumPy array.

## 3. Custom DQN Agent (`agent.py`)

A Deep Q-Network is implemented from scratch using PyTorch in [agent.py](file:///home/gustav/Git/2048/agent.py).
* **Network**: Since 3x3 is very small (9 cells), the network uses a straightforward Multi-Layer Perceptron (MLP) mapping 9 inputs $\rightarrow$ 128 hidden $\rightarrow$ 128 hidden $\rightarrow$ 4 action Q-values. This is much easier to understand than a CNN.
* **Memory**: `ReplayBuffer` records up to 10,000 recent moves to train on using a `batch_size` of 64.
* **Target Network**: A delayed target network is implemented to stabilize Q-value training calculations.

## 4. Execution & Monitoring (`train.py` & TensorBoard)

The training loop brings the environment and the agent together in [train.py](file:///home/gustav/Git/2048/train.py).
* Training handles the exponential $\epsilon$-greedy exploration decay.
* It automatically saves the weights to `best_model.pth` any time the agent beats its previous top score during an episode.
* TensorBoard acts as the monitoring tool. It tracks `Reward/Episode`, `Highest_Tile/Episode`, `Loss/Average`, and `Hyperparameters/Epsilon`.

### Running the Code

**Test the Environment logic visually:**
```bash
venv/bin/python test_env.py
```

**Run the Training Script:**
```bash
venv/bin/python train.py --episodes 5000
```
> [!TIP]
> You can stop the script using `CTRL+C` at any time; the `best_model.pth` updates automatically as it trains.

**Monitor Progress:**
In a separate terminal, launch TensorBoard:
```bash
venv/bin/tensorboard --logdir=runs
```
Then visit [http://localhost:6006](http://localhost:6006) in your browser to watch the agent's progress live!
