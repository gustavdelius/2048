# The Mathematics of Deep Q-Networks (DQN) in 2048

This guide explores the formal mathematical foundation of the Deep Q-Network (DQN) algorithm, translating the theoretical concepts directly into the concrete code used to train our 2048 Agent.

## 1. The Markov Decision Process (MDP)

Reinforcement Learning formally treats the environment as a Markov Decision Process (MDP). An MDP is defined by a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

* **$\mathcal{S}$ (State Space):** The set of all possible configurations of the 2048 board.
* **$\mathcal{A}$ (Action Space):** The set of valid moves $\mathcal{A} = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}$.
* **$\mathcal{P}$ (Transition Probability):** $P(s' \mid s, a)$, the probability of transitioning to state $s'$ after taking action $a$ in state $s$. (In 2048, after a shift, a new tile '2' or '4' spawns in a random empty cell).
* **$\mathcal{R}$ (Reward Function):** $R(s, a, s')$, the immediate scalar reward received after transitioning from $s$ to $s'$ under action $a$. (In our code, this is the sum of $(2 \times \text{merged\_tile})^2 / 100$). We scale the quadratic reward down by a constant factor to prevent Q-values from diverging to infinity during training.
* **$\gamma \in [0, 1)$ (Discount Factor):** A tuning parameter that dictates how much the agent cares about immediate rewards versus future rewards. 

In our `train.py` loop, the interaction follows the MDP sequence at each discrete time step $t$:

1. The agent observes state $S_t$.
2. The agent takes action $A_t \in \mathcal{A}$.
3. The environment transitions to $S_{t+1}$ and emits reward $R_{t+1}$.

```python
# From train.py
next_state, reward, done = env.step(action)
```

## 2. The Reward Function in 2048

The reward function $\mathcal{R}(s, a, s')$ dictates the agent's immediate goals. In our implementation, the reward heavily incentivizes combining large tiles, rather than just surviving by making random valid moves.

When the agent successfully merges two tiles of value $V$ to create a new tile of value $2V$, the mathematical reward is quadratic. We then scale it down by a factor of 100 to keep the Q-values mathematically stable during training:
$$R = \frac{(2V)^2}{100}$$

If a move results in no merged tiles but is still valid (tiles shifted), the reward is $0$ (plus whatever might be gained from other merges on the board). If an agent attempts an invalid move (such as swiping into a wall where no tiles move), it receives a penalty of $-1$.

This is explicitly calculated in `env.py`:

```python
# From env.py (inside slide_and_merge)
new_val = non_zero[j] * 2
merged_row.append(new_val)
# Reward higher-value tiles *quadratically* more, scaled down by 100
reward += (new_val**2) / 100.0

# ... (inside step)
# Give a small negative reward for invalid moves
if not changed:
    reward = -1 
```

## 3. Expected Return and the Q-Function

The agent's objective is to choose actions that maximize the **Expected Return** $G_t$, which is the commutative sum of discounted future rewards:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

To achieve this, we define the **Action-Value Function** (Q-function). The Q-value $Q^\pi(s, a)$ is the expected return if the agent starts in state $s$, takes action $a$, and forever follows policy $\pi$:
$$Q^\pi(s, a) = \mathbb{E}_\pi [ G_t \mid S_t = s, A_t = a ]$$

The **Optimal Action-Value Function**, $Q^*(s, a)$, represents the maximum expected return achievable by any policy. If the agent knows $Q^*(s, a)$, the optimal policy $\pi^*$ is simply to greedily choose the action with the highest Q-value:
$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s, a)$$

## 4. The Bellman Optimality Equation

The foundation of Q-learning is the **Bellman Optimality Equation**, which recursively defines $Q^*(s, a)$ in terms of the immediate reward and the optimal Q-value of the next state:
$$Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{P}} \left[ R(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]$$

This states that the optimal expected return of an action is the expected immediate reward plus the discounted optimal return from the best possible action in the subsequent state.

In our code, $\gamma$ is explicitly defined in `agent.py`:
```python
# From agent.py
self.gamma = 0.9
```

## 5. Deep Q-Networks (Approximating Q)

Because the state space $\mathcal{S}$ of 2048 is too large to compute $Q^*(s,a)$ exactly for every combination, DQN uses a neural network to *approximate* the Q-function:
$$Q(s, a; \theta) \approx Q^*(s, a)$$
where $\theta$ represents the weights and biases of the neural network. 

In `agent.py`, this network $Q(s, a; \theta)$ is an MLP (Multi-Layer Perceptron):
```python
# From agent.py: The Policy Network Q(s, a; θ)
self.policy_net = nn.Sequential(
    nn.Linear(9, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 4) # Outputs Q-values for the 4 actions
)
```

## 6. The Loss Function and Target Network

To train the neural network, we need to define a Loss function. DQN typically uses the **Huber Loss** (in PyTorch, `SmoothL1Loss`) instead of Mean Squared Error (MSE). Huber loss acts like MSE when the error is small, but becomes linear (like Mean Absolute Error) when the error is large. This prevents massive rewards from creating exploding gradients that destabilize training.

The TD Target $Y_t$ derived from the Bellman Equation relies on the maximum Q-value of the next state:
$$Y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$

**The Moving Target Problem:** If we use the same network parameters $\theta$ to calculate both the prediction we are optimizing *and* the target $Y_t$, the optimization becomes highly unstable.

**Solution (Target Network):** We introduce a second, frozen copy of the network called the Target Network with parameters $\theta^-$. We use $\theta^-$ to calculate the stable target, and $\theta$ to calculate the prediction. 

The Loss function $L(\theta)$ computes the Huber loss between the prediction and target over a batch:
$$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim U(D)} \left[ \text{Huber}\left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right) \right]$$

Here is how this exact mathematical formula is calculated in the `Agent.train_step()` method:

```python
# Compute Q(S_t, A_t; θ) - the current state-action values
q_values = self.policy_net(state_batch).gather(1, action_batch)

# Compute max_{a'} Q(S_{t+1}, a'; θ⁻) - the target network's best next actions
next_q_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1)

# Compute the TD Target Y_t: r + γ * max Q(s', a'; θ⁻)
# We multiply by (1 - done) because if the game ends, there is no future return.
expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

# Compute L(θ): Huber Loss between Q(s, a; θ) and Y_t
loss = nn.SmoothL1Loss()(q_values, expected_q_values)
```

## 7. Stochastic Gradient Descent and Experience Replay

To minimize $L(\theta)$, we update the weights $\theta$ using Stochastic Gradient Descent (SGD) or, more specifically, the Adam optimizer. For a learning rate $\alpha$, the update rule is loosely:
$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$

In `agent.py`, doing this explicitly is handled by PyTorch's automatic differentiation. We also globally clip the gradient norm to prevent updates from becoming destructively large:
```python
self.optimizer.zero_grad()    # Clear old gradients
loss.backward()               # Compute gradient: ∇_θ L(θ)
torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0) # Scale gradients
self.optimizer.step()         # Apply update: θ ← θ - α ∇_θ L(θ)
```

**Experience Replay:** Notice the expectation $\mathbb{E}_{(s, a, r, s') \sim U(D)}$ in the loss function. $U(D)$ means a uniform distribution over a replay memory dataset $D$. 
Instead of computing the gradient on sequential transitions, we store transitions $e_t = (S_t, A_t, R_{t+1}, S_{t+1})$ in a Replay Buffer. During the training step, we sample a mini-batch of $N=64$ transitions uniformly at random. This breaks temporal correlations in the data, drastically stabilizing the gradient.

## 8. Epsilon-Greedy Exploration

To gather data to train the network, the agent must interact with the environment. It uses an **$\epsilon$-greedy policy** to balance exploration (gathering data) and exploitation (using its Q-network to get high rewards):
$$
\pi(A_t \mid S_t) = 
\begin{cases} 
\text{random action from } \mathcal{A} & \text{with probability } \epsilon_t \\
\arg\max_{a} Q(S_t, a; \theta) & \text{with probability } 1 - \epsilon_t 
\end{cases}
$$

The exploration rate $\epsilon_t$ decays exponentially at each step $t$ according to the formula:
$$\epsilon_t = \epsilon_{end} + (\epsilon_{start} - \epsilon_{end}) \cdot \exp\left(-\frac{t}{\epsilon_{decay}}\right)$$

This exact equation lives in `agent.py`:
```python
epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
          math.exp(-1. * self.steps_done / self.epsilon_decay)
```
