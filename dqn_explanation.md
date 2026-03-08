# How the 2048 AI Learns: A Guide to DQN

This guide explains how the reinforcement learning program for the 2x2 or 3x3 game of 2048 works, without requiring a background in advanced mathematics or machine learning.

## 1. What is Reinforcement Learning?
Imagine training a dog. When the dog does a trick correctly, you give it a treat (a reward). Over time, the dog figures out that certain actions in certain situations lead to treats.

Reinforcement Learning (RL) works exactly like this. There are a few key terms:

* **The Environment:** The game of 2048 itself. It has rules, a board, and handles spawning new tiles.
* **The Agent:** Our AI player that decides which move to make (Up, Down, Left, or Right).
* **The State:** A snapshot of the current 2048 board (what the board looks like right now).
* **The Action:** A move the agent decides to execute (e.g., swipe left).
* **The Reward:** The "treat." The agent gets a large reward for combining big tiles and a penalty if it makes an invalid move (swiping into a wall where nothing moves).

The goal of the agent is simple: **Maximize the total amount of reward it gets over the course of a game.**

## 2. Q-Learning: Evaluating Moves
How does the agent decide which action brings the most reward? It tries to predict the future. 

It calculates a **Q-Value** for every possible action in a given state. The "Q" stands for Quality. A Q-value is the agent's best guess of "If I am in this exact state, and I take this specific action, what is the *total future reward* I will get from this point until the end of the game?"

If the agent has perfectly accurate Q-values, it just has to look at its 4 possible moves, pick the one with the highest Q-value, and it will play a perfect game!

## 3. What makes it "Deep" Q-Learning (DQN)?
In simple games like Tic-Tac-Toe, there are few enough possible board states that you could just write down every single state in a massive spreadsheet (a Q-Table), along with the Q-values for the 4 actions.

In 2048, there are millions of possible board combinations. A spreadsheet would be too large and impossible to fill.

Instead of a table, we use a **Deep Neural Network**. A neural network acts as a complex function: you feed it the current board state as an input, and it outputs 4 numbers (the predicted Q-values for Up, Down, Left, and Right). This is why it is called a Deep Q-Network (**DQN**).

## 4. Key Concepts in Our Code

Here is how the training actually happens, integrating several crucial techniques to make the learning process stable and effective.

### A. How the Agent "Sees" the Board (The State)
Neural networks work best with small, manageable numbers. In 2048, tiles grow exponentially (2, 4, 8... 1024, 2048). Feeding a tile colored '2048' and a tile colored '2' into the network can overwhelm it because the numbers are so drastically different.

* **Our Solution:** We feed the network the *logarithm* (Base-2) of the tiles. A '2' becomes a 1, a '4' becomes a 2, an '8' becomes a 3, and '2048' becomes 11. This helps the network easily digest the board state.

### B. Balancing Trying New Things vs. Doing What Works (Epsilon-Greedy)
When the agent starts, its neural network represents random guesses. It knows nothing. If it always picked the "best" action according to its (random) knowledge, it might get stuck doing something stupid forever.

* **Our Solution:** The agent has a parameter called heavily **Epsilon** ($\epsilon$). This represents the chance of making a completely random move. 
* We start with $\epsilon = 1.0$ (100% random moves). This forces the agent to furiously explore the game and try every combination.
* Over thousands of games, we slowly decay $\epsilon$ down to $0.01$ (1% random moves). By the end, the agent is relying almost entirely on what its trained network has learned (exploitation), but occasionally tries something random just in case there's a better move it hasn't discovered.

### C. Learning from the Past (The Replay Buffer)
If the agent only learned from the *very last* move it made, it would quickly forget old lessons or get fixated on a weird string of recent events. 

* **Our Solution:** As the agent plays, it saves every single "Memory" (the state, the action it chose, the reward it got, and the new state it ended up in) into a giant memory bank called a Replay Buffer.
* During training, the agent grabs a random handful (a "batch") of 64 past memories and learns from them all at once. This breaks up the chronological sequence of moves and allows the agent to learn from a diverse range of past experiences at every step.

### D. The Two Brains (Target Network)
To train our network, we compare its predicted Q-value against a "Target" Q-value (the actual reward it received + the highest prediction for the *next* state).

* The problem: If we use the *same* network to make the prediction AND calculate the target, it's like a dog chasing its own tail. The target keeps moving every time the network updates.

* **Our Solution:** We actually use two exact copies of the neural network.
    1. The **Policy Network**: The one actively making decisions and being updated every single move.
    2. The **Target Network**: A frozen copy of the Policy Network. We use this frozen duplicate to calculate our target goals. Every few hundred moves, we copy the updated Policy Network over to the Target Network to refresh it. This keeps the training target stable!

## Summary of the Training Loop
In `train.py`, here is what happens millions of times:

1. The agent looks at the board.
2. It either picks a random move (exploration) or uses its Policy Network to pick the best known move (exploitation).
3. The move is executed in the game `env.py`.
4. The environment spits out a Reward and the new board State.
5. This whole interaction is saved into the Memory Bank.
6. The agent grabs 64 random memories from the Memory Bank.
7. It compares what its Policy Network predicted for those 64 memories against what the stable Target Network says they should be.
8. It tweaks the mathematical weights inside the Policy Network slightly to make its future predictions closer to the truth.
9. Every so often, the Target Network is updated to match the improved Policy Network.
