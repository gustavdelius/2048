# Analysis of Learned Embeddings in 2048 DQN

Based on a mathematical analysis of the raw embedding weights learned by the Deep Q-Network (`best_model.pth`), we can observe that the network has learned meaningful structures and representations of the 2048 game state without explicit mathematical programming.

## 1. The "High-Value" Cluster
By calculating the cosine similarity (directional alignment) between the embeddings of consecutive tiles, we see a distinct pattern for high-value tiles:

*   `64` and `128` have a cosine similarity of **0.89**
*   `128` and `256+` have a cosine similarity of **0.93**

The model groups these high-value tiles closely together, pointing in almost the exact same direction in the 4-dimensional latent space. The network realizes that having a 64, 128, or 256 is fundamentally the same type of "good" state component in the late game.

## 2. The "Spawn vs. Merged" Split (Dimension 4)
Looking at the 4th value in the embedding vectors, it acts like a categorical binary switch:

*   `Empty` (0): **-7.6**
*   `2`: **-7.6**
*   `4` through `256+`: Fluctuates tightly between **-1.7 and +2.6**

The game environment spawns `2`s (and occasionally `4`s) into `Empty` spots. The network has empirically learned that empty squares and squares with a newly spawned `2` share a functional similarity: they represent the "background noise" or the stochastic elements of the board where new tiles randomly appear. This is completely distinct from the "constructed" tiles that the agent has actively merged.

## 3. The "Progression" Feature (Dimension 3)
The 3rd value in the embedding vectors acts almost like a smooth, monotonic progress bar indicating tile value:

*   `4`: **+6.3**
*   `8`: **+3.4**
*   `16`: **-3.1**
*   `32`: **-5.2**
*   `64`: **-7.7**
*   `128`: **-11.6**

As the tile value increases, this specific neuron's weight becomes increasingly negative in a nearly monotonic way. The model has organically invented a continuous scalar axis that ranks the intermediate merged tiles from low to high.

## 4. Non-Linear Mathematics and Categorization
Analyzing the *difference vectors* (e.g., comparing the step from `2` $\rightarrow$ `4` to the step from `4` $\rightarrow$ `8`), the cosine similarities between these progression steps are near zero or negative (-0.08 to -0.18 for early tiles).

This indicates that the network has **not** learned that the tiles form a simple linear mathematical sequence (it doesn't think "8 is just 4 plus another standard step"). Instead, it treats the game state categorially, grouping tiles into functional strategic concepts based on their role in receiving rewards:

1. **"Trash/Spawns"**: (`0`, `2`)
2. **"Middle building blocks"**: (`4`, `8`, `16`, `32`)
3. **"The important payload"**: (`64`, `128`, `256+`)

This demonstrates the power of Deep RL embeddings compared to simple one-hot encoding—the network has derived the strategic categories of the 2048 game purely by observing environment dynamics and trying to maximize its score.
