# PPO Training Process for the 9×9 Corridor Game

## 1. Objective

Train a PPO (Proximal Policy Optimization) agent to play the 9×9 Corridor game using simulated self-play and games against fixed opponents.

The agent uses an Actor-Critic neural network:

- **Actor (policy head):** produces action logits for the available actions.
- **Critic (value head):** estimates the expected discounted return from the current state.

The training loop repeatedly collects experience, calculates advantages with Generalized Advantage Estimation (GAE), and updates the network using Adam.

## 2. Training configuration

| Parameter | Value | Meaning |
|---|---:|---|
| Discount factor (`gamma`) | `0.99` | Determines how strongly future rewards are valued. |
| GAE lambda (`gae_lambda`) | `0.95` | Controls how TD errors are combined when estimating advantages. |
| Learning rate | `3e-4` | Adam optimizer learning rate; keep constant initially. |
| Parallel environments | `16` | Number of independent game environments collecting experience. |
| Steps per environment per rollout | `512` | Number of learning-agent decision transitions collected by each environment before an update. |
| Rollout buffer size | `8,192` | Total transitions collected per rollout: `16 × 512`. |
| Minibatch size | `512` | Number of transitions used for one gradient update. |
| PPO epochs | `4` | Number of passes over the rollout data per update cycle. |
| PPO clip range | `0.2` | Limits the policy probability ratio in PPO's clipped objective. |
| Entropy coefficient | `0.01` initially | Encourages policy exploration; consider reducing to `0.001` if exploration remains excessive. |
| Value-loss coefficient | `0.5` | Weight applied to the critic loss. |
| Gradient clipping | `0.5` | Maximum gradient norm used to stabilize optimization. |
| Win reward | `+1` | Reward when the learning agent wins. |
| Loss reward | `-1` | Reward when the learning agent loses. |
| Draw reward | `0` | Reward for a draw. |
| Optional step penalty | `0` initially | Small per-agent-decision penalty to discourage unnecessarily long games; for example, `-0.001`. |

These are starting hyperparameters, not guaranteed optimal settings. Tune them based on learning curves, policy stability, game outcomes, and runtime.

## 3. Opponent setup

Use a mixture of opponent types to make training less dependent on one opponent strategy.

A possible initial allocation across 16 environments is:

| Environments | Opponent |
|---|---|
| 1–8 | Current learning policy (self-play) |
| 9–12 | Fixed baseline policy |
| 13–16 | Older frozen PPO checkpoint, when available |

The exact proportions should be configurable through a list or dictionary supplied by the user. For example:

```python
opponent_config = {
    "self": 8,
    "fixed_baseline": 4,
    "checkpoint_pool": 4,
}
```

Note that there could be multiple type of fixed_basedline !!

### Important policy distinction

The learning agent's decisions are the transitions used to train its policy. If the opponent is a fixed model, its actions are not training samples for the learning agent.

For self-play, decide explicitly whether one shared policy controls both sides or whether only one side is the learner.

- **One-sided learning:** the current policy controls Player A; Player B is a frozen opponent or past checkpoint. Store and update only Player A's transitions.
- **Symmetric self-play:** the same current policy controls both Player A and Player B. Store both players' transitions separately, assign each player the correct reward perspective, and update the shared network using both trajectories.

Include control to fixed between these mods.

If training from both players' decisions, convert rewards and value targets to the correct player's perspective and store the corresponding old log probabilities. For example, if Player A wins, Player A receives `+1` and Player B receives `-1`.

### Rollout policy vs. trainable policy

Do **not** freeze the actor-critic network while simulating games. During rollout, use the current actor-critic to select actions and compute values, but run inference without gradients.

What must remain fixed after collection is the **rollout data**:

- Actions selected.
- Rewards received.
- Old log-probabilities.
- Critic values collected during rollout.
- Action masks.
- Episode boundaries and `done` flags.
- GAE advantages and return targets, after they are calculated.

The current actor-critic remains trainable during the PPO optimization phase. For each minibatch, recompute the new log-probability and value estimate using the updated network, while comparing them against the stored old log-probability and return target.

## 4. Data collected at each agent decision

For every learning-agent decision, store at least:

| Field | Description |
|---|---|
| `state` | Observation supplied to the policy. |
| `action` | Action selected by the actor. |
| `reward` | Reward received from this learning agent's perspective. |
| `value` | Critic estimate `V(s_t)` at collection time. |
| `old_log_prob` | Log probability of the selected action under the rollout policy. |
| `done` | Whether the game truly terminated after this transition. |
| `truncated` | Whether the episode ended because of a rollout or time limit, rather than true game termination. |
| `player_id` | Which player made the decision, such as `0` or `1`. |
| `game_id` | Identifier used to group transitions belonging to the same game and player. |
| `action_mask` | Legal-action mask used when sampling and evaluating the action. |

Also store any information needed to reconstruct observations, action masks, or recurrent state if the model uses them.

Apply the legal-action mask before sampling an action and before calculating its log probability and entropy.

### What counts as a step?

For this setup, a transition is one decision made by the learning agent. If the opponent acts between two learning-agent decisions, the environment should advance through the opponent's turn before producing the next learning-agent observation.

Be consistent about when rewards are assigned. If the reward is intended to be from the learning agent's perspective, an opponent win must produce `-1` for the learning agent.

## 5. Collect one rollout

Run 16 independent environments. Collect 512 learning-agent decision transitions from each environment.

Therefore:

```text
16 environments × 512 transitions = 8,192 transitions
```

An environment may finish a game before collecting 512 transitions. Reset it and continue collecting until it has contributed 512 transitions to the rollout.

An environment may also reach the 512-transition collection limit while a game is still in progress. This is a rollout boundary, not a terminal state. Do not mark it as `done` merely because collection stopped.

The games can have different lengths. Store episode boundaries so that advantage calculations never leak from one game into another.

For symmetric self-play, either:

- Maintain separate rollout buffers for Player A and Player B, each containing 8,192 transitions; or
- Maintain one combined 8,192-transition buffer per player side and clearly label every transition with its `player_id` and `game_id`.

For an initial implementation, separate buffers for each player are recommended because they make reward assignment, GAE computation, and debugging easier.

## 6. Calculate GAE and return targets

Calculate GAE separately for each **player trajectory within each completed game**. Do not calculate GAE across unrelated transitions, different games, or different players.

For a true terminal transition, the bootstrap value after the terminal state is zero.

The temporal-difference residual is:

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

The GAE advantage is:

\[
A_t = \delta_t + \gamma\lambda A_{t+1}
\]

Use:

- `gamma = 0.99`
- `gae_lambda = 0.95`

Calculate the recursion backwards through each player trajectory.

The critic's return target is:

\[
R_t = A_t + V(s_t)
\]

### Terminal states and rollout boundaries

Handle these cases differently:

1. **True game termination:** use a next-state bootstrap value of zero and stop the GAE recursion at the episode boundary.
2. **Rollout ends but game continues:** bootstrap from the critic's value of the next learning-agent state. Do not treat this as a terminal state.
3. **Environment reset:** never allow GAE to propagate across the reset into a different episode.
4. **Opponent moves between learning-agent decisions:** advance the environment through the opponent's move before storing the next learning-agent transition.

A robust implementation stores values for every learning-agent state and one additional bootstrap value at the end of a nonterminal segment:

```text
values = [V(s_0), V(s_1), ..., V(s_T), V_bootstrap]
```

For a terminal game, `V_bootstrap = 0`.

## 7. Prepare the PPO training batch

After calculating advantages and returns for all player trajectories, combine the transitions into one rollout dataset.

The dataset contains up to 8,192 transitions per learning side, with:

- States
- Actions
- Old log probabilities
- Advantages
- Return targets
- Action masks, if needed
- Player and game identifiers, if using a combined buffer

Normalize advantages across the rollout as a common PPO practice:

\[
\hat A_t =
\frac{A_t-\operatorname{mean}(A)}
{\operatorname{std}(A)+\epsilon}
\]

Use the normalized advantages for the policy loss. Keep the original return targets for critic training.

Shuffle the rollout transitions before creating minibatches. Do not shuffle the temporal order before calculating GAE.

If using separate Player A and Player B buffers, normalize advantages separately for each buffer, then either optimize them separately or concatenate them after normalization.

## 8. Optimize the policy and critic

Use Adam with a constant initial learning rate of `3e-4`.

For each rollout:

- Perform 4 PPO epochs.
- Use minibatches of 512 transitions.
- Each epoch contains `8,192 / 512 = 16` minibatches.
- This gives `4 × 16 = 64` optimizer updates per rollout.

For each minibatch, calculate:

### Policy probability ratio

\[
r_t(\theta) =
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\mathrm{old}}(a_t \mid s_t)}
\]

In code, calculate this stably using log probabilities:

```python
ratio = torch.exp(new_log_prob - old_log_prob)
```

### Clipped policy objective

\[
L^{CLIP} =
\mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat A_t,\,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
\]

where `epsilon = 0.2`.

When implementing gradient descent, the policy loss is the negative of this maximization objective.

### Value loss

A basic critic loss is:

\[
L_V = \mathbb{E}_t[(V_\theta(s_t)-R_t)^2]
\]

### Entropy bonus

Calculate policy entropy over legal actions:

\[
H(\pi_\theta(\cdot \mid s_t))
\]

The entropy bonus encourages exploration.

### Combined loss

A common minimized loss is:

\[
L =
-L^{CLIP}
+ 0.5L_V
- c_e H
\]

Start with `c_e = 0.01`. If the policy remains excessively random after checking its behavior and entropy, consider reducing it to `0.001`.

### Gradient update

For every minibatch:

1. Clear gradients with `optimizer.zero_grad()`.
2. Calculate the actor, critic, and entropy losses.
3. Backpropagate with `loss.backward()`.
4. Clip gradients to a maximum norm of `0.5`.
5. Update parameters with `optimizer.step()`.

The old log probabilities, GAE advantages, and return targets stay fixed throughout the four PPO epochs. The current actor-critic parameters are updated during every minibatch.

## 9. Repeat the training cycle

The complete training loop is:

```text
Initialize actor-critic model
Initialize Adam optimizer
Initialize opponent pool

while training:

    1. Select opponents for the 16 environments
    2. Collect 512 learning-agent transitions per environment
       using the current actor-critic in inference mode
    3. Store states, actions, rewards, values,
       old log probabilities, masks, player IDs,
       game IDs, and episode boundaries
    4. Calculate GAE separately for each player trajectory
       within each game
    5. Calculate critic return targets
    6. Normalize advantages
    7. Combine and shuffle the 8,192 transitions
    8. Run 4 PPO epochs with minibatches of 512
    9. Monitor training metrics
   10. Save checkpoints when appropriate
   11. Periodically add the current policy to the opponent pool
   12. Start the next rollout using the updated policy
```

## 10. Metrics to monitor

Track these metrics over training:

| Metric | What it indicates |
|---|---|
| Win rate | How often the agent wins against each opponent type. |
| Draw rate | How often games end in a draw. |
| Average game length | Whether the agent is learning to finish games efficiently. |
| Policy loss | Actor optimization behavior. |
| Value loss | Critic prediction error. |
| Entropy | How exploratory or deterministic the policy is. |
| Approximate KL divergence | How far the updated policy moves from the rollout policy. |
| Clip fraction | Fraction of samples whose probability ratios are clipped. |
| Explained variance | How well the critic predicts return targets. |
| Rollout collection time | Time required to collect 8,192 transitions. |
| Optimization time | Time required for the 64 minibatch updates. |

Evaluate against fixed opponents separately from self-play. Self-play win rate alone can be misleading because both sides may improve together or exploit weaknesses specific to one another.

Consider saving the best checkpoint based on evaluation performance against a consistent set of fixed opponents, not only the latest training checkpoint.

## 11. Initial implementation checklist

- [ ] Implement 16 independent game environments.
- [ ] Define rewards consistently from the learning agent's perspective.
- [ ] Implement legal-action masking.
- [ ] Collect 512 learning-agent transitions per environment.
- [ ] Store old action log probabilities and critic values during rollout.
- [ ] Preserve episode boundaries and distinguish terminal states from rollout cutoffs.
- [ ] Implement GAE separately for each player and each game trajectory.
- [ ] Calculate return targets and normalize advantages.
- [ ] Implement PPO clipped policy loss.
- [ ] Implement critic value loss and entropy bonus.
- [ ] Apply gradient clipping and Adam updates.
- [ ] Run four epochs of 512-transition minibatches.
- [ ] Track evaluation win rate, entropy, KL divergence, and explained variance.
- [ ] Save and evaluate checkpoints.
- [ ] Periodically add current-policy snapshots to the opponent pool.

## 12. Practical notes

1. **Start simple:** Initially use a fixed baseline opponent and a designated learning side. Add self-play and older checkpoints after the basic PPO pipeline works.
2. **Validate GAE:** Test it on a small hand-calculated trajectory, including a terminal episode and a nonterminal rollout cutoff.
3. **Avoid data leakage:** Do not calculate GAE across episode boundaries, environment resets, or different players.
4. **Check legal actions:** Invalid actions should not receive probability mass during sampling or entropy calculation.
5. **Use inference mode during rollout:** Collect actions, log-probabilities, and values with `torch.no_grad()`; do not update network parameters while simulating.
6. **Keep rollout data fixed:** After GAE, advantages, returns, and old log-probabilities are stored, they must not change during PPO optimization.
7. **Measure CPU performance:** Sixteen environments are a starting configuration. Benchmark environment simulation, model inference, and optimization separately.
8. **Keep evaluation separate:** Do not use evaluation games as PPO training data unless they are intentionally collected as training rollouts.

This configuration provides a concrete starting point. Adjust it based on measured training stability, generalization against fixed opponents, and CPU throughput.