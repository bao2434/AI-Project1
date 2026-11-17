import torch
from Policy import PPO_Policy, predict_action
from Agents import MinimaxAgentSummary, MCTS_Agent_v7, WrongWayBot, HardWaller
checkpoint = torch.load(
    "ppo_corridor_model.pt",
    weights_only=True
)

model1 = MinimaxAgentSummary()

# model1.load_state_dict(checkpoint["model_state_dict"])
# model1.eval()

model2 = WrongWayBot()

def run1(obs):
    return model1(obs)
    # return predict_action(model1, obs, deterministic=True)

def run2(obs):
    return model2(obs)

from Arena import run_simulation


print("meow")
result = run_simulation(run1, run2, seed=42, names=["Alice", "Bob"])
print(result.winner, result.winner_name)   # 1  'Bob'
print(result.total_turns)  