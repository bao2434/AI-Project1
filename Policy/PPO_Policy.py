
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 1. CLIP-style activation
# ============================================================

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# ============================================================
# 2. CLIP-style multi-head self-attention
# ============================================================

class CLIPAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # CLIP uses separate Q, K, V projections.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        def split_heads(tensor):
            return tensor.reshape(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim
            ).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        # Scaled dot-product attention.
        attention = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        attention = F.softmax(attention, dim=-1)

        output = torch.matmul(attention, v)

        # Merge attention heads.
        output = output.transpose(1, 2).contiguous()

        output = output.reshape(
            batch_size,
            seq_len,
            embed_dim
        )

        return self.out_proj(output)


# ============================================================
# 3. CLIP Transformer encoder block
# ============================================================

class CLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        mlp_dim=256
    ):
        super().__init__()

        self.self_attn = CLIPAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )

        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            QuickGELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # Pre-LayerNorm + attention residual.
        x = x + self.self_attn(
            self.layer_norm1(x)
        )

        # Pre-LayerNorm + MLP residual.
        x = x + self.mlp(
            self.layer_norm2(x)
        )

        return x


# ============================================================
# 4. CLIP-style Vision Transformer
# ============================================================

class CLIPVisionTransformer(nn.Module):

    def __init__(
        self,
        input_channels=12,
        image_size=9,
        kernel_size=3,
        stride=1,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=256
    ):
        super().__init__()

        assert image_size >= kernel_size
        assert stride > 0
        assert embed_dim % num_heads == 0

        self.image_size = image_size
        self.input_channels = input_channels
        self.embed_dim = embed_dim

        # Patch embedding.
        self.patch_embedding = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            padding="same",
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            num_patches = self.patch_embedding(dummy).flatten(2).shape[-1]

        # CLS token.
        self.class_embedding = nn.Parameter(
            torch.randn(embed_dim) * 0.02
        )

        # Positional embeddings.
        self.position_embedding = nn.Parameter(
            torch.randn(num_patches + 1, embed_dim) * 0.01
        )

        self.pre_layernorm = nn.LayerNorm(embed_dim)

        self.encoder = nn.ModuleList([
            CLIPEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim
            )
            for _ in range(num_layers)
        ])

        self.post_layernorm = nn.LayerNorm(embed_dim)

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.xavier_uniform_(
            self.patch_embedding.weight
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):

        batch_size = x.shape[0]

        assert x.ndim == 4
        assert x.shape[1] == self.input_channels
        assert x.shape[2] == self.image_size
        assert x.shape[3] == self.image_size

        # [B, C, H, W] -> [B, D, H', W']
        x = self.patch_embedding(x)

        # [B, D, H', W'] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS token.
        cls_token = self.class_embedding.view(
            1, 1, -1
        ).expand(batch_size, 1, -1)

        x = torch.cat([cls_token, x], dim=1)

        # Add positional embeddings.
        x = x + self.position_embedding.unsqueeze(0)

        # Pre-LayerNorm.
        x = self.pre_layernorm(x)

        # Transformer encoder.
        for layer in self.encoder:
            x = layer(x)

        # Extract CLS representation.
        cls_output = self.post_layernorm(x[:, 0])

        return cls_output


# ============================================================
# 5. Corridor CLIP Policy
# ============================================================

class PPO_Policy(nn.Module):

    BOARD_SIZE = 9

    NUM_MOVES = 81
    NUM_HORIZONTAL_WALLS = 64
    NUM_VERTICAL_WALLS = 64

    ACTION_DIM = (
        NUM_MOVES
        + NUM_HORIZONTAL_WALLS
        + NUM_VERTICAL_WALLS
    )

    GLOBAL_DIM = 3

    def __init__(
        self,
        input_channels=12,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=256
    ):
        super().__init__()

        self.vision_model = CLIPVisionTransformer(
            input_channels=input_channels,
            image_size=9,
            kernel_size=3,
            stride=1,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim
        )

        # Project non-spatial information into
        # the same representation space.
        self.global_projection = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, embed_dim),
            QuickGELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Fusion normalization.
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # One categorical distribution over all 209 actions.
        self.policy_head = nn.Linear(
            embed_dim,
            self.ACTION_DIM
        )

        # PPO critic.
        self.value_head = nn.Linear(
            embed_dim,
            1
        )

    def forward(self, board, global_features):

        # Extract spatial representation.
        visual_features = self.vision_model(board)

        # Extract global representation.
        global_features = self.global_projection(
            global_features
        )

        # Fuse both representations.
        features = self.fusion_norm(
            visual_features + global_features
        )

        logits = self.policy_head(features)

        value = self.value_head(features).squeeze(-1)

        return logits, value

    # --------------------------------------------------------
    # Apply legal-action masking.
    # --------------------------------------------------------

    @staticmethod
    def apply_action_mask(logits, action_mask):

        if not torch.all(action_mask.any(dim=-1)):
            raise ValueError(
                "At least one state has no legal actions."
            )

        return logits.masked_fill(
            ~action_mask.bool(),
            torch.finfo(logits.dtype).min
        )

    # --------------------------------------------------------
    # Sample an action during rollout.
    # --------------------------------------------------------

    def act(
        self,
        board,
        global_features,
        action_mask,
        deterministic=False
    ):

        logits, value = self(board, global_features)

        logits = self.apply_action_mask(
            logits,
            action_mask
        )

        distribution = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)

        entropy = distribution.entropy()

        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value
        }

    # --------------------------------------------------------
    # Decode an action index into a game action.
    # --------------------------------------------------------

    @staticmethod
    def decode_action(action):

        action = int(action)

        if 0 <= action < 81:
            r = action // 9
            c = action % 9
            return {
                "type": "move",
                "to": [r, c],
                "row": r,
                "col": c
            }

        if 81 <= action < 145:

            index = action - 81

            return {
                "type": "wall",
                "orientation": "H",
                "row": index // 8,
                "col": index % 8
            }

        if 145 <= action < 209:

            index = action - 145

            return {
                "type": "wall",
                "orientation": "V",
                "row": index // 8,
                "col": index % 8
            }

        raise ValueError("Invalid action index")


# ============================================================
# 6. Observation encoder
# ============================================================

def build_blocking_masks(observation):

    size = 9

    # Directional blocking masks.
    up = torch.zeros(size, size)
    down = torch.zeros(size, size)
    left = torch.zeros(size, size)
    right = torch.zeros(size, size)

    # Board boundaries.
    up[0, :] = 1
    down[8, :] = 1
    left[:, 0] = 1
    right[:, 8] = 1

    walls = observation["walls"]

    # Horizontal walls:
    # Block movement between row r and row r+1.
    for r, c in walls["horizontal"]:

        down[r, c] = 1
        down[r, c + 1] = 1

        up[r + 1, c] = 1
        up[r + 1, c + 1] = 1

    # Vertical walls:
    # Block movement between column c and column c+1.
    for r, c in walls["vertical"]:

        right[r, c] = 1
        right[r + 1, c] = 1

        left[r, c + 1] = 1
        left[r + 1, c + 1] = 1

    return up, down, left, right


def calculate_distances(goal_row, masks):

    up, down, left, right = [m.tolist() if isinstance(m, torch.Tensor) else m for m in masks]

    size = 9

    distances = [[-1.0] * size for _ in range(size)]

    queue = deque()

    # Every cell in the goal row is a target.
    for c in range(size):
        distances[goal_row][c] = 0.0
        queue.append((goal_row, c))

    directions = [
        (-1, 0, up),
        (1, 0, down),
        (0, -1, left),
        (0, 1, right)
    ]

    while queue:
        r, c = queue.popleft()
        d_next = distances[r][c] + 1.0

        for dr, dc, blocked in directions:
            nr = r + dr
            nc = c + dc

            if 0 <= nr < size and 0 <= nc < size:
                if blocked[r][c] == 0 and distances[nr][nc] == -1.0:
                    distances[nr][nc] = d_next
                    queue.append((nr, nc))

    return torch.tensor(distances, dtype=torch.float32)


def normalize_distances(distances):

    result = torch.where(
        distances < 0,
        torch.full_like(distances, 2.0),
        distances.clamp(max=80) / 80.0
    )

    return result


def encode_observation(
    observation,
    device="cpu",
    max_walls=10
):

    size = 9

    board = torch.zeros(12, size, size)

    you = observation["you"]
    opponent = observation["opponent"]

    # --------------------------------------------------------
    # Channel 0: Your position.
    # --------------------------------------------------------

    r, c = you["position"]
    board[0, r, c] = 1

    # --------------------------------------------------------
    # Channel 1: Opponent position.
    # --------------------------------------------------------

    r, c = opponent["position"]
    board[1, r, c] = 1

    # --------------------------------------------------------
    # Channel 2-3: Wall anchor positions.
    # --------------------------------------------------------

    for r, c in observation["walls"]["horizontal"]:
        board[2, r, c] = 1

    for r, c in observation["walls"]["vertical"]:
        board[3, r, c] = 1

    # --------------------------------------------------------
    # Channel 4-7: Directional blocking masks.
    # --------------------------------------------------------

    masks = build_blocking_masks(observation)

    for i, mask in enumerate(masks):
        board[4 + i] = mask

    # --------------------------------------------------------
    # Channel 8-9: Goal rows.
    # --------------------------------------------------------

    board[8, you["goal_row"], :] = 1
    board[9, opponent["goal_row"], :] = 1

    # --------------------------------------------------------
    # Channel 10: Your shortest-path distances.
    # --------------------------------------------------------

    own_distances = torch.tensor(
        observation["distances"],
        dtype=torch.float32
    )

    board[10] = normalize_distances(own_distances)

    # --------------------------------------------------------
    # Channel 11: Opponent shortest-path distances.
    # --------------------------------------------------------

    opponent_distances = calculate_distances(
        opponent["goal_row"],
        masks
    )

    board[11] = normalize_distances(
        opponent_distances
    )

    # --------------------------------------------------------
    # Global numerical features.
    # --------------------------------------------------------

    global_features = torch.tensor(
        [
            you["walls_remaining"] / max_walls,
            opponent["walls_remaining"] / max_walls,
            observation["turn"] / max(
                observation["max_turns"], 1
            )
        ],
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Legal-action mask.
    # --------------------------------------------------------

    action_mask = torch.zeros(
        PPO_Policy.ACTION_DIM,
        dtype=torch.bool
    )

    # Movement: indices 0-80.
    for r, c in observation["legal_moves"]:

        action_mask[r * 9 + c] = True

    # Horizontal walls: indices 81-144.
    # Vertical walls: indices 145-208.
    for wall in observation["legal_walls"]:

        r = wall["row"]
        c = wall["col"]

        if wall["orientation"] == "H":

            action_mask[81 + r * 8 + c] = True

        elif wall["orientation"] == "V":

            action_mask[145 + r * 8 + c] = True

    return (
        board.unsqueeze(0).to(device),
        global_features.unsqueeze(0).to(device),
        action_mask.unsqueeze(0).to(device)
    )


# ============================================================
# 7. Example inference
# ============================================================

def predict_action(
    model,
    observation,
    device="cpu",
    deterministic=True
):

    board, global_features, action_mask = encode_observation(
        observation,
        device=device
    )

    model.eval()

    with torch.inference_mode():

        result = model.act(
            board,
            global_features,
            action_mask,
            deterministic=deterministic
        )

    action_index = result["action"].item()

    return PPO_Policy.decode_action(
        action_index
    )




import time

if __name__ == "__main__":

    device = torch.device("cpu")

    model = PPO_Policy(
        input_channels=12,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=256
    ).to(device)

    model.eval()

    # Use your sample observation here.
    observation = {

        "you": {
            "position": [4, 4],
            "goal_row": 8,
            "walls_remaining": 10
        },

        "opponent": {
            "position": [2, 4],
            "goal_row": 0,
            "walls_remaining": 10
        },

        "walls": {
            "horizontal": [],
            "vertical": []
        },

        # Example shortest-path distances.
        "distances": [
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [6, 6, 6, 6, 6, 6, 6, 6, 6],
            [5, 5, 5, 5, 5, 5, 5, 5, 5],
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [3, 3, 3, 3, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],

        "turn": 10,
        "max_turns": 200,

        "player_id": 0,
        "current_player": 0,

        # Example legal movements.
        "legal_moves": [
            [3, 4],
            [4, 3],
            [4, 5],
            [5, 4]
        ],

        # No legal wall placements in this example.
        "legal_walls": []
    }

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(5):
        predict_action(
            model,
            observation,
            device=device,
            deterministic=True
        )

    # --------------------------------------------------------
    # Benchmark prediction
    # --------------------------------------------------------

    num_runs = 100
    total_time = 0.0

    for _ in range(num_runs):

        start_time = time.perf_counter()

        action = predict_action(
            model,
            observation,
            device=device,
            deterministic=True
        )

        end_time = time.perf_counter()

        elapsed = end_time - start_time
        total_time += elapsed

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    average_time = total_time / num_runs

    print("\n========== PPO Prediction Benchmark ==========")

    print("Predicted action:", action)
    print(f"Number of runs: {num_runs}")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"Average prediction: {average_time * 1000:.3f} ms")
    print(f"Average prediction: {average_time:.6f} seconds")
    print(f"Predictions per second: {1 / average_time:.2f}")

    print("==============================================")
