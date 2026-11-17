from MeowMeowBot.sub_architecture import *

import torch
import torch.nn as nn

class Policy_Model(nn.Module):

    BOARD_SIZE = 9

    NUM_MOVES = 81
    NUM_HORIZONTAL_WALLS = 64
    NUM_VERTICAL_WALLS = 64

    ACTION_DIM = NUM_MOVES + NUM_HORIZONTAL_WALLS + NUM_VERTICAL_WALLS  # 209

    def __init__(
        self,
        input_channels=12,
        embed_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        num_layers=8,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.conv_layer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Learnable CLS token: (1, 1, D), expanded to the batch in forward()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.rope_2d = RoPE2D(dim=embed_dim)

        # ModuleList (not Sequential) so we can pass rope / extra args to each block
        self.blocks = nn.ModuleList([
            EncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.final_projection = nn.Linear(embed_dim, self.ACTION_DIM)

        self._init_cls_token()

    def _init_cls_token(self):
        # Small truncated normal, standard for ViT-style CLS tokens
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, action_mask=None):
        """
        x:           (B, C, 9, 9) board state planes
        action_mask: optional (B, ACTION_DIM) bool tensor, True = legal action
        returns:     (B, ACTION_DIM) logits
        """
        B = x.shape[0]

        # (B, C, 9, 9) -> (B, D, 9, 9) -> (B, 81, D)
        x = self.conv_layer(x)
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS -> (B, 82, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x, rope=self.rope_2d)

        x = self.norm(x)
        cls_out = x[:, 0]                      # (B, D)

        logits = self.final_projection(cls_out)  # (B, 209)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return logits
        
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
    
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

        return logits


