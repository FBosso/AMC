#import section
from torch import nn
import torch
import math
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self,features_in):
        #run init of parent module
        super().__init__()
        #define layers
        self.l1 = nn.Linear(features_in,50)
        self.l2 = nn.Linear(50,50)
        self.l3 = nn.Linear(50,features_in)
        #define activation functions
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
    
    def forward(self,x):
        #do forward pass
        x = self.l1(x)
        x = self.tanh(x)
        x = self.l2(x)
        x = self.tanh(x)
        x = self.l3(x)
        x = self.sigm(x)
        return x
    
    

class FusionSelfAttn(nn.Module):
    def __init__(self, input_dim=128, chunk_size=32, d_model=10, num_classes=10):
        super(FusionSelfAttn, self).__init__()
        assert input_dim % chunk_size == 0, "Input dimension must be divisible by chunk size"
        
        self.num_tokens = input_dim // chunk_size  # 128 / 32 = 4 tokens
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Linear projections for Q, K, V: project from 32 -> 10
        self.q_proj = nn.Linear(chunk_size, d_model)
        self.k_proj = nn.Linear(chunk_size, d_model)
        self.v_proj = nn.Linear(chunk_size, d_model)

        # Positional encoding for 4 tokens
        self.pos_encoding = nn.Parameter(self._generate_pos_encoding(self.num_tokens, d_model), requires_grad=False)

        # Output MLP head for classification
        self.output_proj = nn.Linear(d_model, num_classes)

    def _generate_pos_encoding(self, seq_len, d_model):
        """Standard sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

    def forward(self, x):
        """
        x: Tensor of shape (N, 128)
        """
        N = x.size(0)

        # Chunk into 4 tokens of size 32: shape becomes (N, 4, 32)
        x = x.view(N, self.num_tokens, self.chunk_size)

        # Apply Q, K, V projections → shape: (N, 4, 10)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Add positional encoding
        Q = Q + self.pos_encoding  # shape: (N, 4, 10)
        K = K + self.pos_encoding
        V = V  # (optionally, you could add PE to V too)

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)  # (N, 4, 4)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N, 4, 4)
        attn_output = torch.matmul(attn_weights, V)  # (N, 4, 10)

        # Aggregate (mean pooling over tokens)
        pooled = attn_output.mean(dim=1)  # (N, 10)

        # Final classification
        logits = self.output_proj(pooled)  # (N, num_classes)
        return logits
