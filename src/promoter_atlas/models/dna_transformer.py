import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        return F.gelu(x + residual)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Shape should be [1, 1, seq_len, dim]
        return emb[None, None, :, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryAttentionBlock(nn.Module):
    """Rotary attention block with residual connection."""
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0, "dimension must be divisible by number of heads"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.to_qkv = nn.Linear(dim, 3 * dim)  # Combined Q,K,V projection
        self.to_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.norm(x)
        
        batch_size, seq_len, _ = x.shape
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Generate position embeddings
        position_embeddings = self.rotary_emb(seq_len)
        cos = position_embeddings.cos()
        sin = position_embeddings.sin()
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out = self.to_out(out)
        out = out.transpose(1, 2)
        return out + residual, attn_probs
        
class DNATransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        
        self.res_block = ResBlock(dim, dim)
        self.rotary_attention_block = RotaryAttentionBlock(dim, n_heads)
        self.feed_forward_up = nn.Linear(dim, dim*2)
        self.feed_forward_dn = nn.Linear(dim*2, dim)
    
    def forward(self, x):
        # First residual: Resnet block
        x = self.res_block(x)
        
        # Second residual: Rotary attention
        x, attn_scores = self.rotary_attention_block(x)
        
        # Third residual: Feed forward
        residual = x
        x = x.transpose(1, 2)
        x = F.gelu(self.feed_forward_up(x))
        x = self.feed_forward_dn(x)
        x = x.transpose(1, 2)
        x = x + residual
        
        return x, attn_scores

class DNATransformer(nn.Module):
    def __init__(self, dim=128, n_blocks=8, n_heads=8):
        super().__init__()
        
        self.embedding = nn.Conv1d(4, dim, kernel_size=1)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            DNATransformerBlock(dim=dim, n_heads=n_heads) 
            for _ in range(n_blocks)
        ])
        
        # Final projection to 4 nucleotides
        self.to_nucleotides = nn.Conv1d(dim, 4, kernel_size=1)
    
    def forward(self, x, return_latent=False):
        """Process DNA sequences through the transformer model."""
        # Already in channel-first format [batch, 4, seq_len]
        x = self.embedding(x)  # [batch, dim, seq_len]
        
        all_attention_scores = []
        for block in self.blocks:
            x, attn_scores = block(x)
            all_attention_scores.append(attn_scores)
        
        # Store final latent representation
        final_latent = x
        
        # Project to nucleotide logits [batch, 4, seq_len]
        logits = self.to_nucleotides(x)
        
        if return_latent:
            return logits, all_attention_scores, final_latent
        return logits, all_attention_scores
    
    def get_intermediate_representations(self, x):
        """Extract intermediate representations from each transformer block."""
        representations = {
            'block_outputs': [],
            'attention_scores': []
        }
        
        # Initial embedding
        x = self.embedding(x)  # [batch, dim, seq_len]
        representations['input_embedding'] = x.clone()
        
        # Process through blocks and collect intermediate outputs
        for block in self.blocks:
            x, attn_scores = block(x)
            representations['block_outputs'].append(x.clone())
            representations['attention_scores'].append(attn_scores)
        
        # Store final output before projection
        representations['final_output'] = x.clone()
        
        return representations