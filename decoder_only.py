"""
Decoder-Only Transformer (GPT-style Architecture)
==================================================

This module implements a decoder-only transformer, similar to GPT.

Key Characteristics:
- Uses causal (masked) attention - can only see past tokens
- Best for: text generation, completion, language modeling
- Generates text autoregressively (one token at a time)
- Each token can only attend to PREVIOUS tokens (not future)

Use Cases:
- Text completion: "Once upon a" -> "time"
- Code generation: "def fibonacci(" -> "n):"
- Creative writing: Story continuation
- Chatbots: Generate responses

Architecture Overview:
Input -> Token Embedding + Positional Encoding -> 
Decoder Layers (Masked Self-Attention + FFN) -> 
Language Model Head -> Next Token Prediction

Key Difference from Encoder:
- Causal Mask: Prevents looking ahead
- Autoregressive: Generates one token at a time
- Training: Predict next token given all previous tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from toy_dataset import create_dataloaders


class PositionalEncoding(nn.Module):
    """
    Same as encoder version - adds position information to embeddings.
    See encoder_only.py for detailed explanation.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional causal masking.
    Same as encoder version but supports causal attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention with masking support.
        """
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = self.combine_heads(attn_output)
        output = self.W_o(output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Same as encoder version.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer with Causal (Masked) Self-Attention.
    
    Architecture:
        Input
          ↓
        Masked Multi-Head Self-Attention (causal mask)
          ↓
        Add & Norm
          ↓
        Feed-Forward Network
          ↓
        Add & Norm
          ↓
        Output
    
    Key Difference from Encoder Layer:
    - Uses CAUSAL MASK in self-attention
    - Prevents attending to future positions
    - Essential for autoregressive generation
    
    Why Causal Mask?
    - During training: We know the full sequence but must learn to predict
      each token from only previous tokens
    - During inference: We only have previous tokens available
    - Mask ensures training matches inference conditions
    
    Example with "The cat sat":
        Token "The" can attend to: ["The"]
        Token "cat" can attend to: ["The", "cat"]
        Token "sat" can attend to: ["The", "cat", "sat"]
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, seq_length, d_model)
            mask: Combined padding + causal mask (batch_size, 1, seq_length, seq_length)
            
        Returns:
            Output (batch_size, seq_length, d_model)
        """
        # Masked Self-Attention Block (Pre-Norm)
        normed = self.norm1(x)
        attn_output, _ = self.self_attention(normed, normed, normed, mask)
        x = x + self.dropout1(attn_output)
        
        # Feed-Forward Block (Pre-Norm)
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout2(ff_output)
        
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Complete Decoder-Only Transformer (GPT-style).
    
    Architecture:
        Token Embeddings + Positional Encoding
          ↓
        Decoder Layer 1 (with causal mask)
          ↓
        Decoder Layer 2 (with causal mask)
          ↓
        ...
          ↓
        Decoder Layer N (with causal mask)
          ↓
        Language Model Head (predicts next token)
    
    Training Objective:
    - Given sequence: [w1, w2, w3, w4]
    - Predict: w2 from w1, w3 from [w1,w2], w4 from [w1,w2,w3]
    - This is called "causal language modeling"
    
    Inference:
    - Start with prompt: "Once upon a"
    - Generate next token: "time"
    - Add to sequence: "Once upon a time"
    - Repeat until done
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 512,
        max_seq_length: int = 100,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Language model head (projects to vocabulary)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Tie weights: share embeddings between input and output
        # This is a common technique that:
        # 1. Reduces parameters
        # 2. Helps the model learn better representations
        # 3. Used in GPT and many other models
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask to prevent attending to future positions.
        
        The mask is a lower triangular matrix:
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]
        
        Where 1 means "can attend" and 0 means "cannot attend".
        
        Args:
            seq_length: Length of sequence
            device: Device to create mask on
            
        Returns:
            Causal mask (1, 1, seq_length, seq_length)
        """
        # Create lower triangular matrix
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        
        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)
        
        return mask
    
    def create_padding_mask(self, x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Create mask for padding tokens.
        
        Args:
            x: Input indices (batch_size, seq_length)
            pad_idx: Index of padding token
            
        Returns:
            Padding mask (batch_size, 1, 1, seq_length)
        """
        mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_combined_mask(
        self,
        x: torch.Tensor,
        pad_idx: int = 0
    ) -> torch.Tensor:
        """
        Create combined padding + causal mask.

        We need both:
        1) Padding mask: don't attend to padding tokens
        2) Causal mask: don't attend to future tokens

        Args:
            x: Input indices (batch_size, seq_length)
            pad_idx: Index of padding token
        Returns:
            (batch_size, 1, seq_length, seq_length) float
        """
        batch_size, seq_length = x.size()
        device = x.device

        # Padding mask: (B, 1, 1, S), bool
        padding_mask = self.create_padding_mask(x, pad_idx).to(torch.bool)

        # Causal mask: (1, 1, S, S), bool
        causal_mask = self.create_causal_mask(seq_length, device).to(torch.bool)

        # Combine with broadcasting: both must be True to allow attention
        # Result: (B, 1, S, S), bool
        combined_mask = padding_mask & causal_mask
        # or: combined_mask = torch.logical_and(padding_mask, causal_mask)

        # We want 1 where combined_mask==True and 0 otherwise
        additive_mask = torch.where(combined_mask == True, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        
        return additive_mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token indices (batch_size, seq_length)
            mask: Optional pre-computed mask (batch_size, 1, seq_length, seq_length)
            
        Returns:
            logits: Next token prediction logits (batch_size, seq_length, vocab_size)
        """
        # Create combined mask if not provided
        if mask is None:
            mask = self.create_combined_mask(x)
        
        # Token embeddings: (batch_size, seq_length, d_model)
        x = self.embedding(x)
        
        # Scale embeddings
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Project to vocabulary: (batch_size, seq_length, vocab_size)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_idx: int = 0
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        This is the core inference method for text generation.
        
        Strategies:
        1. Greedy: Always pick most likely token (deterministic)
        2. Temperature: Control randomness (low = conservative, high = creative)
        3. Top-k: Sample from k most likely tokens
        4. Top-p (nucleus): Sample from smallest set with cumulative prob >= p
        
        Args:
            prompt: Starting tokens (batch_size, prompt_length)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability >= p
            pad_idx: Padding token index
            
        Returns:
            Generated sequence (batch_size, prompt_length + max_new_tokens)
        """
        self.eval()  # Set to evaluation mode
        
        # Start with prompt
        generated = prompt
        
        for _ in range(max_new_tokens):
            # Truncate if sequence exceeds max length
            if generated.size(1) > self.max_seq_length:
                generated = generated[:, -self.max_seq_length:]
            
            # Get predictions for next token
            logits = self.forward(generated)
            
            # Get logits for last position only
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                # Keep only top k logits, set others to -inf
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                logits = logits_filtered
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False
                
                # Set filtered logits to -inf
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have generated EOS token (if applicable)
            # For simplicity, we'll just generate max_new_tokens
        
        return generated


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch using causal language modeling.
    
    Loss Calculation:
    - For each position, predict the next token
    - Given [w1, w2, w3, w4], compute:
      - Loss at pos 0: predict w2 given w1
      - Loss at pos 1: predict w3 given [w1, w2]
      - Loss at pos 2: predict w4 given [w1, w2, w3]
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        # Use decoder_input and decoder_output from batch
        # decoder_input: <SOS> + tokens (input to model)
        # decoder_output: tokens + <EOS> (target to predict)
        x = batch['decoder_input'].to(device)
        y = batch['decoder_output'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        
        # Reshape for loss calculation
        # logits: (batch_size, seq_length, vocab_size)
        # y: (batch_size, seq_length)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


# Main training script
if __name__ == "__main__":
    print("=" * 80)
    print("DECODER-ONLY TRANSFORMER (GPT-STYLE)")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataset...")
    train_loader, val_loader, dataset = create_dataloaders(batch_size=32, num_samples=1000)
    
    # Model hyperparameters
    vocab_size = max(dataset.src_vocab_size, dataset.tgt_vocab_size)
    d_model = 128
    num_layers = 3
    num_heads = 4
    d_ff = 512
    dropout = 0.1
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  Dropout: {dropout}")
    
    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("GENERATION DEMO")
    print("=" * 80)
    
    # Demo generation
    model.eval()
    
    # Create a prompt: "1 2"
    prompt_tokens = [dataset.src_token2idx['1'], dataset.src_token2idx['2']]
    prompt = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    
    print(f"\nPrompt: {[dataset.src_idx2token[idx] for idx in prompt_tokens]}")
    
    # Generate with different strategies
    print("\n1. Greedy decoding (temperature=0.1):")
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.1)
    generated_tokens = generated[0].tolist()
    print(f"   Generated: {[dataset.src_idx2token.get(idx, '<UNK>') for idx in generated_tokens]}")
    
    print("\n2. Sampling with temperature=1.0:")
    generated = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    generated_tokens = generated[0].tolist()
    print(f"   Generated: {[dataset.src_idx2token.get(idx, '<UNK>') for idx in generated_tokens]}")
    
    print("\n3. Top-k sampling (k=5):")
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=5)
    generated_tokens = generated[0].tolist()
    print(f"   Generated: {[dataset.src_idx2token.get(idx, '<UNK>') for idx in generated_tokens]}")
    
    print("\n4. Top-p (nucleus) sampling (p=0.9):")
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_p=0.9)
    generated_tokens = generated[0].tolist()
    print(f"   Generated: {[dataset.src_idx2token.get(idx, '<UNK>') for idx in generated_tokens]}")
    
    print("\n" + "=" * 80)
    print("Training completed! ✓")
    print("=" * 80)
    
    # Save model
    torch.save(model.state_dict(), 'decoder_only_model.pt')
    print("\nModel saved to 'decoder_only_model.pt'")
