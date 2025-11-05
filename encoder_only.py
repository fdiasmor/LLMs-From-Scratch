"""
Encoder-Only Transformer (BERT-style Architecture)
===================================================

This module implements an encoder-only transformer, similar to BERT.

Key Characteristics:
- Uses bidirectional (full) attention - can see all tokens in the sequence
- Best for: classification, named entity recognition, question answering
- Processes the entire input at once with no autoregressive generation
- Each token can attend to ALL other tokens (past and future)

Use Cases:
- Sentiment analysis: "This movie is great" -> Positive
- Question answering: Given context + question -> Answer span
- Sentence classification: "The cat sat on the mat" -> Grammar: Correct

Architecture Overview:
Input -> Token Embedding + Positional Encoding -> 
Encoder Layers (Self-Attention + FFN) -> 
Output Representations -> Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from toy_dataset import create_dataloaders


class PositionalEncoding(nn.Module):
    """
    Positional Encoding adds position information to embeddings.
    
    Why do we need this?
    - Transformers have no inherent notion of token order (unlike RNNs)
    - Without positional info, "cat chased dog" = "dog chased cat"
    - We add a unique pattern to each position
    
    There are two main approaches:
    1. Sinusoidal (fixed, used in original Transformer paper)
    2. Learned (trainable parameters, used in BERT)
    
    Sinusoidal formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
        pos = position in sequence
        i = dimension index
        d_model = embedding dimension
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of embeddings
            max_seq_length: Maximum sequence length to pre-compute
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # Shape: (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create position indices [0, 1, 2, ..., max_seq_length-1]
        # Shape: (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create dimension indices for the sinusoidal functions
        # div_term determines the frequency of sine/cosine waves
        # Each dimension gets a different frequency
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of model state)
        # This means it will be moved to GPU with model, but not trained
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings (batch_size, seq_length, d_model)
            
        Returns:
            Embeddings with positional information added
        """
        # Add positional encoding (broadcasting handles batch dimension)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism - The core of Transformers.
    
    What is Attention?
    - A way for the model to focus on relevant parts of the input
    - Computes how much each token should "attend to" other tokens
    - Output for each token is a weighted sum of all token values
    
    Intuition:
    For "The cat sat on the mat":
    - "sat" might attend strongly to "cat" (who sat?) and "mat" (where?)
    - This creates contextual representations
    
    Multi-Head Attention:
    - Instead of one attention, we have multiple "heads"
    - Each head can learn different relationships
    - Head 1: grammatical relationships
    - Head 2: semantic relationships
    - Head 3: positional relationships
    
    Attention Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Where:
        Q (Query): "What am I looking for?"
        K (Key): "What do I contain?"
        V (Value): "What do I output?"
        d_k: Dimension of keys (for scaling)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        # We project all heads at once for efficiency
        # Shape: (d_model, d_model) for each
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
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
        Compute scaled dot-product attention.
        
        Steps:
        1. Compute attention scores: Q @ K^T
        2. Scale by sqrt(d_k) to prevent gradients from vanishing
        3. Apply mask (if provided) to ignore certain positions
        4. Apply softmax to get attention weights (sum to 1)
        5. Multiply by V to get weighted output
        
        Args:
            Q: Queries (batch_size, num_heads, seq_length, d_k)
            K: Keys (batch_size, num_heads, seq_length, d_k)
            V: Values (batch_size, num_heads, seq_length, d_k)
            mask: Attention mask (batch_size, 1, 1, seq_length)
            
        Returns:
            output: Attention output (batch_size, num_heads, seq_length, d_k)
            attention_weights: Attention probabilities
        """
        # Step 1: Compute attention scores
        # QK^T: (batch_size, num_heads, seq_length, d_k) @ (batch_size, num_heads, d_k, seq_length)
        # Result: (batch_size, num_heads, seq_length, seq_length)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Step 2: Scale by sqrt(d_k)
        # Why? Without scaling, for large d_k, dot products grow large in magnitude
        # This pushes softmax into regions with small gradients (vanishing gradient)
        scores = scores / math.sqrt(self.d_k)
        
        # Step 3: Apply mask (if provided)
        # Masked positions get -inf, so softmax makes them ~0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 4: Apply softmax to get attention weights
        # Each row sums to 1 (probability distribution)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights (regularization)
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: Multiply by values
        # Weighted sum of values based on attention
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).
        Reshape from (batch_size, seq_length, d_model) to
        (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back together.
        Reshape from (batch_size, num_heads, seq_length, d_k) to
        (batch_size, seq_length, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_length, d_model)
            key: Key tensor (batch_size, seq_length, d_model)
            value: Value tensor (batch_size, seq_length, d_model)
            mask: Attention mask (batch_size, 1, 1, seq_length)
            
        Returns:
            output: Attention output (batch_size, seq_length, d_model)
            attention_weights: For visualization/analysis
        """
        # Project to Q, K, V
        Q = self.W_q(query)  # (batch_size, seq_length, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads
        output = self.combine_heads(attn_output)  # (batch_size, seq_length, d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    After attention, each position goes through the same FFN independently.
    This adds non-linearity and increases model capacity.
    
    Architecture:
        Input -> Linear -> ReLU -> Dropout -> Linear -> Dropout
        
    Typically, the hidden dimension is 4x the model dimension.
    This expansion and compression helps the model learn complex patterns.
    
    Why position-wise?
    - Same FFN applied to each position independently
    - No interaction between positions (unlike attention)
    - Processes the attended representation further
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension (typically 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, seq_length, d_model)
            
        Returns:
            Output (batch_size, seq_length, d_model)
        """
        # First linear transformation with expansion
        x = self.linear1(x)  # (batch_size, seq_length, d_ff)
        
        # ReLU activation for non-linearity
        x = F.relu(x)
        
        # Dropout for regularization
        x = self.dropout1(x)
        
        # Second linear transformation with compression
        x = self.linear2(x)  # (batch_size, seq_length, d_model)
        
        # Final dropout
        x = self.dropout2(x)
        
        return x


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer.
    
    Architecture:
        Input
          ↓
        Multi-Head Self-Attention
          ↓
        Add & Norm (Residual Connection + Layer Normalization)
          ↓
        Feed-Forward Network
          ↓
        Add & Norm (Residual Connection + Layer Normalization)
          ↓
        Output
    
    Key Concepts:
    
    1. Residual Connection (Add):
       - Adds input directly to output: output = layer(input) + input
       - Helps with gradient flow (addresses vanishing gradients)
       - Allows the model to learn incremental changes
    
    2. Layer Normalization (Norm):
       - Normalizes across features for each sample
       - Stabilizes training
       - Formula: (x - mean) / sqrt(var + eps)
    
    3. Pre-Norm vs Post-Norm:
       - Post-Norm: LayerNorm after residual (original Transformer)
       - Pre-Norm: LayerNorm before sub-layer (better for deep models)
       - We use Pre-Norm here for training stability
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Self-attention layer
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward layer
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
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
            mask: Attention mask (batch_size, 1, 1, seq_length)
            
        Returns:
            Output (batch_size, seq_length, d_model)
        """
        # Self-Attention Block (Pre-Norm)
        # 1. Normalize input
        normed = self.norm1(x)
        
        # 2. Self-attention (Q, K, V all come from same source)
        attn_output, _ = self.self_attention(normed, normed, normed, mask)
        
        # 3. Residual connection + dropout
        x = x + self.dropout1(attn_output)
        
        # Feed-Forward Block (Pre-Norm)
        # 1. Normalize
        normed = self.norm2(x)
        
        # 2. Feed-forward
        ff_output = self.feed_forward(normed)
        
        # 3. Residual connection + dropout
        x = x + self.dropout2(ff_output)
        
        return x


class EncoderOnlyTransformer(nn.Module):
    """
    Complete Encoder-Only Transformer (BERT-style).
    
    Architecture:
        Token Embeddings + Positional Encoding
          ↓
        Encoder Layer 1
          ↓
        Encoder Layer 2
          ↓
        ...
          ↓
        Encoder Layer N
          ↓
        Classification Head / Task-Specific Layer
    
    This model can see the entire input at once (bidirectional attention).
    Perfect for understanding tasks but not for autoregressive generation.
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
        num_classes: int = 10,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
            num_classes: Number of output classes (for classification)
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Token embeddings
        # Converts token indices to dense vectors
        # Each token gets a learnable embedding vector
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        # Adds position information to embeddings
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        # For sequence classification, we use the [CLS] token or mean pooling
        # Here we'll use mean pooling: average all token representations
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights properly for stable training.
        
        Good initialization is crucial for:
        - Faster convergence
        - Avoiding vanishing/exploding gradients
        - Better final performance
        """
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def create_padding_mask(self, x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Create mask for padding tokens.
        
        Padding tokens should not be attended to.
        
        Args:
            x: Input indices (batch_size, seq_length)
            pad_idx: Index of padding token
            
        Returns:
            Mask (batch_size, 1, 1, seq_length)
        """
        # True where there's actual content, False for padding
        mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input token indices (batch_size, seq_length)
            mask: Attention mask (batch_size, 1, 1, seq_length)
            
        Returns:
            logits: Classification logits (batch_size, num_classes)
            sequence_output: Full sequence representations (batch_size, seq_length, d_model)
        """
        # Create padding mask if not provided
        if mask is None:
            mask = self.create_padding_mask(x)
        
        # Token embeddings
        # Shape: (batch_size, seq_length, d_model)
        x = self.embedding(x)
        
        # Scale embeddings (from original Transformer paper)
        # Helps balance gradients between embeddings and positional encoding
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Final normalization
        x = self.norm(x)
        
        # For classification: mean pooling over sequence
        # Shape: (batch_size, d_model)
        pooled = x.mean(dim=1)
        
        # Classification head
        # Shape: (batch_size, num_classes)
        logits = self.classifier(pooled)
        
        return logits, x


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)
        
        # For classification, we'll use first target token as label
        # In a real scenario, you'd have proper labels
        labels = tgt[:, 0]  # Just a dummy task for demonstration
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(src, src_mask.unsqueeze(1).unsqueeze(2))
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# Main training script
if __name__ == "__main__":
    print("=" * 80)
    print("ENCODER-ONLY TRANSFORMER (BERT-STYLE)")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataset...")
    train_loader, val_loader, dataset = create_dataloaders(batch_size=32, num_samples=1000)
    
    # Model hyperparameters
    vocab_size = dataset.src_vocab_size
    d_model = 128
    num_layers = 3
    num_heads = 4
    d_ff = 512
    dropout = 0.1
    num_classes = dataset.tgt_vocab_size  # Predict target vocabulary
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  Dropout: {dropout}")
    
    # Create model
    model = EncoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        num_classes=num_classes
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
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    
    print("\n" + "=" * 80)
    print("Training completed! ✓")
    print("=" * 80)
    
    # Save model
    torch.save(model.state_dict(), 'encoder_only_model.pt')
    print("\nModel saved to 'encoder_only_model.pt'")
