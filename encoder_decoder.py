"""
Encoder-Decoder Transformer (T5/BART-style Architecture)
==========================================================

This module implements an encoder-decoder transformer, similar to T5, BART, or original Transformer.

Key Characteristics:
- Encoder: Uses bidirectional attention (sees full input)
- Decoder: Uses causal attention + cross-attention to encoder
- Best for: translation, summarization, question answering
- Maps one sequence to another sequence

Use Cases:
- Machine translation: "Hello" (English) -> "Bonjour" (French)
- Summarization: Long article -> Short summary
- Question answering: Context + Question -> Answer
- Text-to-SQL: Natural language -> SQL query

Architecture Overview:
Input Sequence -> Encoder (bidirectional) -> Hidden States
                                               ↓
Target Sequence -> Decoder (causal + cross-attention) -> Output

Key Innovation: Cross-Attention
- Decoder attends to encoder's output
- Allows decoder to "look at" the input while generating
- This is how translation models align source and target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from toy_dataset import create_dataloaders


class PositionalEncoding(nn.Module):
    """Positional Encoding - same as previous implementations."""
    
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
    """Multi-Head Attention - same as previous implementations."""
    
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
    """Position-wise Feed-Forward Network."""
    
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


class EncoderLayer(nn.Module):
    """
    Encoder Layer with bidirectional self-attention.
    
    Same as encoder-only model.
    Can see all input tokens at once.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-Attention Block
        normed = self.norm1(x)
        attn_output, _ = self.self_attention(normed, normed, normed, mask)
        x = x + self.dropout1(attn_output)
        
        # Feed-Forward Block
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout2(ff_output)
        
        return x


class DecoderLayer(nn.Module):
    """
    Decoder Layer with:
    1. Masked self-attention (causal)
    2. Cross-attention to encoder output (THE KEY INNOVATION!)
    3. Feed-forward network
    
    Architecture:
        Input
          ↓
        Masked Self-Attention (look at previous decoder tokens)
          ↓
        Add & Norm
          ↓
        Cross-Attention (look at encoder output) ← NEW!
          ↓
        Add & Norm
          ↓
        Feed-Forward
          ↓
        Add & Norm
          ↓
        Output
    
    Cross-Attention Explained:
    - Query (Q): Comes from decoder (what decoder is looking for)
    - Key (K), Value (V): Come from encoder (what input contains)
    - This allows decoder to attend to relevant parts of input
    
    Example (Translation):
        Input: "The cat sits"
        Decoder generating: "Le chat"
        
        When generating "chat":
        - Decoder self-attention: looks at "Le"
        - Cross-attention: looks at "The cat sits" and focuses on "cat"
        - This alignment is learned automatically!
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Masked self-attention (decoder looks at itself)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Cross-attention (decoder looks at encoder)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch_size, tgt_seq_length, d_model)
            encoder_output: Encoder output (batch_size, src_seq_length, d_model)
            src_mask: Source padding mask (batch_size, 1, 1, src_seq_length)
            tgt_mask: Target causal + padding mask (batch_size, 1, tgt_seq_length, tgt_seq_length)
            
        Returns:
            Output (batch_size, tgt_seq_length, d_model)
        """
        # 1. Masked Self-Attention Block
        # Decoder attends to previous decoder positions
        normed = self.norm1(x)
        attn_output, _ = self.self_attention(normed, normed, normed, tgt_mask)
        x = x + self.dropout1(attn_output)
        
        # 2. Cross-Attention Block ← THE MAGIC HAPPENS HERE!
        # Query from decoder, Key and Value from encoder
        normed = self.norm2(x)
        # Q: from decoder (what we're looking for)
        # K, V: from encoder (what the input contains)
        cross_attn_output, _ = self.cross_attention(
            query=normed,           # From decoder
            key=encoder_output,     # From encoder
            value=encoder_output,   # From encoder
            mask=src_mask
        )
        x = x + self.dropout2(cross_attn_output)
        
        # 3. Feed-Forward Block
        normed = self.norm3(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout3(ff_output)
        
        return x


class EncoderDecoderTransformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer.
    
    Full Architecture:
    
        SOURCE SEQUENCE                    TARGET SEQUENCE
             ↓                                    ↓
        Token Embed + Pos Enc            Token Embed + Pos Enc
             ↓                                    ↓
        Encoder Layer 1                   Decoder Layer 1
             ↓                            ↙      ↓       (cross-attn)
        Encoder Layer 2                   Decoder Layer 2
             ↓                            ↙      ↓
        ...                               ...
             ↓                            ↙      ↓
        Encoder Layer N                   Decoder Layer N
             ↓                            ↙      ↓
        Encoder Output ───────────────────      Output Projection
                                                 ↓
                                            Next Token Logits
    
    Training:
    - Input: Source sequence
    - Target: Target sequence (shifted by 1)
    - Loss: Cross-entropy for next token prediction
    - Teacher forcing: Use ground truth previous tokens during training
    
    Inference:
    - Encode source once
    - Decode autoregressively (one token at a time)
    - Use previously generated tokens as decoder input
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 512,
        max_seq_length: int = 100,
        dropout: float = 0.1,
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_layers: Number of encoder/decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Encoder components
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.encoder_pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder components
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layers
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.encoder_embedding.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.decoder_embedding.weight, mean=0, std=self.d_model ** -0.5)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def create_padding_mask(self, x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Create padding mask.
        
        Args:
            x: Token indices (batch_size, seq_length)
            pad_idx: Padding token index
            
        Returns:
            Mask (batch_size, 1, 1, seq_length)
        """
        mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for decoder.
        
        Args:
            seq_length: Sequence length
            device: Device
            
        Returns:
            Causal mask (1, 1, seq_length, seq_length)
        """
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask
    
    def create_target_mask(self, tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Create combined target mask (padding + causal).
        
        Args:
            tgt: Target token indices (batch_size, seq_length)
            pad_idx: Padding token index
            
        Returns:
            Combined mask (batch_size, 1, seq_length, seq_length)
        """
        batch_size, seq_length = tgt.size()
        device = tgt.device

        # Padding mask: (B, 1, 1, S), bool
        padding_mask = self.create_padding_mask(tgt, pad_idx).to(torch.bool)

        # Causal mask: (1, 1, S, S), bool
        causal_mask = self.create_causal_mask(seq_length, device).to(torch.bool)

        # Combine with broadcasting: both must be True to allow attention
        # Result: (B, 1, S, S), bool
        combined_mask = padding_mask & causal_mask
        # or: combined_mask = torch.logical_and(padding_mask, causal_mask)
 
        # We want 0 where combined_mask==True, and -inf where combined_mask==False
        additive_mask = torch.where(combined_mask == True, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

        return additive_mask
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token indices (batch_size, src_seq_length)
            src_mask: Source padding mask (batch_size, 1, 1, src_seq_length)
            
        Returns:
            Encoder output (batch_size, src_seq_length, d_model)
        """
        # Create mask if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        
        # Embeddings + positional encoding
        x = self.encoder_embedding(src)
        x = x * math.sqrt(self.d_model)
        x = self.encoder_pos_encoding(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        
        # Final normalization
        x = self.encoder_norm(x)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target token indices (batch_size, tgt_seq_length)
            encoder_output: Encoder output (batch_size, src_seq_length, d_model)
            src_mask: Source padding mask (batch_size, 1, 1, src_seq_length)
            tgt_mask: Target mask (batch_size, 1, tgt_seq_length, tgt_seq_length)
            
        Returns:
            Decoder output (batch_size, tgt_seq_length, d_model)
        """
        # Create target mask if not provided
        if tgt_mask is None:
            tgt_mask = self.create_target_mask(tgt)
        
        # Embeddings + positional encoding
        x = self.decoder_embedding(tgt)
        x = x * math.sqrt(self.d_model)
        x = self.decoder_pos_encoding(x)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, src_mask, tgt_mask)
        
        # Final normalization
        x = self.decoder_norm(x)
        
        return x
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: Source token indices (batch_size, src_seq_length)
            tgt: Target token indices (batch_size, tgt_seq_length)
            src_mask: Source mask (batch_size, 1, 1, src_seq_length)
            tgt_mask: Target mask (batch_size, 1, tgt_seq_length, tgt_seq_length)
            
        Returns:
            Logits (batch_size, tgt_seq_length, tgt_vocab_size)
        """
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate target sequence using greedy/sampling decoding.
        
        Process:
        1. Encode source once (efficient - don't need to re-encode)
        2. Start with <SOS> token
        3. Generate one token at a time
        4. Add generated token to decoder input
        5. Repeat until <EOS> or max_length
        
        Args:
            src: Source sequence (batch_size, src_seq_length)
            max_length: Maximum generation length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated sequence (batch_size, generated_length)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source once
        src_mask = self.create_padding_mask(src)
        encoder_output = self.encode(src, src_mask)
        
        # Start with <SOS> token
        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Decode current sequence
            tgt_mask = self.create_target_mask(generated)
            decoder_output = self.decode(generated, encoder_output, src_mask, tgt_mask)
            
            # Get logits for last position
            logits = self.output_projection(decoder_output[:, -1, :])
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k if specified
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                logits = logits_filtered
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == eos_idx).all():
                break
        
        return generated
    
    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        beam_width: int = 5,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
        length_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate using beam search for better quality.
        
        Beam Search:
        - Instead of greedy (picking best token), keep top-k candidates
        - At each step, expand each candidate and keep best k overall
        - More computational but usually better results
        
        Example with beam_width=2:
            Step 0: ["<SOS>"]
            Step 1: ["<SOS> the", "<SOS> a"]  (keep top 2)
            Step 2: ["<SOS> the cat", "<SOS> the dog", "<SOS> a dog", "<SOS> a cat"]
                    ↓ keep top 2
                    ["<SOS> the cat", "<SOS> a dog"]
            ...
        
        Length Penalty:
        - Longer sequences have lower total probability
        - Apply penalty: score = log_prob / (length ^ length_penalty)
        - length_penalty > 1: favor longer sequences
        - length_penalty < 1: favor shorter sequences
        
        Args:
            src: Source sequence (1, src_seq_length) - only supports batch_size=1
            beam_width: Number of beams
            max_length: Maximum generation length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            length_penalty: Length normalization penalty
            
        Returns:
            best_sequence: Best generated sequence (1, length)
            best_score: Score of best sequence
        """
        self.eval()
        device = src.device
        
        # Encode source
        src_mask = self.create_padding_mask(src)
        encoder_output = self.encode(src, src_mask)
        
        # Expand encoder output for beam search
        # Shape: (beam_width, src_seq_length, d_model)
        encoder_output = encoder_output.expand(beam_width, -1, -1)
        src_mask = src_mask.expand(beam_width, -1, -1, -1)
        
        # Initialize beams: (beam_width, 1)
        beams = torch.full((beam_width, 1), sos_idx, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_width, device=device)
        beam_scores[1:] = float('-inf')  # Only first beam is active initially
        
        # Track completed sequences
        completed_sequences = []
        completed_scores = []
        
        for step in range(max_length):
            # Decode all beams
            tgt_mask = self.create_target_mask(beams)
            decoder_output = self.decode(beams, encoder_output, src_mask, tgt_mask)
            
            # Get logits for last position
            logits = self.output_projection(decoder_output[:, -1, :])
            
            # Log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute scores for all possible next tokens
            # Shape: (beam_width, vocab_size)
            vocab_size = log_probs.size(-1)
            scores = beam_scores.unsqueeze(1) + log_probs
            
            # Reshape to (beam_width * vocab_size)
            scores = scores.view(-1)
            
            # Get top beam_width scores
            top_scores, top_indices = torch.topk(scores, beam_width)
            
            # Convert flat indices back to (beam_idx, token_idx)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Create new beams
            new_beams = []
            new_scores = []
            
            for i, (beam_idx, token_idx, score) in enumerate(zip(beam_indices, token_indices, top_scores)):
                # Get previous beam sequence
                prev_beam = beams[beam_idx]
                
                # Append new token
                new_beam = torch.cat([prev_beam, token_idx.unsqueeze(0)])
                
                # Check if EOS
                if token_idx.item() == eos_idx:
                    # Apply length penalty
                    length = new_beam.size(0)
                    normalized_score = score / (length ** length_penalty)
                    completed_sequences.append(new_beam)
                    completed_scores.append(normalized_score)
                else:
                    new_beams.append(new_beam)
                    new_scores.append(score)
            
            # If no active beams, stop
            if len(new_beams) == 0:
                break
            
            # Pad beams to same length
            max_len = max(beam.size(0) for beam in new_beams)
            beams = torch.stack([
                F.pad(beam, (0, max_len - beam.size(0)), value=0)
                for beam in new_beams
            ])
            beam_scores = torch.tensor(new_scores, device=device)
            
            # Keep only top beam_width beams
            if len(new_beams) > beam_width:
                top_beam_indices = torch.topk(beam_scores, beam_width)[1]
                beams = beams[top_beam_indices]
                beam_scores = beam_scores[top_beam_indices]
        
        # Add remaining beams to completed
        for beam, score in zip(beams, beam_scores):
            length = beam.size(0)
            normalized_score = score / (length ** length_penalty)
            completed_sequences.append(beam)
            completed_scores.append(normalized_score)
        
        # Get best sequence
        if len(completed_sequences) > 0:
            best_idx = torch.tensor(completed_scores).argmax()
            best_sequence = completed_sequences[best_idx].unsqueeze(0)
            best_score = completed_scores[best_idx]
        else:
            # Fallback: return first beam
            best_sequence = beams[0].unsqueeze(0)
            best_score = beam_scores[0]
        
        return best_sequence, best_score


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Training uses "teacher forcing":
    - We provide the correct previous tokens to the decoder
    - Even if model would have predicted wrong token
    - This speeds up training significantly
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        src = batch['src'].to(device)
        tgt_input = batch['decoder_input'].to(device)
        tgt_output = batch['decoder_output'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(src, tgt_input)
        
        # Compute loss
        # Reshape: (batch_size * seq_length, vocab_size) vs (batch_size * seq_length)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
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
    print("ENCODER-DECODER TRANSFORMER (T5/BART-STYLE)")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\nCreating dataset...")
    train_loader, val_loader, dataset = create_dataloaders(batch_size=32, num_samples=1000)
    
    # Model hyperparameters
    src_vocab_size = dataset.src_vocab_size
    tgt_vocab_size = dataset.tgt_vocab_size
    d_model = 128
    num_layers = 3
    num_heads = 4
    d_ff = 512
    dropout = 0.1
    
    print(f"\nModel Configuration:")
    print(f"  Source vocabulary size: {src_vocab_size}")
    print(f"  Target vocabulary size: {tgt_vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  Dropout: {dropout}")
    
    # Create model
    model = EncoderDecoderTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
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
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("INFERENCE DEMO")
    print("=" * 80)
    
    # Demo inference
    model.eval()
    
    # Get a sample
    sample = dataset[0]
    src_tokens = sample['src_tokens']
    tgt_tokens = sample['tgt_tokens']
    
    print(f"\nSource: {' '.join(src_tokens)}")
    print(f"Target: {' '.join(tgt_tokens)}")
    
    # Prepare input
    src = sample['src'].unsqueeze(0).to(device)
    
    # Greedy decoding
    print("\n1. Greedy Decoding:")
    generated = model.generate(src, max_length=10, sos_idx=dataset.sos_idx, eos_idx=dataset.eos_idx, temperature=0.1)
    generated_tokens = [dataset.tgt_idx2token.get(idx.item(), '<UNK>') for idx in generated[0]]
    print(f"   Generated: {' '.join(generated_tokens)}")
    
    # Sampling
    print("\n2. Sampling (temperature=1.0):")
    generated = model.generate(src, max_length=10, sos_idx=dataset.sos_idx, eos_idx=dataset.eos_idx, temperature=1.0)
    generated_tokens = [dataset.tgt_idx2token.get(idx.item(), '<UNK>') for idx in generated[0]]
    print(f"   Generated: {' '.join(generated_tokens)}")
    
    # Beam search
    print("\n3. Beam Search (beam_width=5):")
    best_seq, score = model.beam_search(src, beam_width=5, max_length=10, sos_idx=dataset.sos_idx, eos_idx=dataset.eos_idx)
    generated_tokens = [dataset.tgt_idx2token.get(idx.item(), '<UNK>') for idx in best_seq[0]]
    print(f"   Generated: {' '.join(generated_tokens)}")
    print(f"   Score: {score:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed! ✓")
    print("=" * 80)
    
    # Save model
    torch.save(model.state_dict(), 'encoder_decoder_model.pt')
    print("\nModel saved to 'encoder_decoder_model.pt'")
