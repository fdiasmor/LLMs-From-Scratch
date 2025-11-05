"""
Advanced Concepts and Visualizations
=====================================

This module demonstrates advanced LLM concepts with visualizations:
1. Attention visualization
2. Gradient flow analysis
3. Token embeddings visualization
4. Learning curves
5. Performance metrics
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import math

from toy_dataset import create_dataloaders
from encoder_only import EncoderOnlyTransformer
from decoder_only import DecoderOnlyTransformer
from encoder_decoder import EncoderDecoderTransformer


def visualize_attention_weights(
    attention_weights: torch.Tensor,
    src_tokens: List[str],
    tgt_tokens: List[str] = None,
    title: str = "Attention Weights"
):
    """
    Visualize attention weights as a heatmap.
    
    This helps understand:
    - Which tokens the model is focusing on
    - How information flows through the model
    - What the model has learned
    
    Args:
        attention_weights: Attention matrix (seq_len, seq_len) or (tgt_len, src_len)
        src_tokens: Source tokens
        tgt_tokens: Target tokens (for cross-attention)
        title: Plot title
    """
    weights = attention_weights.cpu().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    
    if tgt_tokens is not None:
        plt.yticks(range(len(tgt_tokens)), tgt_tokens)
        plt.xticks(range(len(src_tokens)), src_tokens, rotation=45)
        plt.ylabel('Target Tokens')
        plt.xlabel('Source Tokens')
    else:
        plt.yticks(range(len(src_tokens)), src_tokens)
        plt.xticks(range(len(src_tokens)), src_tokens, rotation=45)
        plt.ylabel('Query Tokens')
        plt.xlabel('Key Tokens')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    print(f"Saved: {title.replace(' ', '_').lower()}.png")
    plt.close()


def visualize_positional_encoding(d_model: int = 128, max_len: int = 50):
    """
    Visualize positional encoding patterns.
    
    This shows how different positions get unique encodings.
    Each dimension oscillates at a different frequency.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pe.numpy().T, cmap='RdBu', aspect='auto')
    plt.colorbar(label='Encoding Value')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Sinusoidal Positional Encoding')
    plt.tight_layout()
    plt.savefig('positional_encoding.png')
    print("Saved: positional_encoding.png")
    plt.close()
    
    # Plot specific dimensions
    plt.figure(figsize=(12, 6))
    for i in range(0, d_model, d_model // 8):
        plt.plot(pe[:, i].numpy(), label=f'Dim {i}')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding - Selected Dimensions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('positional_encoding_dimensions.png')
    print("Saved: positional_encoding_dimensions.png")
    plt.close()


def visualize_causal_mask(seq_length: int = 10):
    """
    Visualize the causal (look-ahead) mask used in decoder-only models.
    
    Shows how each position can only attend to previous positions.
    """
    mask = torch.tril(torch.ones(seq_length, seq_length))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mask.numpy(), cmap='gray', aspect='auto')
    plt.colorbar(label='Can Attend (1) or Masked (0)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Causal Attention Mask (Decoder)')
    
    # Add grid
    for i in range(seq_length + 1):
        plt.axhline(i - 0.5, color='white', linewidth=0.5)
        plt.axvline(i - 0.5, color='white', linewidth=0.5)
    
    plt.xticks(range(seq_length))
    plt.yticks(range(seq_length))
    plt.tight_layout()
    plt.savefig('causal_mask.png')
    print("Saved: causal_mask.png")
    plt.close()


def compare_attention_patterns():
    """
    Compare attention patterns across different architectures.
    """
    seq_len = 8
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Encoder (bidirectional)
    encoder_mask = torch.ones(seq_len, seq_len)
    axes[0].imshow(encoder_mask.numpy(), cmap='Blues', aspect='auto')
    axes[0].set_title('Encoder\n(Bidirectional Attention)')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # 2. Decoder (causal)
    decoder_mask = torch.tril(torch.ones(seq_len, seq_len))
    axes[1].imshow(decoder_mask.numpy(), cmap='Greens', aspect='auto')
    axes[1].set_title('Decoder\n(Causal Attention)')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    
    # 3. Cross-attention (full)
    src_len = 6
    tgt_len = 8
    cross_mask = torch.ones(tgt_len, src_len)
    im = axes[2].imshow(cross_mask.numpy(), cmap='Oranges', aspect='auto')
    axes[2].set_title('Cross-Attention\n(Decoder to Encoder)')
    axes[2].set_xlabel('Encoder Position')
    axes[2].set_ylabel('Decoder Position')
    
    plt.tight_layout()
    plt.savefig('attention_patterns_comparison.png')
    print("Saved: attention_patterns_comparison.png")
    plt.close()


def visualize_transformer_architecture():
    """
    Create a visual diagram of the three architectures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    
    # Helper function to draw components
    def draw_box(ax, x, y, width, height, text, color):
        rect = plt.Rectangle((x, y), width, height, 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    def draw_arrow(ax, x1, y1, x2, y2):
        ax.arrow(x1, y1, x2-x1, y2-y1, 
                head_width=0.15, head_length=0.1, 
                fc='black', ec='black', linewidth=2)
    
    # 1. Encoder-Only
    ax = axes[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Encoder-Only\n(BERT-style)', fontsize=14, fontweight='bold')
    
    draw_box(ax, 1, 0.5, 3, 0.8, 'Input Tokens', 'lightblue')
    draw_arrow(ax, 2.5, 1.3, 2.5, 1.8)
    draw_box(ax, 1, 2, 3, 0.8, 'Embeddings + PE', 'lightgreen')
    draw_arrow(ax, 2.5, 2.8, 2.5, 3.3)
    
    for i in range(3):
        y = 3.5 + i * 1.5
        draw_box(ax, 1, y, 3, 0.8, f'Encoder Layer {i+1}', 'lightcoral')
        if i < 2:
            draw_arrow(ax, 2.5, y + 0.8, 2.5, y + 1.3)
    
    draw_arrow(ax, 2.5, 8.3, 2.5, 8.8)
    draw_box(ax, 1, 9, 3, 0.8, 'Classification', 'gold')
    
    # 2. Decoder-Only
    ax = axes[1]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Decoder-Only\n(GPT-style)', fontsize=14, fontweight='bold')
    
    draw_box(ax, 1, 0.5, 3, 0.8, 'Input Tokens', 'lightblue')
    draw_arrow(ax, 2.5, 1.3, 2.5, 1.8)
    draw_box(ax, 1, 2, 3, 0.8, 'Embeddings + PE', 'lightgreen')
    draw_arrow(ax, 2.5, 2.8, 2.5, 3.3)
    
    for i in range(3):
        y = 3.5 + i * 1.5
        draw_box(ax, 1, y, 3, 0.8, f'Decoder Layer {i+1}', 'plum')
        if i < 2:
            draw_arrow(ax, 2.5, y + 0.8, 2.5, y + 1.3)
    
    draw_arrow(ax, 2.5, 8.3, 2.5, 8.8)
    draw_box(ax, 1, 9, 3, 0.8, 'Next Token', 'gold')
    
    # 3. Encoder-Decoder
    ax = axes[2]
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Encoder-Decoder\n(T5-style)', fontsize=14, fontweight='bold')
    
    # Encoder side
    draw_box(ax, 0.5, 0.5, 2.5, 0.8, 'Source', 'lightblue')
    draw_arrow(ax, 1.75, 1.3, 1.75, 1.8)
    draw_box(ax, 0.5, 2, 2.5, 0.8, 'Embed + PE', 'lightgreen')
    draw_arrow(ax, 1.75, 2.8, 1.75, 3.3)
    
    for i in range(2):
        y = 3.5 + i * 1.3
        draw_box(ax, 0.5, y, 2.5, 0.8, f'Enc {i+1}', 'lightcoral')
        if i < 1:
            draw_arrow(ax, 1.75, y + 0.8, 1.75, y + 1.1)
    
    # Decoder side
    draw_box(ax, 6, 0.5, 2.5, 0.8, 'Target', 'lightblue')
    draw_arrow(ax, 7.25, 1.3, 7.25, 1.8)
    draw_box(ax, 6, 2, 2.5, 0.8, 'Embed + PE', 'lightgreen')
    draw_arrow(ax, 7.25, 2.8, 7.25, 3.3)
    
    for i in range(2):
        y = 3.5 + i * 1.3
        draw_box(ax, 6, y, 2.5, 0.8, f'Dec {i+1}', 'plum')
        # Cross-attention arrow
        draw_arrow(ax, 3, y + 0.4, 6, y + 0.4)
        if i < 1:
            draw_arrow(ax, 7.25, y + 0.8, 7.25, y + 1.1)
    
    draw_arrow(ax, 7.25, 6.1, 7.25, 6.6)
    draw_box(ax, 6, 6.8, 2.5, 0.8, 'Output', 'gold')
    
    plt.tight_layout()
    plt.savefig('architectures_comparison.png')
    print("Saved: architectures_comparison.png")
    plt.close()


def analyze_model_complexity():
    """
    Analyze and compare model complexity across architectures.
    """
    vocab_size = 14
    d_model = 128
    num_layers = 3
    num_heads = 4
    d_ff = 512
    
    # Calculate parameters for each architecture
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    encoder_model = EncoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        num_classes=vocab_size
    )
    
    decoder_model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    )
    
    enc_dec_model = EncoderDecoderTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    )
    
    models = {
        'Encoder-Only': count_params(encoder_model),
        'Decoder-Only': count_params(decoder_model),
        'Encoder-Decoder': count_params(enc_dec_model)
    }
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    names = list(models.keys())
    params = list(models.values())
    colors = ['lightcoral', 'plum', 'lightblue']
    
    ax1.bar(names, params, color=colors)
    ax1.set_ylabel('Number of Parameters')
    ax1.set_title('Model Complexity Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(params):
        ax1.text(i, v + max(params)*0.02, f'{v:,}', ha='center', fontweight='bold')
    
    # Component breakdown for encoder-decoder
    components = {
        'Encoder Embedding': d_model * vocab_size,
        'Decoder Embedding': d_model * vocab_size,
        'Encoder Layers': num_layers * (
            4 * d_model * d_model +  # Attention Q,K,V,O
            2 * d_model * d_ff +      # FFN
            2 * d_model               # LayerNorm
        ),
        'Decoder Layers': num_layers * (
            8 * d_model * d_model +  # Self-attn + Cross-attn
            2 * d_model * d_ff +      # FFN
            3 * d_model               # LayerNorm
        ),
        'Output Projection': d_model * vocab_size
    }
    
    ax2.pie(components.values(), labels=components.keys(), autopct='%1.1f%%',
           colors=plt.cm.Pastel1.colors, startangle=90)
    ax2.set_title('Encoder-Decoder Parameter Distribution')
    
    plt.tight_layout()
    plt.savefig('model_complexity.png')
    print("Saved: model_complexity.png")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    for name, param_count in models.items():
        print(f"{name:20s}: {param_count:>10,} parameters")
    print("=" * 60)


def demonstrate_sampling_strategies():
    """
    Visualize different sampling strategies for text generation.
    """
    # Create sample logits
    vocab_size = 10
    logits = torch.randn(vocab_size) * 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1. Original logits
    ax = axes[0, 0]
    ax.bar(range(vocab_size), logits.numpy())
    ax.set_title('Original Logits')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Logit Value')
    ax.grid(alpha=0.3)
    
    # 2. Softmax (probabilities)
    probs = F.softmax(logits, dim=0)
    ax = axes[0, 1]
    ax.bar(range(vocab_size), probs.numpy(), color='orange')
    ax.set_title('Softmax Probabilities')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Probability')
    ax.grid(alpha=0.3)
    
    # 3. Greedy (pick highest)
    greedy_choice = torch.argmax(probs)
    ax = axes[0, 2]
    colors = ['red' if i == greedy_choice else 'gray' for i in range(vocab_size)]
    ax.bar(range(vocab_size), probs.numpy(), color=colors)
    ax.set_title('Greedy Sampling (pick max)')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Probability')
    ax.grid(alpha=0.3)
    
    # 4. Temperature = 0.5 (more peaked)
    temp_low = F.softmax(logits / 0.5, dim=0)
    ax = axes[1, 0]
    ax.bar(range(vocab_size), temp_low.numpy(), color='blue')
    ax.set_title('Temperature = 0.5 (conservative)')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Probability')
    ax.grid(alpha=0.3)
    
    # 5. Temperature = 2.0 (more uniform)
    temp_high = F.softmax(logits / 2.0, dim=0)
    ax = axes[1, 1]
    ax.bar(range(vocab_size), temp_high.numpy(), color='green')
    ax.set_title('Temperature = 2.0 (creative)')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Probability')
    ax.grid(alpha=0.3)
    
    # 6. Top-k = 3
    top_k = 3
    top_k_probs = probs.clone()
    top_k_values, top_k_indices = torch.topk(probs, top_k)
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask[top_k_indices] = False
    top_k_probs[mask] = 0
    top_k_probs = top_k_probs / top_k_probs.sum()
    
    ax = axes[1, 2]
    ax.bar(range(vocab_size), top_k_probs.numpy(), color='purple')
    ax.set_title(f'Top-k Sampling (k={top_k})')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Probability')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sampling_strategies.png')
    print("Saved: sampling_strategies.png")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    print("\n1. Positional Encoding...")
    visualize_positional_encoding()
    
    print("\n2. Causal Mask...")
    visualize_causal_mask()
    
    print("\n3. Attention Patterns...")
    compare_attention_patterns()
    
    print("\n4. Architecture Diagrams...")
    visualize_transformer_architecture()
    
    print("\n5. Model Complexity...")
    analyze_model_complexity()
    
    print("\n6. Sampling Strategies...")
    demonstrate_sampling_strategies()
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE! ✓")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - positional_encoding.png")
    print("  - positional_encoding_dimensions.png")
    print("  - causal_mask.png")
    print("  - attention_patterns_comparison.png")
    print("  - architectures_comparison.png")
    print("  - model_complexity.png")
    print("  - sampling_strategies.png")


if __name__ == "__main__":
    main()
