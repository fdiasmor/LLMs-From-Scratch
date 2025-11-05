"""
Architecture Comparison Script
===============================

This script demonstrates the key differences between the three architectures
with side-by-side examples and explanations.
"""

import torch
import torch.nn.functional as F
from toy_dataset import create_dataloaders


def demonstrate_attention_patterns():
    """Show how attention patterns differ across architectures."""
    
    print("=" * 80)
    print("ATTENTION PATTERN COMPARISON")
    print("=" * 80)
    
    seq_length = 5
    
    print("\nConsider a sequence: ['The', 'cat', 'sat', 'on', 'mat']")
    print("Position indices:      [ 0  ,  1  ,  2  ,  3  ,  4  ]")
    
    print("\n" + "-" * 80)
    print("1. ENCODER-ONLY (Bidirectional Attention)")
    print("-" * 80)
    
    print("\nAttention mask: Each token can attend to ALL tokens")
    encoder_mask = torch.ones(seq_length, seq_length)
    print(encoder_mask.int())
    
    print("\nExamples:")
    print("  - Token 'cat' (pos 1) can attend to:")
    print("    ['The', 'cat', 'sat', 'on', 'mat'] - ALL positions")
    print("  - Token 'sat' (pos 2) can attend to:")
    print("    ['The', 'cat', 'sat', 'on', 'mat'] - ALL positions")
    
    print("\n✓ Use Case: Understanding full context")
    print("  Example: 'The cat sat on the mat' - what is the sentiment?")
    print("  Model needs to see ENTIRE sentence to understand")
    
    print("\n" + "-" * 80)
    print("2. DECODER-ONLY (Causal Attention)")
    print("-" * 80)
    
    print("\nAttention mask: Each token can attend to PAST tokens only")
    decoder_mask = torch.tril(torch.ones(seq_length, seq_length))
    print(decoder_mask.int())
    
    print("\nExamples:")
    print("  - Token 'cat' (pos 1) can attend to:")
    print("    ['The', 'cat'] - Only positions 0, 1")
    print("  - Token 'sat' (pos 2) can attend to:")
    print("    ['The', 'cat', 'sat'] - Only positions 0, 1, 2")
    print("  - Token 'on' (pos 3) can attend to:")
    print("    ['The', 'cat', 'sat', 'on'] - Only positions 0, 1, 2, 3")
    
    print("\n✓ Use Case: Text generation")
    print("  Prompt: 'The cat sat'")
    print("  Generate: 'on' (can only see 'The cat sat')")
    print("  Generate: 'the' (can only see 'The cat sat on')")
    print("  Generate: 'mat' (can only see 'The cat sat on the')")
    
    print("\n" + "-" * 80)
    print("3. ENCODER-DECODER (Bidirectional + Causal + Cross-Attention)")
    print("-" * 80)
    
    src_len = 4
    tgt_len = 4
    
    print("\nSource: ['Hello', 'world', '!']")
    print("Target: ['Bonjour', 'le', 'monde', '!']")
    
    print("\na) Encoder Self-Attention (bidirectional):")
    print("   Each source token attends to ALL source tokens")
    encoder_self_mask = torch.ones(src_len, src_len)
    print(encoder_self_mask.int())
    
    print("\nb) Decoder Self-Attention (causal):")
    print("   Each target token attends to PAST target tokens")
    decoder_self_mask = torch.tril(torch.ones(tgt_len, tgt_len))
    print(decoder_self_mask.int())
    
    print("\nc) Cross-Attention (full):")
    print("   Each target token attends to ALL source tokens")
    cross_mask = torch.ones(tgt_len, src_len)
    print(cross_mask.int())
    
    print("\nExample - Generating 'monde':")
    print("  1. Decoder self-attention:")
    print("     'monde' looks at: ['Bonjour', 'le', 'monde']")
    print("  2. Cross-attention:")
    print("     'monde' looks at: ['Hello', 'world', '!']")
    print("     ↑ Learns to focus on 'world'!")
    
    print("\n✓ Use Case: Translation, summarization")
    print("  Input (source): 'Hello world'")
    print("  Output (target): 'Bonjour le monde'")
    print("  Decoder can see FULL input but only PAST output")


def demonstrate_training_objectives():
    """Show how training differs across architectures."""
    
    print("\n" + "=" * 80)
    print("TRAINING OBJECTIVE COMPARISON")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("1. ENCODER-ONLY: Masked Language Modeling (MLM)")
    print("-" * 80)
    
    print("\nOriginal:  The [MASK] sat on the mat")
    print("Task:      Predict 'cat' at [MASK] position")
    print("Loss:      Cross-entropy between prediction and 'cat'")
    
    print("\nTraining Example:")
    print("  Input:   ['The', '[MASK]', 'sat', 'on', 'the', 'mat']")
    print("  Target:  Predict token at position 1")
    print("  Output:  Probability distribution over vocabulary")
    print("  Loss:    -log P(cat | context)")
    
    print("\nKey Point: Model sees ENTIRE context (bidirectional)")
    
    print("\n" + "-" * 80)
    print("2. DECODER-ONLY: Causal Language Modeling (CLM)")
    print("-" * 80)
    
    print("\nSequence:  The cat sat on the mat")
    print("Task:      Predict NEXT token at each position")
    
    print("\nTraining Example:")
    print("  Position 0: Given ''          -> Predict 'The'")
    print("  Position 1: Given 'The'       -> Predict 'cat'")
    print("  Position 2: Given 'The cat'   -> Predict 'sat'")
    print("  Position 3: Given 'The cat sat' -> Predict 'on'")
    print("  Position 4: Given 'The cat sat on' -> Predict 'the'")
    print("  Position 5: Given 'The cat sat on the' -> Predict 'mat'")
    
    print("\nLoss: Sum of -log P(token_i | tokens_<i) for all positions")
    
    print("\nKey Point: Model only sees PAST context (causal)")
    
    print("\n" + "-" * 80)
    print("3. ENCODER-DECODER: Sequence-to-Sequence")
    print("-" * 80)
    
    print("\nSource:    'Hello world'")
    print("Target:    'Bonjour le monde'")
    print("Task:      Generate target given source")
    
    print("\nTraining Example (with teacher forcing):")
    print("  Encoder input:  ['Hello', 'world']")
    print("  Decoder step 1: Given '<SOS>'          -> Predict 'Bonjour'")
    print("  Decoder step 2: Given '<SOS> Bonjour'  -> Predict 'le'")
    print("  Decoder step 3: Given '<SOS> Bonjour le' -> Predict 'monde'")
    print("  Decoder step 4: Given '<SOS> Bonjour le monde' -> Predict '<EOS>'")
    
    print("\nLoss: Sum of -log P(target_i | source, targets_<i)")
    
    print("\nKey Points:")
    print("  - Encoder sees FULL source (bidirectional)")
    print("  - Decoder sees PAST target + FULL source (causal + cross-attention)")


def demonstrate_inference():
    """Show how inference differs across architectures."""
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPARISON")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("1. ENCODER-ONLY: Single Forward Pass")
    print("-" * 80)
    
    print("\nInput: 'This movie is great!'")
    print("\nSteps:")
    print("  1. Encode entire sequence at once")
    print("  2. Pass through encoder layers")
    print("  3. Pool representations (e.g., [CLS] token or mean)")
    print("  4. Classification head")
    print("  5. Output: Positive/Negative")
    
    print("\nComplexity: O(1) - Single pass, very fast!")
    print("Time: ~5ms for typical sequence")
    
    print("\n" + "-" * 80)
    print("2. DECODER-ONLY: Autoregressive Generation")
    print("-" * 80)
    
    print("\nPrompt: 'Once upon a'")
    print("\nSteps:")
    print("  Step 1:")
    print("    Input:  ['Once', 'upon', 'a']")
    print("    Output: 'time' (most likely next token)")
    print("  ")
    print("  Step 2:")
    print("    Input:  ['Once', 'upon', 'a', 'time']")
    print("    Output: 'there'")
    print("  ")
    print("  Step 3:")
    print("    Input:  ['Once', 'upon', 'a', 'time', 'there']")
    print("    Output: 'was'")
    print("  ")
    print("  ... continue until <EOS> or max length")
    
    print("\nComplexity: O(n) - One token per step")
    print("Time: ~50ms per token × 20 tokens = ~1 second")
    
    print("\n" + "-" * 80)
    print("3. ENCODER-DECODER: Encode Once, Decode Autoregressively")
    print("-" * 80)
    
    print("\nSource: 'Hello world'")
    print("\nSteps:")
    print("  Encoding Phase (once):")
    print("    Input:  ['Hello', 'world']")
    print("    Output: Encoder hidden states [h1, h2]")
    print("  ")
    print("  Decoding Phase (autoregressive):")
    print("    Step 1:")
    print("      Decoder input: ['<SOS>']")
    print("      Cross-attend to: [h1, h2]")
    print("      Output: 'Bonjour'")
    print("    ")
    print("    Step 2:")
    print("      Decoder input: ['<SOS>', 'Bonjour']")
    print("      Cross-attend to: [h1, h2]")
    print("      Output: 'le'")
    print("    ")
    print("    Step 3:")
    print("      Decoder input: ['<SOS>', 'Bonjour', 'le']")
    print("      Cross-attend to: [h1, h2]")
    print("      Output: 'monde'")
    print("    ")
    print("    ... continue until <EOS>")
    
    print("\nComplexity: O(1) encode + O(m) decode")
    print("Time: ~10ms encode + ~50ms per token × 15 tokens = ~760ms")


def demonstrate_use_cases():
    """Show practical use cases for each architecture."""
    
    print("\n" + "=" * 80)
    print("PRACTICAL USE CASES")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("1. ENCODER-ONLY (BERT, RoBERTa, ALBERT)")
    print("-" * 80)
    
    print("\n✓ Text Classification")
    print("  Input:  'This product is amazing!'")
    print("  Output: Positive (with confidence score)")
    
    print("\n✓ Named Entity Recognition")
    print("  Input:  'Apple announced iPhone in California'")
    print("  Output: Apple[ORG], iPhone[PRODUCT], California[LOC]")
    
    print("\n✓ Question Answering (Extractive)")
    print("  Context: 'Paris is the capital of France.'")
    print("  Question: 'What is the capital of France?'")
    print("  Output: 'Paris' (extracted span)")
    
    print("\n✓ Sentence Similarity")
    print("  Sentence 1: 'The cat sits on the mat'")
    print("  Sentence 2: 'A feline rests on a rug'")
    print("  Output: 0.85 similarity score")
    
    print("\n✗ NOT suitable for:")
    print("  - Text generation (not trained for it)")
    print("  - Translation (no decoder)")
    
    print("\n" + "-" * 80)
    print("2. DECODER-ONLY (GPT-2, GPT-3, GPT-4)")
    print("-" * 80)
    
    print("\n✓ Text Completion")
    print("  Input:  'def fibonacci(n):'")
    print("  Output: '\\n    if n <= 1:\\n        return n'")
    
    print("\n✓ Creative Writing")
    print("  Input:  'Once upon a time in a distant galaxy'")
    print("  Output: 'there lived a brave astronaut named Alex...'")
    
    print("\n✓ Chat (with proper prompting)")
    print("  Input:  'Human: What is the capital of France?\\nAI:'")
    print("  Output: ' The capital of France is Paris.'")
    
    print("\n✓ Few-Shot Learning")
    print("  Input:  'Translate English to French:\\n'")
    print("          'Hello -> Bonjour\\n'")
    print("          'Goodbye -> Au revoir\\n'")
    print("          'Thank you ->'")
    print("  Output: ' Merci'")
    
    print("\n✓ Code Generation")
    print("  Input:  '# Function to sort a list'")
    print("  Output: '\\ndef sort_list(lst):\\n    return sorted(lst)'")
    
    print("\n✗ NOT optimal for:")
    print("  - Classification (can work but less efficient)")
    print("  - Tasks requiring bidirectional context")
    
    print("\n" + "-" * 80)
    print("3. ENCODER-DECODER (T5, BART, mT5)")
    print("-" * 80)
    
    print("\n✓ Machine Translation")
    print("  Input:  'Hello, how are you?'")
    print("  Output: 'Bonjour, comment allez-vous?'")
    
    print("\n✓ Text Summarization")
    print("  Input:  'Long article about climate change... (500 words)'")
    print("  Output: 'Climate change causes rising temperatures and...'")
    
    print("\n✓ Question Answering (Generative)")
    print("  Input:  'Context: ... Question: What is the main idea?'")
    print("  Output: 'The main idea is that...' (generated)")
    
    print("\n✓ Text-to-SQL")
    print("  Input:  'Show me all users who signed up last month'")
    print("  Output: 'SELECT * FROM users WHERE signup_date >= ...'")
    
    print("\n✓ Paraphrasing")
    print("  Input:  'The cat sat on the mat'")
    print("  Output: 'A feline was resting on a small rug'")
    
    print("\n✗ NOT optimal for:")
    print("  - Open-ended generation (decoder-only is better)")
    print("  - Simple classification (encoder-only is faster)")


def demonstrate_parameter_sharing():
    """Show parameter differences."""
    
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON (for d_model=128, num_layers=3)")
    print("=" * 80)
    
    vocab = 14
    d_model = 128
    num_layers = 3
    d_ff = 512
    
    # Encoder-only
    enc_params = (
        vocab * d_model +           # Embedding
        num_layers * (
            4 * d_model * d_model + # Attention Q,K,V,O
            2 * d_model * d_ff +    # FFN
            2 * d_model             # LayerNorm
        ) +
        d_model * vocab             # Output projection
    )
    
    # Decoder-only
    dec_params = (
        vocab * d_model +           # Embedding
        num_layers * (
            4 * d_model * d_model + # Attention Q,K,V,O
            2 * d_model * d_ff +    # FFN
            2 * d_model             # LayerNorm
        )
        # Note: Output projection shares weights with embedding
    )
    
    # Encoder-decoder
    enc_dec_params = (
        vocab * d_model +           # Encoder embedding
        vocab * d_model +           # Decoder embedding
        num_layers * (
            4 * d_model * d_model + # Encoder attention
            2 * d_model * d_ff +    # Encoder FFN
            2 * d_model             # Encoder LayerNorm
        ) +
        num_layers * (
            8 * d_model * d_model + # Decoder self + cross attention
            2 * d_model * d_ff +    # Decoder FFN
            3 * d_model             # Decoder LayerNorm
        ) +
        d_model * vocab             # Output projection
    )
    
    print(f"\nEncoder-Only:     {enc_params:>8,} parameters")
    print(f"Decoder-Only:     {dec_params:>8,} parameters")
    print(f"Encoder-Decoder:  {enc_dec_params:>8,} parameters")
    
    print(f"\nEncoder-Decoder has ~{enc_dec_params / dec_params:.1f}x more parameters")
    print("(because it has both encoder and decoder stacks)")


def main():
    """Run all comparisons."""
    
    print("=" * 80)
    print("COMPREHENSIVE ARCHITECTURE COMPARISON")
    print("=" * 80)
    
    demonstrate_attention_patterns()
    demonstrate_training_objectives()
    demonstrate_inference()
    demonstrate_use_cases()
    demonstrate_parameter_sharing()
    
    print("\n" + "=" * 80)
    print("SUMMARY: CHOOSING THE RIGHT ARCHITECTURE")
    print("=" * 80)
    
    print("\n🎯 Decision Tree:")
    print("")
    print("  Do you need to GENERATE text?")
    print("    No  → Use ENCODER-ONLY (fastest, most efficient)")
    print("    Yes ↓")
    print("")
    print("  Is your input and output clearly separated?")
    print("    Yes → Use ENCODER-DECODER (translation, summarization)")
    print("    No  → Use DECODER-ONLY (open-ended generation, chat)")
    
    print("\n💡 Pro Tips:")
    print("")
    print("  1. Modern trend: Decoder-only models are increasingly popular")
    print("     (GPT-3, GPT-4, LLaMA) because they're more flexible")
    print("")
    print("  2. For production: Consider inference speed")
    print("     Encoder-only: 5ms")
    print("     Decoder-only: ~1s for 20 tokens")
    print("     Encoder-decoder: ~750ms for 15 tokens")
    print("")
    print("  3. For training: All three require similar compute for")
    print("     comparable model sizes")
    print("")
    print("  4. For fine-tuning: Start with pretrained models when possible")
    print("     BERT/RoBERTa for encoder")
    print("     GPT-2/3 for decoder")
    print("     T5/BART for encoder-decoder")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE! ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
