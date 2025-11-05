"""
Quick Start Script
==================

This script provides a quick demo of all LLM architectures.
Perfect for getting started quickly!
"""

import torch
from toy_dataset import create_dataloaders
from encoder_only import EncoderOnlyTransformer
from decoder_only import DecoderOnlyTransformer
from encoder_decoder import EncoderDecoderTransformer


def quick_demo():
    """Quick demonstration of all architectures."""
    
    print("=" * 80)
    print("QUICK START - LLM ARCHITECTURES DEMO")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create small dataset
    print("\n1. Creating toy dataset...")
    train_loader, val_loader, dataset = create_dataloaders(
        batch_size=16,
        num_samples=100  # Small for quick demo
    )
    print(f"   ✓ Dataset ready: {len(dataset)} samples")
    print(f"   ✓ Source vocab: {dataset.src_vocab_size} tokens")
    print(f"   ✓ Target vocab: {dataset.tgt_vocab_size} tokens")
    
    # Show sample
    sample = dataset[0]
    print(f"\n   Example:")
    print(f"   Input:  {' '.join(sample['src_tokens'])}")
    print(f"   Output: {' '.join(sample['tgt_tokens'])}")
    
    # Model parameters (small for quick demo)
    d_model = 64
    num_layers = 2
    num_heads = 2
    d_ff = 256
    
    print("\n" + "=" * 80)
    print("2. ENCODER-ONLY MODEL (BERT-style)")
    print("=" * 80)
    print("   Use case: Text classification, understanding tasks")
    print("   Attention: Bidirectional (can see all tokens)")
    
    encoder_model = EncoderOnlyTransformer(
        vocab_size=dataset.src_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        num_classes=dataset.tgt_vocab_size
    ).to(device)
    
    params = sum(p.numel() for p in encoder_model.parameters())
    print(f"\n   Parameters: {params:,}")
    
    # Quick training
    print("   Training for 3 epochs...")
    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    encoder_model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            optimizer.zero_grad()
            logits, _ = encoder_model(src)
            loss = criterion(logits, tgt[:, 0])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")
    
    print("   ✓ Encoder-Only training complete!")
    
    print("\n" + "=" * 80)
    print("3. DECODER-ONLY MODEL (GPT-style)")
    print("=" * 80)
    print("   Use case: Text generation, completion")
    print("   Attention: Causal (can only see past tokens)")
    
    decoder_model = DecoderOnlyTransformer(
        vocab_size=max(dataset.src_vocab_size, dataset.tgt_vocab_size),
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    ).to(device)
    
    params = sum(p.numel() for p in decoder_model.parameters())
    print(f"\n   Parameters: {params:,}")
    
    # Quick training
    print("   Training for 3 epochs...")
    optimizer = torch.optim.Adam(decoder_model.parameters(), lr=0.001)
    
    decoder_model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            x = batch['decoder_input'].to(device)
            y = batch['decoder_output'].to(device)
            
            optimizer.zero_grad()
            logits = decoder_model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")
    
    print("   ✓ Decoder-Only training complete!")
    
    # Demo generation
    print("\n   Generation demo:")
    decoder_model.eval()
    prompt_tokens = [dataset.src_token2idx['1'], dataset.src_token2idx['2']]
    prompt = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated = decoder_model.generate(prompt, max_new_tokens=3, temperature=0.5)
        gen_tokens = [dataset.src_idx2token.get(idx.item(), '<UNK>') for idx in generated[0]]
        print(f"   Input:  {' '.join([dataset.src_idx2token[idx] for idx in prompt_tokens])}")
        print(f"   Output: {' '.join(gen_tokens)}")
    
    print("\n" + "=" * 80)
    print("4. ENCODER-DECODER MODEL (T5-style)")
    print("=" * 80)
    print("   Use case: Translation, summarization, seq2seq")
    print("   Attention: Encoder (bidirectional) + Decoder (causal + cross-attention)")
    
    enc_dec_model = EncoderDecoderTransformer(
        src_vocab_size=dataset.src_vocab_size,
        tgt_vocab_size=dataset.tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    ).to(device)
    
    params = sum(p.numel() for p in enc_dec_model.parameters())
    print(f"\n   Parameters: {params:,}")
    
    # Quick training
    print("   Training for 5 epochs...")
    optimizer = torch.optim.Adam(enc_dec_model.parameters(), lr=0.001)
    
    enc_dec_model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            src = batch['src'].to(device)
            tgt_input = batch['decoder_input'].to(device)
            tgt_output = batch['decoder_output'].to(device)
            
            optimizer.zero_grad()
            logits = enc_dec_model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")
    
    print("   ✓ Encoder-Decoder training complete!")
    
    # Demo translation
    print("\n   Translation demo:")
    enc_dec_model.eval()
    sample = dataset[0]
    src = sample['src'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = enc_dec_model.generate(
            src, max_length=10,
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            temperature=0.5
        )
        gen_tokens = [dataset.tgt_idx2token.get(idx.item(), '<UNK>') for idx in generated[0]]
        print(f"   Input:    {' '.join(sample['src_tokens'])}")
        print(f"   Target:   {' '.join(sample['tgt_tokens'])}")
        print(f"   Generated: {' '.join(gen_tokens)}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE! ✓")
    print("=" * 80)
    
    print("\n📚 Next Steps:")
    print("   1. Read CONCEPTS_GUIDE.md for detailed explanations")
    print("   2. Run individual scripts (encoder_only.py, decoder_only.py, encoder_decoder.py)")
    print("   3. Run train_all.py for full training pipeline")
    print("   4. Run visualizations.py to generate helpful diagrams")
    print("   5. Experiment with hyperparameters!")
    
    print("\n🎯 Key Takeaways:")
    print("   • Encoder-Only: Best for understanding (classification, NER)")
    print("   • Decoder-Only: Best for generation (text completion, creative writing)")
    print("   • Encoder-Decoder: Best for transformation (translation, summarization)")
    
    print("\n💡 Tips:")
    print("   • All models use the same core: Multi-head attention + FFN")
    print("   • The difference is in masking and architecture composition")
    print("   • Real models are much larger but follow the same principles!")


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    quick_demo()
