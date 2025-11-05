"""
Unified Training Script for All LLM Architectures
==================================================

This script trains and compares all three transformer architectures:
1. Encoder-Only (BERT-style)
2. Decoder-Only (GPT-style)
3. Encoder-Decoder (T5-style)

It also demonstrates:
- Learning rate scheduling
- Gradient accumulation
- Model evaluation
- Inference strategies
- Performance comparison
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math
import time
from tqdm import tqdm

from toy_dataset import create_dataloaders
from encoder_only import EncoderOnlyTransformer
from decoder_only import DecoderOnlyTransformer
from encoder_decoder import EncoderDecoderTransformer


class WarmupCosineScheduler:
    """
    Learning Rate Scheduler with Warmup and Cosine Decay.
    
    Why Learning Rate Scheduling?
    - Start with small LR to stabilize training (warmup)
    - Gradually increase to reach full learning capacity
    - Decay towards end to fine-tune (cosine decay)
    
    Schedule:
    - Warmup: Linear increase from 0 to max_lr
    - Decay: Cosine decay from max_lr to min_lr
    
    This is the state-of-the-art schedule used in GPT-3, BERT, etc.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of initial LR
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def get_lr_scale(self, step: int) -> float:
        """Compute LR scale factor for current step."""
        if step < self.warmup_steps:
            # Linear warmup
            return step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
    
    def step(self, current_step: int):
        """Update learning rate."""
        scale = self.get_lr_scale(current_step)
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * scale


def count_parameters(model: nn.Module) -> tuple:
    """
    Count total and trainable parameters.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_encoder_only(
    model: EncoderOnlyTransformer,
    train_loader,
    val_loader,
    device,
    num_epochs: int = 5,
    learning_rate: float = 0.0001,
    use_scheduler: bool = True,
    accumulation_steps: int = 1
):
    """
    Train encoder-only model.
    
    Gradient Accumulation:
    - Simulates larger batch size with limited memory
    - Accumulate gradients over multiple mini-batches
    - Update weights once per accumulation_steps batches
    - Effective batch size = batch_size * accumulation_steps
    """
    print("\n" + "=" * 80)
    print("TRAINING ENCODER-ONLY MODEL")
    print("=" * 80)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Setup scheduler
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    warmup_steps = total_steps // 10  # 10% warmup
    
    if use_scheduler:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask = batch['src_mask'].to(device)
            
            # Use first target token as label (dummy task)
            labels = tgt[:, 0]
            
            # Forward pass
            logits, _ = model(src, src_mask.unsqueeze(1).unsqueeze(2))
            
            # Compute loss
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Update learning rate
                if use_scheduler:
                    scheduler.step(global_step)
                
                global_step += 1
            
            # Track metrics
            train_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%',
                'lr': f'{current_lr:.6f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                src_mask = batch['src_mask'].to(device)
                
                labels = tgt[:, 0]
                
                logits, _ = model(src, src_mask.unsqueeze(1).unsqueeze(2))
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_encoder_only.pt')
            print("✓ Saved best model")
    
    return model


def train_decoder_only(
    model: DecoderOnlyTransformer,
    train_loader,
    val_loader,
    device,
    num_epochs: int = 5,
    learning_rate: float = 0.0001,
    use_scheduler: bool = True,
    accumulation_steps: int = 1
):
    """Train decoder-only model."""
    print("\n" + "=" * 80)
    print("TRAINING DECODER-ONLY MODEL")
    print("=" * 80)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Setup scheduler
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    warmup_steps = total_steps // 10
    
    if use_scheduler:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            x = batch['decoder_input'].to(device)
            y = batch['decoder_output'].to(device)
            
            # Forward pass
            logits = model(x)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if use_scheduler:
                    scheduler.step(global_step)
                
                global_step += 1
            
            train_loss += loss.item() * accumulation_steps
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['decoder_input'].to(device)
                y = batch['decoder_output'].to(device)
                
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation - Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_decoder_only.pt')
            print("✓ Saved best model")
    
    return model


def train_encoder_decoder(
    model: EncoderDecoderTransformer,
    train_loader,
    val_loader,
    device,
    num_epochs: int = 10,
    learning_rate: float = 0.0001,
    use_scheduler: bool = True,
    accumulation_steps: int = 1
):
    """Train encoder-decoder model."""
    print("\n" + "=" * 80)
    print("TRAINING ENCODER-DECODER MODEL")
    print("=" * 80)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Setup scheduler
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    warmup_steps = total_steps // 10
    
    if use_scheduler:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            src = batch['src'].to(device)
            tgt_input = batch['decoder_input'].to(device)
            tgt_output = batch['decoder_output'].to(device)
            
            # Forward pass
            logits = model(src, tgt_input)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if use_scheduler:
                    scheduler.step(global_step)
                
                global_step += 1
            
            train_loss += loss.item() * accumulation_steps
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt_input = batch['decoder_input'].to(device)
                tgt_output = batch['decoder_output'].to(device)
                
                logits = model(src, tgt_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation - Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_encoder_decoder.pt')
            print("✓ Saved best model")
    
    return model


def demo_inference(dataset, device):
    """Demonstrate inference with all three models."""
    print("\n" + "=" * 80)
    print("INFERENCE COMPARISON")
    print("=" * 80)
    
    # Get a test sample
    sample = dataset[0]
    src_tokens = sample['src_tokens']
    tgt_tokens = sample['tgt_tokens']
    
    print(f"\nTest Example:")
    print(f"  Input (digits): {' '.join(src_tokens)}")
    print(f"  Target (words): {' '.join(tgt_tokens)}")
    
    src = sample['src'].unsqueeze(0).to(device)
    
    # Load models
    print("\n" + "-" * 80)
    print("1. ENCODER-ONLY MODEL")
    print("-" * 80)
    
    try:
        encoder_model = EncoderOnlyTransformer(
            vocab_size=dataset.src_vocab_size,
            d_model=128,
            num_layers=3,
            num_heads=4,
            d_ff=512,
            num_classes=dataset.tgt_vocab_size
        ).to(device)
        encoder_model.load_state_dict(torch.load('best_encoder_only.pt'))
        encoder_model.eval()
        
        with torch.no_grad():
            logits, _ = encoder_model(src)
            predictions = torch.argmax(logits, dim=-1)
            pred_token = dataset.tgt_idx2token.get(predictions[0].item(), '<UNK>')
            print(f"  Classification prediction: {pred_token}")
            print("  Note: Encoder-only models are for understanding, not generation")
    except:
        print("  Model not found or error loading")
    
    print("\n" + "-" * 80)
    print("2. DECODER-ONLY MODEL")
    print("-" * 80)
    
    try:
        decoder_model = DecoderOnlyTransformer(
            vocab_size=max(dataset.src_vocab_size, dataset.tgt_vocab_size),
            d_model=128,
            num_layers=3,
            num_heads=4,
            d_ff=512
        ).to(device)
        decoder_model.load_state_dict(torch.load('best_decoder_only.pt'))
        decoder_model.eval()
        
        prompt = src
        
        print("  Greedy generation:")
        generated = decoder_model.generate(prompt, max_new_tokens=5, temperature=0.1)
        gen_tokens = [dataset.src_idx2token.get(idx.item(), '<UNK>') for idx in generated[0]]
        print(f"    {' '.join(gen_tokens)}")
        
        print("  Sampling (temp=1.0):")
        generated = decoder_model.generate(prompt, max_new_tokens=5, temperature=1.0)
        gen_tokens = [dataset.src_idx2token.get(idx.item(), '<UNK>') for idx in generated[0]]
        print(f"    {' '.join(gen_tokens)}")
        
        print("  Top-k sampling (k=5):")
        generated = decoder_model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=5)
        gen_tokens = [dataset.src_idx2token.get(idx.item(), '<UNK>') for idx in generated[0]]
        print(f"    {' '.join(gen_tokens)}")
    except Exception as e:
        print(f"  Model not found or error loading: {e}")
    
    print("\n" + "-" * 80)
    print("3. ENCODER-DECODER MODEL")
    print("-" * 80)
    
    try:
        enc_dec_model = EncoderDecoderTransformer(
            src_vocab_size=dataset.src_vocab_size,
            tgt_vocab_size=dataset.tgt_vocab_size,
            d_model=128,
            num_layers=3,
            num_heads=4,
            d_ff=512
        ).to(device)
        enc_dec_model.load_state_dict(torch.load('best_encoder_decoder.pt'))
        enc_dec_model.eval()
        
        print("  Greedy generation:")
        generated = enc_dec_model.generate(
            src, max_length=10,
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            temperature=0.1
        )
        gen_tokens = [dataset.tgt_idx2token.get(idx.item(), '<UNK>') for idx in generated[0]]
        print(f"    {' '.join(gen_tokens)}")
        
        print("  Beam search (width=5):")
        best_seq, score = enc_dec_model.beam_search(
            src, beam_width=5,
            max_length=10,
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx
        )
        gen_tokens = [dataset.tgt_idx2token.get(idx.item(), '<UNK>') for idx in best_seq[0]]
        print(f"    {' '.join(gen_tokens)}")
        print(f"    Score: {score:.4f}")
    except Exception as e:
        print(f"  Model not found or error loading: {e}")


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("UNIFIED LLM TRAINING PIPELINE")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create dataset
    print("\nCreating dataset...")
    train_loader, val_loader, dataset = create_dataloaders(
        batch_size=32,
        num_samples=1000
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Source vocab size: {dataset.src_vocab_size}")
    print(f"  Target vocab size: {dataset.tgt_vocab_size}")
    
    # Hyperparameters
    d_model = 128
    num_layers = 3
    num_heads = 4
    d_ff = 512
    dropout = 0.1
    learning_rate = 0.0001
    
    # Train Encoder-Only
    print("\n" + "=" * 80)
    print("ENCODER-ONLY TRANSFORMER")
    print("=" * 80)
    
    encoder_model = EncoderOnlyTransformer(
        vocab_size=dataset.src_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        num_classes=dataset.tgt_vocab_size
    )
    
    total, trainable = count_parameters(encoder_model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    
    encoder_model = train_encoder_only(
        encoder_model, train_loader, val_loader, device,
        num_epochs=5, learning_rate=learning_rate, use_scheduler=True
    )
    
    # Train Decoder-Only
    print("\n" + "=" * 80)
    print("DECODER-ONLY TRANSFORMER")
    print("=" * 80)
    
    decoder_model = DecoderOnlyTransformer(
        vocab_size=max(dataset.src_vocab_size, dataset.tgt_vocab_size),
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )
    
    total, trainable = count_parameters(decoder_model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    
    decoder_model = train_decoder_only(
        decoder_model, train_loader, val_loader, device,
        num_epochs=5, learning_rate=learning_rate, use_scheduler=True
    )
    
    # Train Encoder-Decoder
    print("\n" + "=" * 80)
    print("ENCODER-DECODER TRANSFORMER")
    print("=" * 80)
    
    enc_dec_model = EncoderDecoderTransformer(
        src_vocab_size=dataset.src_vocab_size,
        tgt_vocab_size=dataset.tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )
    
    total, trainable = count_parameters(enc_dec_model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    
    enc_dec_model = train_encoder_decoder(
        enc_dec_model, train_loader, val_loader, device,
        num_epochs=10, learning_rate=learning_rate, use_scheduler=True
    )
    
    # Demo inference
    demo_inference(dataset, device)
    
    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE! ✓")
    print("=" * 80)
    
    print("\nKey Takeaways:")
    print("1. Encoder-Only: Best for classification/understanding tasks")
    print("2. Decoder-Only: Best for autoregressive generation")
    print("3. Encoder-Decoder: Best for sequence-to-sequence tasks")
    print("\nAll models saved with 'best_' prefix")


if __name__ == "__main__":
    main()
