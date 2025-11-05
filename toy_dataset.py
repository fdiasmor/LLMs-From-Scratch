"""
Toy Dataset for LLM Training
=============================

This module creates a simple synthetic dataset for demonstrating LLM training.
We'll create a simple number-to-word translation task:
    Input: "1 2 3" -> Output: "one two three"

This is simple enough to train quickly but complex enough to demonstrate
all the key concepts of sequence modeling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple, Dict


class NumberTranslationDataset(Dataset):
    """
    A toy dataset that translates sequences of numbers to their word equivalents.
    
    Example:
        Input: "1 2 3" -> Output: "one two three"
        Input: "5 1 4" -> Output: "five one four"
    
    This dataset demonstrates:
    - Sequence-to-sequence mapping
    - Vocabulary building
    - Tokenization
    - Padding and masking
    """
    
    def __init__(self, num_samples: int = 1000, max_seq_length: int = 5, seed: int = 42):
        """
        Args:
            num_samples: Number of training examples to generate
            max_seq_length: Maximum length of number sequences (1 to max_seq_length)
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Define our vocabulary
        # Special tokens:
        # - <PAD>: Padding token (for sequences of different lengths)
        # - <SOS>: Start of sequence (decoder needs to know when to start)
        # - <EOS>: End of sequence (decoder needs to know when to stop)
        # - <UNK>: Unknown token (for handling out-of-vocabulary words)
        
        self.number_words = ['zero', 'one', 'two', 'three', 'four', 
                            'five', 'six', 'seven', 'eight', 'nine']
        self.number_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'  # Start of sequence
        self.EOS_TOKEN = '<EOS>'  # End of sequence
        self.UNK_TOKEN = '<UNK>'  # Unknown token
        
        # Build vocabularies
        # Source vocabulary: digits + special tokens
        self.src_vocab = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, 
                         self.UNK_TOKEN] + self.number_digits
        
        # Target vocabulary: words + special tokens
        self.tgt_vocab = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, 
                         self.UNK_TOKEN] + self.number_words
        
        # Create token-to-index mappings (essential for embedding layers)
        self.src_token2idx = {token: idx for idx, token in enumerate(self.src_vocab)}
        self.src_idx2token = {idx: token for token, idx in self.src_token2idx.items()}
        
        self.tgt_token2idx = {token: idx for idx, token in enumerate(self.tgt_vocab)}
        self.tgt_idx2token = {idx: token for token, idx in self.tgt_token2idx.items()}
        
        # Vocabulary sizes (needed for embedding layers)
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
        
        # Special token indices (used frequently, so cache them)
        self.pad_idx = self.src_token2idx[self.PAD_TOKEN]
        self.sos_idx = self.tgt_token2idx[self.SOS_TOKEN]
        self.eos_idx = self.tgt_token2idx[self.EOS_TOKEN]
        
        # Generate dataset
        self.data = self._generate_data(num_samples, max_seq_length)
        
    def _generate_data(self, num_samples: int, max_seq_length: int) -> List[Tuple[List[str], List[str]]]:
        """
        Generate random number sequences and their word translations.
        
        Returns:
            List of (source_sequence, target_sequence) pairs
        """
        data = []
        for _ in range(num_samples):
            # Random sequence length (1 to max_seq_length)
            seq_len = random.randint(1, max_seq_length)
            
            # Generate random digits
            digits = [str(random.randint(0, 9)) for _ in range(seq_len)]
            
            # Convert to words
            words = [self.number_words[int(d)] for d in digits]
            
            data.append((digits, words))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns a dictionary containing:
        - src: Source sequence (digits) as token indices
        - tgt: Target sequence (words) as token indices
        - src_mask: Mask for padding tokens in source
        - tgt_mask: Mask for padding tokens in target
        """
        src_tokens, tgt_tokens = self.data[idx]
        
        # Convert tokens to indices
        # We add EOS token to mark the end of sequences
        src_indices = [self.src_token2idx[token] for token in src_tokens]
        src_indices.append(self.eos_idx)  # Add EOS
        
        # For target, we need both input (with SOS) and output (with EOS)
        # Decoder input: <SOS> word1 word2 word3
        # Decoder output: word1 word2 word3 <EOS>
        tgt_indices = [self.tgt_token2idx[token] for token in tgt_tokens]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_tokens': src_tokens,  # Keep original for reference
            'tgt_tokens': tgt_tokens
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    This function:
    1. Pads sequences to the same length within a batch
    2. Creates attention masks to ignore padding tokens
    3. Prepares decoder input and target sequences
    
    Padding is necessary because:
    - Neural networks require fixed-size inputs
    - Sequences in a batch have different lengths
    - We use masking to tell the model to ignore padding
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Dictionary with batched and padded tensors
    """
    # Extract sequences
    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]
    
    # Get padding index (same for both source and target in our case)
    pad_idx = 0  # PAD token is always at index 0
    
    # Pad source sequences
    # torch.nn.utils.rnn.pad_sequence adds padding to make all sequences same length
    # batch_first=True means shape will be (batch_size, seq_length)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_seqs, batch_first=True, padding_value=pad_idx
    )
    
    # Pad target sequences
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_seqs, batch_first=True, padding_value=pad_idx
    )
    
    # Create padding masks
    # Mask is True where there's actual content, False for padding
    # This is used in attention to ignore padding positions
    src_mask = (src_padded != pad_idx)  # Shape: (batch_size, src_seq_len)
    tgt_mask = (tgt_padded != pad_idx)  # Shape: (batch_size, tgt_seq_len)
    
    # For decoder training, we need:
    # 1. Decoder input: <SOS> + target sequence (without last token)
    # 2. Decoder output: target sequence + <EOS> (without first token)
    # This is the "teacher forcing" strategy
    
    sos_idx = 1  # SOS token index
    eos_idx = 2  # EOS token index
    
    # Create decoder input: prepend SOS token
    batch_size = tgt_padded.size(0)
    sos_tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long)
    decoder_input = torch.cat([sos_tokens, tgt_padded], dim=1)
    
    # Create decoder output: append EOS token
    eos_tokens = torch.full((batch_size, 1), eos_idx, dtype=torch.long)
    decoder_output = torch.cat([tgt_padded, eos_tokens], dim=1)
    
    # Update masks for decoder sequences
    decoder_input_mask = (decoder_input != pad_idx)
    decoder_output_mask = (decoder_output != pad_idx)
    
    return {
        'src': src_padded,                    # (batch_size, src_seq_len)
        'tgt': tgt_padded,                    # (batch_size, tgt_seq_len)
        'src_mask': src_mask,                 # (batch_size, src_seq_len)
        'tgt_mask': tgt_mask,                 # (batch_size, tgt_seq_len)
        'decoder_input': decoder_input,       # (batch_size, tgt_seq_len + 1)
        'decoder_output': decoder_output,     # (batch_size, tgt_seq_len + 1)
        'decoder_input_mask': decoder_input_mask,
        'decoder_output_mask': decoder_output_mask,
    }


def create_dataloaders(batch_size: int = 32, num_samples: int = 1000):
    """
    Create train and validation dataloaders.
    
    Args:
        batch_size: Number of samples per batch
        num_samples: Total number of samples to generate
        
    Returns:
        train_loader, val_loader, dataset (for accessing vocab info)
    """
    # Create dataset
    dataset = NumberTranslationDataset(num_samples=num_samples)
    
    # Split into train/val (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    # DataLoader handles batching and shuffling
    # num_workers=0 for Windows compatibility (multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data for better generalization
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader, dataset


# Test the dataset
if __name__ == "__main__":
    print("=" * 80)
    print("TOY DATASET DEMO")
    print("=" * 80)
    
    # Create dataset
    dataset = NumberTranslationDataset(num_samples=10, max_seq_length=5)
    
    print(f"\nSource Vocabulary ({dataset.src_vocab_size} tokens):")
    print(dataset.src_vocab)
    
    print(f"\nTarget Vocabulary ({dataset.tgt_vocab_size} tokens):")
    print(dataset.tgt_vocab)
    
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    
    # Show some examples
    for i in range(5):
        item = dataset[i]
        print(f"\nExample {i+1}:")
        print(f"  Source tokens: {item['src_tokens']}")
        print(f"  Target tokens: {item['tgt_tokens']}")
        print(f"  Source indices: {item['src'].tolist()}")
        print(f"  Target indices: {item['tgt'].tolist()}")
    
    print("\n" + "=" * 80)
    print("DATALOADER DEMO (with padding and masking)")
    print("=" * 80)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(batch_size=4, num_samples=20)
    
    # Get one batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"\nSource sequences (padded):")
    print(batch['src'])
    
    print(f"\nSource masks (True = real token, False = padding):")
    print(batch['src_mask'])
    
    print(f"\nDecoder input sequences (with SOS):")
    print(batch['decoder_input'])
    
    print(f"\nDecoder output sequences (with EOS):")
    print(batch['decoder_output'])
    
    print("\n" + "=" * 80)
    print("Dataset creation successful! ✓")
    print("=" * 80)
