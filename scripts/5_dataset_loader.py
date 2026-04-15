import os
import glob
import csv
import json
import torch
from torch.utils.data import IterableDataset, DataLoader

class CyberDataset(IterableDataset):
    def __init__(self, data_dir, vocab_path, config_path):
        """
        Custom Dataset for Cyber Log Sequences using Block-level buffering.
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.block_size = config['n_positions']

        self.files = glob.glob(os.path.join(data_dir, "part-*.csv"))

    def tokenize_and_encode(self, sequence_str):
        """
        Converts raw string sequences into integer IDs based on prefix-specific UNK mapping.
        """
        tokens = sequence_str.split(" ")
        encoded = []
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            else:
                # Prefix-based OOV (Out-of-Vocabulary) handling
                clean_token = token.strip("[]")
                if "_" in clean_token:
                    try:
                        prefix = clean_token.rsplit("_", 1)[0]
                        unk_key = f"[{prefix}_UNK]"
                        # Fallback to prefix-specific UNK, then index 0 if not found
                        encoded.append(self.vocab.get(unk_key, 0))
                    except ValueError:
                        encoded.append(0)
                else:
                    encoded.append(0)
        return encoded

    def __iter__(self):
        """
        Iterates over CSV partitions and yields (x, y) pairs with block isolation.
        """
        for file_path in self.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None) # Skip header
                
                current_block_id = None
                block_buffer = []

                for row in reader:
                    if not row or len(row) < 4: continue
                    
                    # Alignment with Spark output schema: [block_id: row[1], label: row[2], sequence: row[3]]
                    block_id = row[1]
                    sequence_str = row[3]

                    # Detect block boundary: If ID changes, process the accumulated buffer
                    if current_block_id is not None and block_id != current_block_id:

                        # Session change detected: Flush accumulated tokens into the sliding-window pipeline 
                        # to ensure data isolation between block_ids.
                        yield from self._generate_windows(block_buffer)
                        block_buffer = []

                    current_block_id = block_id
                    # Accumulate tokens from the current row into the buffer
                    block_buffer.extend(self.tokenize_and_encode(sequence_str))

                # Final flush for the last block in the file
                if block_buffer:
                    yield from self._generate_windows(block_buffer)

    def _generate_windows(self, tokens):
        """
        Slices the accumulated token buffer into (x, y) pairs for Next Token Prediction.
        """
        # Minimum required tokens = context_window + 1 (to have a target for the last input)
        target_len = self.block_size + 1
        
        if len(tokens) < target_len:
            return

        # Use stride = block_size // 2 to balance data coverage and redundancy
        for i in range(0, len(tokens) - target_len + 1, self.block_size // 2):
            chunk = tokens[i : i + target_len]
            
            # Input x: [T0...T255], Target y: [T1...T256]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            
            yield x, y

def get_dataloader(data_dir, vocab_path, config_path, batch_size=32):
    dataset = CyberDataset(data_dir, vocab_path, config_path)
    # num_workers=0 is safer for initial debugging with IterableDataset
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\nano_gpt_sequences"
    VOCAB_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\vocab_only_benign.json"
    CONFIG_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\config\config.json"
    
    loader = get_dataloader(DATA_DIR, VOCAB_PATH, CONFIG_PATH)
    
    print(">>> Sanity Check: Streaming Data Batches...")
    for x, y in loader:
        print(f"Batch X Shape: {x.shape}") # Expected: [batch_size, 256]
        print(f"Batch Y Shape: {y.shape}") # Expected: [batch_size, 256]
        # Shift Check: x[0, 1] must equal y[0, 0]
        is_correct = torch.equal(x[0, 1:], y[0, :-1])
        print(f"NTP Shift Integrity: {'PASS' if is_correct else 'FAIL'}")
        break