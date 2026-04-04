import os
import glob
import csv
import json
import torch
from torch.utils.data import IterableDataset, DataLoader


class CyberDataset(IterableDataset):
    def __init__(self, data_dir, vocab_path, config_path):

        # 1. Load Vocabulary and Config
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.block_size = config['n_positions']

        # 2. Identify all Spark partition files
        self.files = glob.glob(os.path.join(data_dir, "part-*.csv"))
        self.unk_token = "[UNK]" # Fallback

    def tokenize_and_encode(self, sequence_str):
        tokens = sequence_str.split(" ")
        encoded = []
        
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            else:
                # 3. Prefix-based OOV Mapping
                # Logic: [PORT_443] -> prefix: PORT
                clean_token = token.strip("[]")
                if "_" in clean_token:
                    prefix = clean_token.rsplit("_", 1)[0]
                    unk_key = f"[{prefix}_UNK]"
                    encoded.append(self.vocab.get(unk_key, self.vocab.get("[UNK]", 0)))
                else:
                    encoded.append(self.vocab.get("[UNK]", 0))
        return encoded

    def __iter__(self):
        for file_path in self.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None) # Skip header
                
                for row in reader:
                    if not row: continue
                    sequence_str = row[0]
                    label_str = row[1]
                    
                    # 4. Encoding and Label Mapping
                    tokens_id = self.tokenize_and_encode(sequence_str)
                    label = 1 if label_str != "Benign" else 0
                    
                    # 5. Sliding Window for Transformer Block Size
                    # This ensures the model receives fixed-length sequences
                    for i in range(0, len(tokens_id) - self.block_size + 1, self.block_size // 2):
                        chunk = tokens_id[i : i + self.block_size]
                        if len(chunk) < self.block_size:
                            continue # Or apply padding
                        
                        yield torch.tensor(chunk, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def get_dataloader(data_dir, vocab_path, config_path, batch_size=32):
    dataset = CyberDataset(data_dir, vocab_path, config_path)
    return DataLoader(dataset, batch_size=batch_size, num_workers=2)

if __name__ == "__main__":
    # Test Implementation
    DATA_DIR = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\nano_gpt_sequences"
    VOCAB_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\data\processed\vocab_only_benign.json"
    CONFIG_PATH = r"C:\Users\jgwak\OneDrive\Desktop\cyber-nano-gpt\config\config.json"
    
    loader = get_dataloader(DATA_DIR, VOCAB_PATH, CONFIG_PATH)
    
    print(">>> Testing DataLoader Stream...")
    for x, y in loader:
        print(f"Batch Shape: {x.shape}, Labels Sample: {y[:5].tolist()}")
        break