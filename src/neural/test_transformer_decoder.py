import os
import io
import re
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import editdistance
from tqdm import tqdm

os.environ["HIP_VISIBLE_DEVICES"] = "1" 
torch.backends.cudnn.enabled = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubwordVocab:
    def __init__(self, special_tokens):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        # Regex MUST use sorted for longest-match-first
        self.special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        self.pattern = re.compile("|".join([re.escape(t) for t in self.special_tokens_sorted]) + "|.")
        # Index assignment MUST use the original order to match training
        self.raw_special_tokens = special_tokens

    def build_vocab(self, datasets):
        unique_chars = set()
        for ds in datasets:
            label_key = next(c for c in ['txt', 'latex', 'text', 'formula'] if c in ds.column_names)
            for text in ds[label_key]:
                if text: unique_chars.update(text)
        
        for char in sorted(list(unique_chars)):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
        
        # USE RAW LIST TO MATCH TRAINING INDEXING
        for t in self.raw_special_tokens:
            if t not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[t] = idx
                self.idx2char[idx] = t
        print(f"Vocab Synchronized. Size: {len(self.char2idx)}")

    def encode(self, text):
        if not text: return []
        tokens = self.pattern.findall(text)
        return [self.char2idx.get(t, 1) for t in tokens]

    def __len__(self): return len(self.char2idx)

def collate_fn(batch):
    images, tokens = zip(*batch)
    images = torch.stack(images)
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    return images, padded_tokens

class UnifiedOCRDataset(Dataset):
    def __init__(self, hf_dataset, vocab, transform=None):
        self.data, self.vocab, self.transform = hf_dataset, vocab, transform
        self.label_key = next(c for c in ['txt', 'latex', 'text', 'formula'] if c in hf_dataset.column_names)
        self.image_key = next(c for c in ['png', 'image', 'img', 'line_image'] if c in hf_dataset.column_names)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item[self.image_key].convert("L")
        if self.transform: image = self.transform(image)
        encoded = [2] + self.vocab.encode(item[self.label_key]) + [3]
        return image, torch.tensor(encoded)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=32, max_w=128):
        super().__init__()
        pe = torch.zeros(max_h, max_w, d_model)
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        pos_w, pos_h = torch.arange(0., max_w).unsqueeze(1), torch.arange(0., max_h).unsqueeze(1)
        pe_w, pe_h = torch.zeros(max_w, d_model_half), torch.zeros(max_h, d_model_half)
        pe_w[:, 0::2], pe_w[:, 1::2] = torch.sin(pos_w * div_term), torch.cos(pos_w * div_term)
        pe_h[:, 0::2], pe_h[:, 1::2] = torch.sin(pos_h * div_term), torch.cos(pos_h * div_term)
        for h in range(max_h):
            for w in range(max_w): pe[h, w, :d_model_half] = pe_h[h]; pe[h, w, d_model_half:] = pe_w[w]
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        B, C, H, W = x.shape
        return (x.permute(0, 2, 3, 1) + self.pe[:, :H, :W, :]).reshape(B, H * W, C)

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class CNNTransformerOCR(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, d_model, 3, 1, 1), nn.BatchNorm2d(d_model), nn.ReLU()
        )
        self.pos_encoder_2d = PositionalEncoding2D(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_1d = PositionalEncoding1D(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

def greedy_decode_batch(model, images, vocab, max_len=150):
    model.eval()
    B = images.size(0)
    with torch.no_grad():
        features = model.cnn(images)
        memory = model.pos_encoder_2d(features)
        tgt = torch.full((B, 1), 2, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            tgt_emb = model.pos_encoder_1d(model.embedding(tgt))
            mask = model.generate_square_subsequent_mask(tgt.size(1))
            output = model.transformer_decoder(tgt_emb, memory, tgt_mask=mask)
            logits = model.fc_out(output)[:, -1, :]
            next_tokens = logits.argmax(dim=-1).unsqueeze(1)
            tgt = torch.cat([tgt, next_tokens], dim=1)
            done |= (next_tokens.squeeze(1) == 3)
            if done.all(): break
                
        decoded_strings = []
        for b in range(B):
            indices = tgt[b].tolist()[1:]
            tokens = []
            for idx in indices:
                if idx == 3: break
                tokens.append(vocab.idx2char.get(idx, ""))
            decoded_strings.append("".join(tokens))
        return decoded_strings

if __name__ == "__main__":
    ST = ["\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
          "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"]
    
    # CRITICAL: Build Vocab from TRAIN splits to match model weights
    print("Re-building Training Vocab for synchronization...")
    latex_train = load_dataset("LiamYo/MathWritingHandwritten", split="train")
    text_train = load_dataset("Teklia/IAM-line", split="train")
    
    vocab = SubwordVocab(ST)
    vocab.build_vocab([latex_train, text_train])
    
    # Free up memory
    del latex_train, text_train

    # Load actual Validation Sets
    print("Fetching Validation Datasets...")
    l_val = load_dataset("LiamYo/MathWritingHandwritten", split="validation")
    t_val = load_dataset("Teklia/IAM-line", split="validation")
    
    transform = transforms.Compose([
        transforms.Resize((128, 1024)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_set = ConcatDataset([UnifiedOCRDataset(t_val, vocab, transform), 
                             UnifiedOCRDataset(l_val, vocab, transform)])
    
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    checkpoints = [12]
    print(f"\n{'Epoch':<8} | {'Exact Match':<12} | {'CER':<8} | {'WER':<8}")
    print("-" * 50)
    
    for ep in checkpoints:
        fname = f"ocr_hybrid_transformer_epoch_{ep}.pth"
        if not os.path.exists(fname): continue
        
        model = CNNTransformerOCR(len(vocab.char2idx)).to(device)
        model.load_state_dict(torch.load(fname, map_location=device))
        model.eval()
        
        t_cer, t_wer, t_acc, t_chars, t_words, count = 0, 0, 0, 0, 0, 0
        
        for images, tokens in tqdm(val_loader, desc=f"Evaluating Epoch {ep}"):
            images = images.to(device)
            preds = greedy_decode_batch(model, images, vocab)
            
            for i, pred in enumerate(preds):
                gt = "".join([vocab.idx2char.get(idx.item(), "") 
                              for idx in tokens[i] if idx.item() not in [0, 2, 3]])
                if count < 5: # Just show the first few
                    print(f"\n--- Sample {count} ---")
                    print(f"GT:   {gt}")
                    print(f"PRED: {pred}")
                    print(f"Match? {pred.strip() == gt.strip()}")
                t_cer += editdistance.eval(pred, gt)
                t_chars += len(gt) if len(gt) > 0 else 1
                t_wer += editdistance.eval(pred.split(), gt.split())
                t_words += len(gt.split()) if len(gt.split()) > 0 else 1
                if pred.strip() == gt.strip(): t_acc += 1
                count += 1
        
        print(f"{ep:<8} | {t_acc/count:<12.4f} | {t_cer/t_chars:<8.4f} | {t_wer/t_words:<8.4f}")