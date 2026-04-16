# LLM help with reformatting (I use camel casing usually, but it looks better with snake casing)
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
from tqdm import tqdm

os.environ["HIP_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
torch.backends.cudnn.enabled = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basically the same I've already been using
class SubwordVocab:
    def __init__(self, special_tokens):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        self.pattern = re.compile("|".join([re.escape(t) for t in self.special_tokens_sorted]) + "|.")
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
        
        for t in self.raw_special_tokens:
            if t not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[t] = idx
                self.idx2char[idx] = t
        print(f"Subword Vocab Size: {len(self.char2idx)}")

    def encode(self, text):
        tokens = self.pattern.findall(text)
        return [self.char2idx.get(t, 1) for t in tokens]

    def __len__(self): return len(self.char2idx)

# Basically the same as I've been using
class UnifiedOCRDataset(Dataset):
    def __init__(self, hf_dataset, vocab, transform=None):
        self.data, self.vocab, self.transform = hf_dataset, vocab, transform
        self.label_key = next(c for c in ['txt', 'latex', 'text', 'formula'] if c in hf_dataset.column_names)
        self.image_key = next(c for c in ['png', 'image', 'img', 'line_image'] if c in hf_dataset.column_names)
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_data = item[self.image_key]
        image = img_data.convert("L") if isinstance(img_data, Image.Image) else Image.open(io.BytesIO(img_data)).convert("L")
        if self.transform: image = self.transform(image)
        
        # Wrap sequence in SOS (2) and EOS (3) for Autoregressive training
        encoded = [2] + self.vocab.encode(item[self.label_key]) + [3]
        return image, torch.tensor(encoded)

def collate_fn(batch):
    images, tokens = zip(*batch)
    images = torch.stack(images)
    # Pad sequences to the max length in the batch using <PAD> (0)
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    return images, padded_tokens

# LLM: "How do I do flatten the positional encoding for the transformer?"
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=32, max_w=128):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_h, max_w, d_model)
        d_model_half = d_model // 2
        
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        pos_w = torch.arange(0., max_w).unsqueeze(1)
        pos_h = torch.arange(0., max_h).unsqueeze(1)
        
        pe_w = torch.zeros(max_w, d_model_half)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)
        
        pe_h = torch.zeros(max_h, d_model_half)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)
        
        for h in range(max_h):
            for w in range(max_w):
                pe[h, w, :d_model_half] = pe_h[h]
                pe[h, w, d_model_half:] = pe_w[w]
                
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = x + self.pe[:, :H, :W, :]
        return x.reshape(B, H * W, C) # Flatten spatial dims for Transformer

# From LLM: "How do I get the positional encoding after it's flattened?"
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CNNTransformerOCR(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_decoder_layers=3):
        super().__init__()
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, d_model, 3, 1, 1), nn.BatchNorm2d(d_model), nn.ReLU()
        )
        self.pos_encoder_2d = PositionalEncoding2D(d_model)
        
        # Transformer Decoder; structure with help from LLM
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_1d = PositionalEncoding1D(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    # From LLM: "Does PyTorch have a way to help with masking so the transformer can't cheat by looking at future values?"
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, images, tgt):
        features = self.cnn(images) 
        memory = self.pos_encoder_2d(features) 

        # With help from LLM
        tgt_emb = self.pos_encoder_1d(self.embedding(tgt))
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
        tgt_key_padding_mask = (tgt == 0) # <PAD> is 0
        
        output = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output)

# This is based on the same loading used before
def load_pretrained_cnn(model, checkpoint_path):
    print(f"Loading pre-trained CNN weights from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: {checkpoint_path} not found. Training CNN from scratch.")
        return model
        
    old_state = torch.load(checkpoint_path, map_location=device)
    new_state = model.state_dict()
    
    transferred_layers = 0
    for name, param in old_state.items():
        new_name = name
        if not new_name.startswith('cnn.'):
            new_name = 'cnn.' + name.replace('features.', '') 

        if new_name in new_state and new_state[new_name].shape == param.shape:
            new_state[new_name].copy_(param)
            transferred_layers += 1
            
    model.load_state_dict(new_state)
    print(f"Successfully transferred {transferred_layers} CNN parameter tensors.")
    return model

def train_transformer(myToken=""):
    ST = ["\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
          "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"]
    
    latex_train = load_dataset("LiamYo/MathWritingHandwritten", split="train", token=myToken)
    text_train = load_dataset("Teklia/IAM-line", split="train", token=myToken)
    
    vocab = SubwordVocab(ST)
    vocab.build_vocab([latex_train, text_train])
    
    transform = transforms.Compose([
        transforms.Resize((128, 1024)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set = ConcatDataset([UnifiedOCRDataset(text_train, vocab, transform), 
                               UnifiedOCRDataset(latex_train, vocab, transform)])
    
    loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    # Initialize Model
    model = CNNTransformerOCR(len(vocab)).to(device)
    
    # Transfer Weights
    model = load_pretrained_cnn(model, "ocr_subword_v2_epoch_9.pth")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # OneCycleLR updated for 240 epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, steps_per_epoch=len(loader), epochs=50)

    # Got help from LLM to update the training loop to the transformer architecture
    print("Starting Hybrid Transformer training for 5 epochs...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, tokens) in enumerate(loader):
            images, tokens = images.to(device), tokens.to(device)
            
            # Autoregressive inputs/targets
            tgt_input = tokens[:, :-1]
            tgt_expected = tokens[:, 1:]
            
            optimizer.zero_grad()
            outputs = model(images, tgt_input) 
            
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_expected.reshape(-1))
            loss.backward()
            
            # Gradient clipping protects the transferred CNN weights from sudden spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"E{epoch} B{batch_idx} | Loss: {loss.item():.4f}")
                
        torch.save(model.state_dict(), f"ocr_hybrid_transformer_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_transformer()
