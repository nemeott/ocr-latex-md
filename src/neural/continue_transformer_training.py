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
        
        # Transformer Decoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_1d = PositionalEncoding1D(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, images, tgt):
        features = self.cnn(images) 
        memory = self.pos_encoder_2d(features) 
        
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

# ==========================================
# 5. TRAINING LOOP
# ==========================================
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
    
    # Load Epoch 4 Weights directly
    checkpoint_path = "ocr_hybrid_transformer_epoch_4.pth"
    print(f"\nLoading saved weights from {checkpoint_path}...")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Could not find {checkpoint_path}. Ensure it is in the same directory.")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Re-initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # Total epochs is 50 based on your last script configuration
    TOTAL_EPOCHS = 50
    START_EPOCH = 5
    
    # Re-initialize Scheduler for the full 50 epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, steps_per_epoch=len(loader), epochs=TOTAL_EPOCHS)
    
    # Fast-forward the scheduler to the end of Epoch 4
    # OneCycleLR calculates its internal position based on the number of `.step()` calls.
    print(f"Fast-forwarding the OneCycleLR scheduler by {START_EPOCH * len(loader)} steps to match the current epoch...")
    for _ in range(START_EPOCH * len(loader)):
        scheduler.step()

    print(f"Resuming Hybrid Transformer training from Epoch {START_EPOCH} to {TOTAL_EPOCHS}...")
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, tokens) in enumerate(loader):
            images, tokens = images.to(device), tokens.to(device)
            
            tgt_input = tokens[:, :-1]
            tgt_expected = tokens[:, 1:]
            
            optimizer.zero_grad()
            outputs = model(images, tgt_input) 
            
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_expected.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"E{epoch} B{batch_idx} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                
        # Save standard weights for the evaluation script
        torch.save(model.state_dict(), f"ocr_hybrid_transformer_epoch_{epoch}.pth")
        
        # Save full checkpoint with optimizer and scheduler states for future resumption
        full_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item()
        }
        torch.save(full_checkpoint, f"checkpoint_epoch_{epoch}.pt")

if __name__ == "__main__":
    train_transformer()