import os
import io
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

os.environ["HIP_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
torch.backends.cudnn.enabled = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubwordVocab:
    def __init__(self, special_tokens):
        self.char2idx = {"[blank]": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        # Sort for regex matching only (greedy match longest first)
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

    def decode(self, indices):
        res = []
        for i in range(len(indices)):
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            if idx != 0 and (i == 0 or idx != (indices[i-1].item() if torch.is_tensor(indices[i-1]) else indices[i-1])):
                if idx > 3: res.append(self.idx2char[idx])
        return "".join(res)

    def __len__(self): return len(self.char2idx)

class C2RNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        # Shared CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), 
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)) 
        )
        
        # Domain 1: Text Decoder
        self.rnn_text = nn.LSTM(512, hidden_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.3)
        self.fc_text = nn.Linear(hidden_dim * 2, vocab_size)
        
        # Domain 2: LaTeX Decoder
        self.rnn_latex = nn.LSTM(512, hidden_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.3)
        self.fc_latex = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, domain="text"):
        features = self.cnn(x).squeeze(2).permute(0, 2, 1) 
        
        if domain == "text":
            recurrent, _ = self.rnn_text(features)
            return self.fc_text(recurrent).permute(1, 0, 2).log_softmax(2)
        elif domain == "latex":
            recurrent, _ = self.rnn_latex(features)
            return self.fc_latex(recurrent).permute(1, 0, 2).log_softmax(2)
        else:
            raise ValueError(f"Unknown domain: {domain}")

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
        tokens = torch.tensor(self.vocab.encode(item[self.label_key]))
        return image, tokens, torch.tensor(len(tokens))

def collate_fn(batch):
    images, tokens, lengths = zip(*batch)
    return torch.stack(images), torch.cat(tokens), torch.stack(lengths)

# Helper function to cycle dataloaders of unequal length seamlessly
def cycle_loader(iterable):
    while True:
        for x in iterable:
            yield x

if __name__ == "__main__":
    CHECKPOINT_TO_CONVERT = "ocr_subword_v2_epoch_9.pth" 
    ST = ["\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
          "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"]
    
    latex_ds = load_dataset("LiamYo/MathWritingHandwritten", split="train")
    text_ds = load_dataset("Teklia/IAM-line", split="train")
    
    vocab = SubwordVocab(ST)
    vocab.build_vocab([latex_ds, text_ds])
    
    model = C2RNN(len(vocab)).to(device)

    # --- WEIGHT TRANSFER & DUPLICATION LOGIC ---
    if os.path.exists(CHECKPOINT_TO_CONVERT):
        print(f"Executing Specialist Weight Transfer from {CHECKPOINT_TO_CONVERT}...")
        old_sd = torch.load(CHECKPOINT_TO_CONVERT, map_location=device)
        
        # 1. Load CNN Backbone (CNN weights are named 'cnn.X.weight' in your architecture)
        cnn_sd = {k: v for k, v in old_sd.items() if k.startswith("cnn")}
        model.load_state_dict(cnn_sd, strict=False)
        
        # 2. Duplicate old RNN weights into BOTH new specialist RNNs
        rnn_sd_text = {k.replace("rnn", "rnn_text"): v for k, v in old_sd.items() if k.startswith("rnn")}
        rnn_sd_latex = {k.replace("rnn", "rnn_latex"): v for k, v in old_sd.items() if k.startswith("rnn")}
        model.load_state_dict(rnn_sd_text, strict=False)
        model.load_state_dict(rnn_sd_latex, strict=False)
        
        # 3. Load FC weights into BOTH new FC layers
        with torch.no_grad():
            old_w, old_b = old_sd['fc.weight'], old_sd['fc.bias']
            
            # Transfer to Text Specialist
            model.fc_text.weight.copy_(old_w)
            model.fc_text.bias.copy_(old_b)
            
            # Transfer to LaTeX Specialist
            model.fc_latex.weight.copy_(old_w)
            model.fc_latex.bias.copy_(old_b)
            
        print(f"Successfully duplicated subword weights from Epoch 9 into C2RNN.")
    else:
        print(f"WARNING: {CHECKPOINT_TO_CONVERT} not found. Training from scratch.")
    
    transform = transforms.Compose([
        transforms.Resize((128, 1024)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create separate loaders to prevent batch contamination
    loader_text = DataLoader(UnifiedOCRDataset(text_ds, vocab, transform), batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    loader_latex = DataLoader(UnifiedOCRDataset(latex_ds, vocab, transform), batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    iter_text = iter(cycle_loader(loader_text))
    iter_latex = iter(cycle_loader(loader_latex))
    
    # Calculate total steps per epoch based on the larger dataset
    steps_per_epoch = max(len(loader_text), len(loader_latex))
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5) 
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    print(f"Starting C2RNN fine-tuning on {device}...")
    for epoch in range(10):
        for step in range(steps_per_epoch):
            
            # Only train text every 20 steps to match the dataset sizes better
            if step % 20 == 0: 
                imgs_t, tgs_t, t_lens_t = next(iter_text)
                imgs_t, tgs_t = imgs_t.to(device), tgs_t.to(device)
                
                optimizer.zero_grad()
                outs_t = model(imgs_t, domain="text")
                i_lens_t = torch.full((imgs_t.size(0),), outs_t.size(0), dtype=torch.long)
                loss_t = criterion(outs_t, tgs_t, i_lens_t, t_lens_t)
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            # LaTeX
            imgs_l, tgs_l, t_lens_l = next(iter_latex)
            imgs_l, tgs_l = imgs_l.to(device), tgs_l.to(device)
            
            optimizer.zero_grad()
            outs_l = model(imgs_l, domain="latex")
            i_lens_l = torch.full((imgs_l.size(0),), outs_l.size(0), dtype=torch.long)
            loss_l = criterion(outs_l, tgs_l, i_lens_l, t_lens_l)
            loss_l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 100 == 0:
                with torch.no_grad():
                    pred_t = vocab.decode(outs_t[:, 0, :].argmax(1))
                    pred_l = vocab.decode(outs_l[:, 0, :].argmax(1))
                    print(f"E{epoch} B{step} | T-Loss: {loss_t.item():.4f} | L-Loss: {loss_l.item():.4f}")
                    print(f"  -> T-Pred: {pred_t[:40]}...")
                    print(f"  -> L-Pred: {pred_l[:40]}...")
        
        torch.save(model.state_dict(), f"ocr_c2rnn_epoch_{epoch}.pth")