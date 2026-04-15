import os
import io
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
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
        # First, add the original 1-char tokens (0-94) to match the old model
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
        
        # Second, append the new structural subwords at the end (95+)
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

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
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
        self.rnn = nn.LSTM(512, hidden_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        features = self.cnn(x).squeeze(2).permute(0, 2, 1) 
        recurrent, _ = self.rnn(features)
        return self.fc(recurrent).permute(1, 0, 2).log_softmax(2)

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

if __name__ == "__main__":
    CHECKPOINT_TO_CONVERT = "ocr_ctc_epoch_239.pth" 
    ST = ["\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
          "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"]
    
    latex_ds = load_dataset("LiamYo/MathWritingHandwritten", split="train")
    text_ds = load_dataset("Teklia/IAM-line", split="train")
    
    vocab = SubwordVocab(ST)
    vocab.build_vocab([latex_ds, text_ds])
    
    model = CRNN(len(vocab)).to(device)

    if os.path.exists(CHECKPOINT_TO_CONVERT):
        print(f"Executing Partial Weight Transfer from {CHECKPOINT_TO_CONVERT}...")
        old_sd = torch.load(CHECKPOINT_TO_CONVERT, map_location=device)
        
        # Load Backbone (CNN + RNN)
        backbone_sd = {k: v for k, v in old_sd.items() if "fc" not in k}
        model.load_state_dict(backbone_sd, strict=False)
        
        # Partially Load FC layer
        with torch.no_grad():
            old_w, old_b = old_sd['fc.weight'], old_sd['fc.bias']
            num_transfer = min(old_w.size(0), model.fc.weight.size(0))
            model.fc.weight[:num_transfer].copy_(old_w[:num_transfer])
            model.fc.bias[:num_transfer].copy_(old_b[:num_transfer])
        print(f"Successfully transferred {num_transfer} character weights.")
    
    transform = transforms.Compose([
        transforms.Resize((128, 1024)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = ConcatDataset([UnifiedOCRDataset(text_ds, vocab, transform), UnifiedOCRDataset(latex_ds, vocab, transform)])
    wts = ([1.0/len(text_ds)] * len(text_ds)) + ([1.0/len(latex_ds)] * len(latex_ds))
    loader = DataLoader(train_set, batch_size=32, sampler=WeightedRandomSampler(wts, 25000), collate_fn=collate_fn, num_workers=4)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5) # Lower LR for fine-tuning
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    print(f"Starting fine-tuning on {device}...")
    for epoch in range(20):
        for i, (imgs, tgs, t_lens) in enumerate(loader):
            imgs, tgs = imgs.to(device), tgs.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            i_lens = torch.full((imgs.size(0),), outs.size(0), dtype=torch.long)
            loss = criterion(outs, tgs, i_lens, t_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if i % 100 == 0:
                with torch.no_grad():
                    pred = vocab.decode(outs[:, 0, :].argmax(1))
                    print(f"E{epoch} B{i} | Loss: {loss.item():.4f} | Pred: {pred[:50]}")
        
        torch.save(model.state_dict(), f"ocr_subword_v2_epoch_{epoch}.pth")