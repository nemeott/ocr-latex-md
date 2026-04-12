import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

os.environ["HIP_VISIBLE_DEVICES"] = "1" # For some reason, ROCm lists my iGPU first
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
torch.backends.cudnn.enabled = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharVocab:
    def __init__(self):
        # Index 0 is the [blank] token for CTC
        self.char2idx = {"[blank]": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        
    def build_vocab(self, datasets):
        unique_chars = set()
        for ds in datasets:
            label_key = next(c for c in ['txt', 'latex', 'text', 'formula'] if c in ds.column_names)
            texts = ds[label_key]
            for text in texts:
                if text: unique_chars.update(text)
        for char in sorted(list(unique_chars)):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
        print(f"CTC Vocab Size: {len(self.char2idx)}")

    def encode(self, text):
        return [self.char2idx.get(c, 1) for c in text]

    def decode(self, indices):
        res = []
        for i in range(len(indices)):
            # Convert tensor to int for dict lookup
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            if idx != 0 and (i == 0 or idx != (indices[i-1].item() if torch.is_tensor(indices[i-1]) else indices[i-1])):
                if idx > 3:
                    res.append(self.idx2char[idx])
        return "".join(res)

    def __len__(self):
        return len(self.char2idx)

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
        output = self.fc(recurrent)
        return output.permute(1, 0, 2).log_softmax(2)

if __name__ == "__main__":
    myToken = ""
    latex_ds = load_dataset("LiamYo/MathWritingHandwritten", split="train", token=myToken)
    text_ds = load_dataset("Teklia/IAM-line", split="train", token=myToken)
    
    vocab = CharVocab()
    vocab.build_vocab([latex_ds, text_ds])
    
    transform = transforms.Compose([
        transforms.Resize((128, 1024)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = ConcatDataset([UnifiedOCRDataset(text_ds, vocab, transform), UnifiedOCRDataset(latex_ds, vocab, transform)])
    weights = ([1.0/len(text_ds)] * len(text_ds)) + ([1.0/len(latex_ds)] * len(latex_ds))
    sampler = WeightedRandomSampler(weights, num_samples=25000, replacement=True)
    loader = DataLoader(train_set, batch_size=32, sampler=sampler, collate_fn=collate_fn, num_workers=4)
    
    model = CRNN(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    print(f"Training on {device}")
    for epoch in range(240):
        for i, (images, targets, target_lengths) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)
            
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if i % 100 == 0:
                try:
                    with torch.no_grad():
                        sample_out = outputs[:, 0, :].argmax(1)
                        decoded = vocab.decode(sample_out)
                        print(f"E{epoch} B{i} | Loss: {loss.item():.4f} | Pred: {decoded[:40]}")
                except Exception:
                    print(f"E{epoch} B{i} | Loss: {loss.item():.4f} | (Decode Failed)")

        torch.save(model.state_dict(), f"ocr_ctc_epoch_{epoch}.pth")