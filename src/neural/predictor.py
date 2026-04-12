import os
import re
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

os.environ["HIP_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
torch.backends.cudnn.enabled = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubwordVocab:
    def __init__(self, special_tokens):
        self.char2idx = {"[blank]": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
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
                idx = len(self.char2idx); self.char2idx[char] = idx; self.idx2char[idx] = char
        for t in self.raw_special_tokens:
            if t not in self.char2idx:
                idx = len(self.char2idx); self.char2idx[t] = idx; self.idx2char[idx] = t

    def decode(self, indices):
        res = []
        prev_idx = -1
        for i in range(len(indices)):
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            if idx != 0 and idx != prev_idx:
                if idx in self.idx2char:
                    char = self.idx2char[idx]
                    if char not in ["[blank]", "<UNK>", "<SOS>", "<EOS>"]:
                        res.append(char)
            prev_idx = idx
        return "".join(res)
    def __len__(self): return len(self.char2idx)

class C2RNN(nn.Module):
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
        self.rnn_text = nn.LSTM(512, hidden_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.3)
        self.fc_text = nn.Linear(hidden_dim * 2, vocab_size)
        self.rnn_latex = nn.LSTM(512, hidden_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.3)
        self.fc_latex = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, domain="text"):
        features = self.cnn(x).squeeze(2).permute(0, 2, 1) 
        rnn = self.rnn_text if domain == "text" else self.rnn_latex
        fc = self.fc_text if domain == "text" else self.fc_latex
        recurrent, _ = rnn(features)
        return fc(recurrent).permute(1, 0, 2).log_softmax(2)

class DomainRouter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone 
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, x):
        features = self.backbone(x).squeeze(2).permute(0, 2, 1)
        return self.classifier(torch.mean(features, dim=1))

def segment_natural_ratio(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return []
    
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    clean_bw = cv2.bitwise_not(thresh)

    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 5))
    mask = cv2.dilate(thresh, line_kernel, iterations=1)
    
    projection = np.sum(mask, axis=1)
    is_text = projection > (np.max(projection) * 0.1) 
    starts, ends = np.where(np.diff(np.pad(is_text, (1, 1), mode='constant').astype(int)) == 1)[0], \
                   np.where(np.diff(np.pad(is_text, (1, 1), mode='constant').astype(int)) == -1)[0]

    line_images = []
    for s, e in zip(starts, ends):
        if (e - s) < 25: continue
        
        # Horizontal Tight Crop to find content bounds
        line_mask = thresh[s:e, :]
        cols = np.sum(line_mask, axis=0)
        if np.max(cols) > 0:
            x_s, x_e = np.where(cols > 0)[0][0], np.where(cols > 0)[0][-1]
            # Extra horizontal padding for context
            x_s, x_e = max(0, x_s - 30), min(img.shape[1], x_e + 30)
            
            line_crop = clean_bw[s:e, x_s:x_e]
            line_pil = Image.fromarray(line_crop).convert("L")
            w, h = line_pil.size

            # PRESERVE ASPECT RATIO: Target height is 100px (centered in 128px)
            target_h = 100 
            ratio = target_h / h
            new_w = min(1024, int(w * ratio))
            
            # Resize keeping the ratio
            line_pil = line_pil.resize((new_w, target_h), Image.Resampling.LANCZOS)
            
            # Paste onto standard 1024x128 canvas
            canvas = Image.new('L', (1024, 128), color=255)
            y_offset = (128 - target_h) // 2
            canvas.paste(line_pil, (0, y_offset))
            line_images.append(canvas)
            
    return line_images

def run_final_presentation_pipeline(image_path):
    ST = ["\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
          "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"]
    
    print("--- Loading Data & Vocab ---")
    latex_ds = load_dataset("LiamYo/MathWritingHandwritten", split="train")
    text_ds = load_dataset("Teklia/IAM-line", split="train")
    vocab = SubwordVocab(ST); vocab.build_vocab([latex_ds, text_ds])
    
    print("--- Loading Models ---")
    model = C2RNN(len(vocab)).to(device)
    model.load_state_dict(torch.load("ocr_c2rnn_epoch_9.pth", map_location=device))
    model.eval()
    router = DomainRouter(model.cnn).to(device)
    router.load_state_dict(torch.load("router_final.pth", map_location=device))
    router.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    lines = segment_natural_ratio(image_path)
    
    # Result Composite
    final_canvas = Image.new('RGB', (1024, len(lines) * 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(final_canvas)
    try: font = ImageFont.truetype("arial.ttf", 26)
    except: font = ImageFont.load_default()

    print(f"--- Processing {len(lines)} lines ---")
    with torch.no_grad():
        for i, img in enumerate(lines):
            img_t = transform(img).unsqueeze(0).to(device)
            dom_idx = torch.argmax(router(img_t), dim=1).item()
            domain = "text" if dom_idx == 0 else "latex"
            
            out = model(img_t, domain=domain)
            pred = vocab.decode(out.argmax(2).squeeze())
            
            final_canvas.paste(img.convert("RGB"), (0, i * 200))
            color = (0, 0, 180) if domain == "text" else (180, 0, 0)
            draw.text((30, i * 200 + 140), f"[{domain.upper()}] PRED: {pred}", fill=color, font=font)

    final_canvas.save("FINAL_PRESENTATION_DEMO.png")
    print("DONE. Open 'FINAL_PRESENTATION_DEMO.png'.")

if __name__ == "__main__":
    run_final_presentation_pipeline("CLOPEN_TEST_IMAGE.jpg")