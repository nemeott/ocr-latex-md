import os
import re
import torch
import torch.nn as nn
from torchvision import transforms
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
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

def save_result_image(original_img, truth, pred, filename):
    # Resize original for display
    w, h = original_img.size
    display_w = 800
    display_h = int(h * (display_w / w))
    img_resized = original_img.resize((display_w, display_h), Image.LANCZOS)
    
    # Create canvas
    canvas = Image.new('RGB', (display_w, display_h + 150), color=(255, 255, 255))
    canvas.paste(img_resized, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    # Using default font (usually looks okay on most systems)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    draw.text((20, display_h + 30), f"Truth: {truth}", fill=(0, 0, 0), font=font)
    draw.text((20, display_h + 80), f"Pred:  {pred}", fill=(0, 100, 0), font=font)
    
    canvas.save(filename)
    print(f"✅ Saved result to {filename}")

def run_dataset_validation_and_save():
    ST = ["\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
          "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"]
    
    print("--- Loading Datasets ---")
    latex_ds = load_dataset("LiamYo/MathWritingHandwritten", split="train")
    text_ds = load_dataset("Teklia/IAM-line", split="train")
    
    vocab = SubwordVocab(ST)
    vocab.build_vocab([latex_ds, text_ds])
    
    print("--- Loading Model ---")
    model = C2RNN(len(vocab)).to(device)
    model.load_state_dict(torch.load("ocr_c2rnn_epoch_9.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("\n--- RUNNING VALIDATION ---")

    # TEST 1: IAM TEXT
    sample_t = text_ds[42]
    img_t_raw = sample_t['image'].convert("L")
    inp_t = transform(img_t_raw).unsqueeze(0).to(device)
    with torch.no_grad():
        out_t = model(inp_t, domain="text")
        pred_t = vocab.decode(out_t.argmax(2).squeeze())
        save_result_image(img_t_raw, sample_t['text'], pred_t, "IAM_Result.png")

    # TEST 2: MATHWRITING LATEX
    sample_l = latex_ds[100]
    label = sample_l['txt']
    img_obj = sample_l['png']
            
    if img_obj:
        if not isinstance(img_obj, Image.Image):
            img_l_raw = Image.open(io.BytesIO(img_obj)).convert("L")
        else:
            img_l_raw = img_obj.convert("L")
            
        inp_l = transform(img_l_raw).unsqueeze(0).to(device)
        with torch.no_grad():
            out_l = model(inp_l, domain="latex")
            pred_l = vocab.decode(out_l.argmax(2).squeeze())
            save_result_image(img_l_raw, label, pred_l, "MathWriting_Result.png")

if __name__ == "__main__":
    run_dataset_validation_and_save()