# LLM help with reformatting (I use camel casing usually, but it looks better with snake casing)
import os
import io
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
import jiwer
from tqdm import tqdm
from collections import defaultdict

os.environ["HIP_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
torch.backends.cudnn.enabled = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same vocab
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
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
        for t in self.raw_special_tokens:
            if t not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[t] = idx
                self.idx2char[idx] = t
                
    def encode(self, text):
        tokens = self.pattern.findall(text)
        return [self.char2idx.get(t, 1) for t in tokens]

    # Keeping standard decode for fallback, though beam search does its own collapsing
    def decode(self, indices):
        res = []
        for i in range(len(indices)):
            idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
            if idx != 0 and (i == 0 or idx != (indices[i-1].item() if torch.is_tensor(indices[i-1]) else indices[i-1])):
                if idx > 3: res.append(self.idx2char[idx])
        return "".join(res)
    
    def __len__(self):
        return len(self.char2idx)

# Needs to be the same
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
        if domain == "text":
            recurrent, _ = self.rnn_text(features)
            return self.fc_text(recurrent).permute(1, 0, 2).log_softmax(2)
        elif domain == "latex":
            recurrent, _ = self.rnn_latex(features)
            return self.fc_latex(recurrent).permute(1, 0, 2).log_softmax(2)

# Was having issues because of the weird formatting that IAM-line has
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# With help from LLM for beam search decoding
def ctc_beam_search(probs_seq, beam_width=3):
    """
    Standard CTC Prefix Beam Search in linear probability space.
    probs_seq: shape (seq_len, vocab_size)
    """
    T, V = probs_seq.shape
    # beam dict: {prefix_tuple: [prob_blank, prob_non_blank]}
    beam = {tuple(): [1.0, 0.0]}
    
    for t in range(T):
        next_beam = defaultdict(lambda: [0.0, 0.0])
        p = probs_seq[t]
        
        # Optimization: Only search top probabilities to save CPU time
        top_k_probs, top_k_idx = torch.topk(p, min(beam_width + 2, V))
        
        for prefix, (p_b, p_nb) in beam.items():
            p_total = p_b + p_nb
            
            # Extend with blank
            next_beam[prefix][0] += p_total * p[0].item()
            
            # Extend with non-blank characters
            for c_tensor, p_c_tensor in zip(top_k_idx, top_k_probs):
                c = c_tensor.item()
                p_c = p_c_tensor.item()
                
                if c == 0 or p_c < 1e-4: 
                    continue 
                
                prefix_ext = prefix + (c,)
                if len(prefix) > 0 and c == prefix[-1]:
                    # Repeated char: probability splits
                    next_beam[prefix_ext][1] += p_b * p_c
                    next_beam[prefix][1] += p_nb * p_c
                else:
                    # New char
                    next_beam[prefix_ext][1] += p_total * p_c
                    
        # Prune down to beam_width
        best_paths = sorted(next_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:beam_width]
        beam = {k: v for k, v in best_paths}
        
    best_prefix = max(beam.items(), key=lambda x: x[1][0] + x[1][1])[0]
    return best_prefix

# Basically all the same stuff
class UnifiedOCRDataset(Dataset):
    def __init__(self, hf_dataset, vocab, transform=None):
        self.data, self.vocab, self.transform = hf_dataset, vocab, transform
        self.label_key = next(c for c in ['txt', 'latex', 'text', 'formula'] if c in hf_dataset.column_names)
        self.image_key = next(c for c in ['png', 'image', 'img', 'line_image'] if c in hf_dataset.column_names)
        self.valid_indices = [i for i in range(len(self.data)) if str(self.data[i][self.label_key]).strip() != ""]
    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        item = self.data[self.valid_indices[idx]]
        image = item[self.image_key].convert("L")
        if self.transform: image = self.transform(image)
        return image, str(item[self.label_key])

def evaluate_domain_beam(model, dataloader, vocab, domain_name, beam_width=3):
    model.eval()
    exact_matches = 0
    total_samples = 0
    all_preds, all_targets = [], []

    # LLM: "Given my model architecture and the beam search code, help me write a beam search tester on the datasets."
    print(f"\n--- Testing {domain_name.upper()} (Beam Search w={beam_width}) ---")
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            # outs shape: (batch, seq_len, vocab_size) in log_softmax
            outs = model(images, domain=domain_name).permute(1, 0, 2)
            
            # Convert out of log space for the custom CPU beam search
            outs_linear = torch.exp(outs).cpu()
            
            for i in range(images.size(0)):
                seq_probs = outs_linear[i]
                
                # Run the beam search
                best_indices = ctc_beam_search(seq_probs, beam_width=beam_width)
                
                # Beam search naturally collapses repeats and blanks, so we just map to chars
                pred_chars = [vocab.idx2char[idx] for idx in best_indices if idx > 3]
                pred = clean_text("".join(pred_chars))
                target = clean_text(targets[i])
                
                all_preds.append(pred if pred else " ")
                all_targets.append(target)
                if pred == target: exact_matches += 1
                total_samples += 1

    em_rate = (exact_matches / total_samples) * 100
    cer = jiwer.cer(all_targets, all_preds) * 100
    wer = jiwer.wer(all_targets, all_preds) * 100
    
    print(f"\n{domain_name.upper()} FINAL REPORT (BEAM):")
    print(f"Exact Match: {em_rate:.2f}% | CER: {cer:.2f}% | WER: {wer:.2f}%")

# Same main function stuff by this point
if __name__ == "__main__":
    CHECKPOINT = "ocr_c2rnn_epoch_9.pth"
    ST = ["\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
          "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"]
    
    latex_train = load_dataset("LiamYo/MathWritingHandwritten", split="train")
    text_train = load_dataset("Teklia/IAM-line", split="train")
    vocab = SubwordVocab(ST)
    vocab.build_vocab([latex_train, text_train])
    
    model = C2RNN(len(vocab)).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    transform = transforms.Compose([
        transforms.Resize((128, 1024)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    
    latex_test = load_dataset("LiamYo/MathWritingHandwritten", split="validation")
    text_test = load_dataset("Teklia/IAM-line", split="test")

    # Can increase batch size because less expensive to compute vs training
    l_eval = DataLoader(UnifiedOCRDataset(text_test, vocab, transform), batch_size=64, shuffle=False, num_workers=4)
    m_eval = DataLoader(UnifiedOCRDataset(latex_test, vocab, transform), batch_size=64, shuffle=False, num_workers=4)

    # I set beam_width to 3. You can increase this to 5 for slightly better accuracy, but it will run slower.
    evaluate_domain_beam(model, l_eval, vocab, "text", beam_width=3)
    evaluate_domain_beam(model, m_eval, vocab, "latex", beam_width=3)
