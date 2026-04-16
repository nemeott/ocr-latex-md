# LLM help with reformatting (I use camel casing usually, but it looks better with snake casing)
import os
import io
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from jiwer import cer, wer
import collections

os.environ["HIP_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8" # This I used an LLM for; was getting wonky errors because ROCm doesn't like me.
torch.backends.cudnn.enabled = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# With help from LLM: "Help me build a joint vocabulary that first adds in the original char vocab and second the new ones helpful for LaTeX.
class SubwordVocab:
    def __init__(self, special_tokens):
        self.char2idx = {"[blank]": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        # Sort for regex matching only (greedy match longest first)
        self.special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        self.pattern = re.compile("|".join([re.escape(t) for t in self.special_tokens_sorted]) + "|.")
        self.raw_special_tokens = special_tokens

    def build_vocab(self, datasets):
        # FIRST: Add the original 1-char tokens (0-94) to match the old model
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
        
        # SECOND: Append the new structural subwords at the end (95+)
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

# This was from an LLM. "Help me create a beam search decoder that also does pruning. This is the relevant code for my CRNN model: [...]."
class PrunedBeamSearchDecoder:
    def __init__(self, vocab, beam_width=2, top_k=3):
        self.vocab = vocab
        self.beam_width = beam_width
        self.top_k = top_k 

    def decode(self, log_probs):
        T, V = log_probs.shape
        beams = {(): (0.0, -float('inf'))}
        for t in range(T):
            new_beams = collections.defaultdict(lambda: (-float('inf'), -float('inf')))
            top_indices = np.argsort(log_probs[t])[-self.top_k:]
            for prefix, (p_b, p_nb) in beams.items():
                for char_idx in top_indices:
                    p_char = log_probs[t, char_idx]
                    if char_idx == 0:
                        curr_b, curr_nb = new_beams[prefix]
                        new_beams[prefix] = (np.logaddexp(curr_b, np.logaddexp(p_b, p_nb) + p_char), curr_nb)
                    else:
                        new_prefix = prefix + (char_idx,)
                        curr_b, curr_nb = new_beams[new_prefix]
                        if prefix and char_idx == prefix[-1]:
                            new_beams[new_prefix] = (curr_b, np.logaddexp(curr_nb, p_b + p_char))
                            old_b, old_nb = new_beams[prefix]
                            new_beams[prefix] = (old_b, np.logaddexp(old_nb, p_nb + p_char))
                        else:
                            new_beams[new_prefix] = (curr_b, np.logaddexp(curr_nb, np.logaddexp(p_b, p_nb) + p_char))
            beams = dict(sorted(new_beams.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]), reverse=True)[:self.beam_width])
        best_prefix = max(beams.keys(), key=lambda x: np.logaddexp(beams[x][0], beams[x][1]))
        return "".join([self.vocab.idx2char[idx] for idx in best_prefix if idx > 3])

# This is the same as what was used for building the whole dataset in a prior program.
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
    # "How should I write the forward method given my model structure? [...]"
    def forward(self, x):
        features = self.cnn(x).squeeze(2).permute(0, 2, 1) 
        recurrent, _ = self.rnn(features)
        return self.fc(recurrent).permute(1, 0, 2).log_softmax(2)

def run_subword_evaluation(model_path, special_tokens, myToken=""):
    print("Loading datasets for vocab construction...")
    latex_train = load_dataset("LiamYo/MathWritingHandwritten", split="train", token=myToken)
    text_train = load_dataset("Teklia/IAM-line", split="train", token=myToken)
    latex_val = load_dataset("LiamYo/MathWritingHandwritten", split="validation", token=myToken)
    text_val = load_dataset("Teklia/IAM-line", split="validation", token=myToken)
    
    vocab = SubwordVocab(special_tokens)
    vocab.build_vocab([latex_train, text_train])

    transform = transforms.Compose([
        transforms.Resize((128, 1024)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    val_set = ConcatDataset([UnifiedOCRDataset(text_val, vocab, transform), UnifiedOCRDataset(latex_val, vocab, transform)])
    loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = CRNN(len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    decoder = PrunedBeamSearchDecoder(vocab, beam_width=2, top_k=3)
    all_gt, all_pred = [], []

    # This part was with help from an LLM, specifically "I know how to get a single output. How do I do it over the whole datasets? Code for a single output: [...]."
    print(f"Starting Subword Beam Search Inference on {device}...")
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(loader):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            
            for b in range(outputs.shape[1]):
                all_pred.append(decoder.decode(outputs[:, b, :]))
                
            current_pos = 0
            for length in target_lengths:
                gt_indices = targets[current_pos : current_pos + length]
                # Note: Ground truth reconstruction must account for multi-char tokens
                gt_str = "".join([vocab.idx2char[idx.item()] for idx in gt_indices if idx.item() > 3])
                all_gt.append(gt_str)
                current_pos += length

    avg_cer, avg_wer = cer(all_gt, all_pred), wer(all_gt, all_pred)
    exact_acc = accuracy_score(all_gt, all_pred)

    # Asked LLM to make things look fancier
    print("\n" + "="*60)
    print(f"SUBWORD BEAM SEARCH REPORT: {os.path.basename(model_path)}")
    print("="*60)
    print(f"{'Atomic Accuracy (Exact)':<35} | {exact_acc:.4f}")
    print(f"{'Character Error Rate (CER)':<35} | {avg_cer:.4f}")
    print(f"{'Word Error Rate (WER)':<35} | {avg_wer:.4f}")
    print("="*60)

if __name__ == "__main__":
    # Ensure this matches the fine-tuning script exactly
    ST = [
        "\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
        "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"
    ]
    # Update to latest fine-tuned checkpoint
    run_subword_evaluation("ocr_subword_v2_epoch_9.pth", ST)
