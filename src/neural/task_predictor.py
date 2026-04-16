# LLM help with reformatting (I use camel casing usually, but it looks better with snake casing)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from train_2_rnns import C2RNN, SubwordVocab, UnifiedOCRDataset, collate_fn, cycle_loader

os.environ["HIP_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
torch.backends.cudnn.enabled = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the CNN from the other models to make training go faster
class DomainRouter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone 
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2) # [0: Text, 1: LaTeX]
        )

    def forward(self, x):
        features = self.backbone(x).squeeze(2).permute(0, 2, 1)
        pooled = torch.mean(features, dim=1) 
        return self.classifier(pooled)

# Validation function written with help of LLM
def validate_router(model, text_loader, latex_loader, steps=10):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Check a few batches of each to get a solid accuracy percentage
        for _ in range(steps):
            imgs_t, _, _ = next(text_loader)
            imgs_l, _, _ = next(latex_loader)
            
            batch_imgs = torch.cat([imgs_t, imgs_l]).to(device)
            batch_labels = torch.cat([torch.zeros(len(imgs_t)), torch.ones(len(imgs_l))]).long().to(device)
            
            logits = model(batch_imgs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
    model.train()
    return (correct / total) * 100

def train_router():
    ST = ["\\frac{", "^{", "_{", "}^{", "\\sqrt{", "\\begin{matrix}", "\\end{matrix}", 
          "\\alpha", "\\beta", "\\gamma", "\\theta", "\\sum_{", "\\int_{", "\\rightarrow"]
    
    # Load Train and Validation splits
    latex_ds = load_dataset("LiamYo/MathWritingHandwritten", split="train")
    text_ds = load_dataset("Teklia/IAM-line", split="train")
    latex_val_ds = load_dataset("LiamYo/MathWritingHandwritten", split="validation")
    text_val_ds = load_dataset("Teklia/IAM-line", split="validation")
    
    vocab = SubwordVocab(ST)
    vocab.build_vocab([latex_ds, text_ds])

    full_model = C2RNN(len(vocab)).to(device)
    checkpoint_path = "ocr_c2rnn_epoch_9.pth"
    if os.path.exists(checkpoint_path):
        full_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded {checkpoint_path} for Router training.")
    
    for param in full_model.cnn.parameters():
        param.requires_grad = False

    router = DomainRouter(full_model.cnn).to(device)
    optimizer = optim.Adam(router.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize((128, 1024)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Train Loaders. Probably could use a larger batch_size now that the model is not serial, but whatever.
    loader_text = DataLoader(UnifiedOCRDataset(text_ds, vocab, transform), batch_size=32, shuffle=True, collate_fn=collate_fn)
    loader_latex = DataLoader(UnifiedOCRDataset(latex_ds, vocab, transform), batch_size=32, shuffle=True, collate_fn=collate_fn)
    iter_text = iter(cycle_loader(loader_text))
    iter_latex = iter(cycle_loader(loader_latex))

    # Val Loaders
    val_loader_text = iter(cycle_loader(DataLoader(UnifiedOCRDataset(text_val_ds, vocab, transform), batch_size=32, collate_fn=collate_fn)))
    val_loader_latex = iter(cycle_loader(DataLoader(UnifiedOCRDataset(latex_val_ds, vocab, transform), batch_size=32, collate_fn=collate_fn)))

    # Written with help of LLM
    print("Training Router")
    router.train()
    for step in range(101): # Ended up going back to 100 because overfitting occurred after that
        imgs_t, _, _ = next(iter_text)
        imgs_l, _, _ = next(iter_latex)
        
        batch_imgs = torch.cat([imgs_t, imgs_l]).to(device)
        batch_labels = torch.cat([torch.zeros(len(imgs_t)), torch.ones(len(imgs_l))]).long().to(device)
        
        optimizer.zero_grad()
        logits = router(batch_imgs)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            # Calculate Training Accuracy for the current batch
            preds = torch.argmax(logits, dim=1)
            train_acc = (preds == batch_labels).float().mean().item() * 100
            
            # Calculate Validation Accuracy
            val_acc = validate_router(router, val_loader_text, val_loader_latex)
            
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:6.2f}% | Val Acc: {val_acc:6.2f}%")

    torch.save(router.state_dict(), "router_final.pth")
    print("\nRouter Saved! System ready for full pipeline.")

if __name__ == "__main__":
    train_router()
