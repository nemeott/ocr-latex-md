
from datasets import load_dataset


def load_math_writing(split, val_ratio=0.1, seed=42):

    if split not in ("train", "validation"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'validation'.")

    ds = load_dataset("LiamYo/MathWritingHandwritten", split=split)
    
    return ds

def load_iam_lines(split):


    if split not in ("train", "validation", "test"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'validation', or 'test'.")

    ds = load_dataset("Teklia/IAM-line", split=split)


    return ds
