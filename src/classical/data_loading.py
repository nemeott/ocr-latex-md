import os

from datasets import load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN") # you need to make sure that this variable is set up in the .env file 


def load_math_writing(split): 

    """
    This function helps us to load the math writing dataset from the hugging face dataset.

    Args:
        split: The split of the dataset to load
    Returns:
        The dataset
    """

    if split not in ("train", "validation", "test"):

        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'validation', or 'test'.")

    if split not in ("train", "validation"):


        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'validation'.")

    return load_dataset("LiamYo/MathWritingHandwritten", split=split, token=HF_TOKEN, num_proc=2)


def load_iam_lines(split):

    """
    This function helps us to load the iam lines dataset from the hugging face dataset.
    
    Args:
        split: The split of the dataset to load
    Returns:
        The dataset
    """

    if split not in ("train", "validation", "test"):

        
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'validation', or 'test'.")

    return load_dataset("Teklia/IAM-line", split=split, token=HF_TOKEN, num_proc=2)
