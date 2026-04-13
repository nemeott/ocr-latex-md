"""Symbol mapping utilities for EMNIST and HASYv2.

Provides functions to load label-to-character/LaTeX mappings for use in OCR postprocessing.
"""

import csv


def load_emnist_mapping(path):
    """Loads the EMNIST label-to-character mapping from a mapping file.
    Each line in the file should be: <label> <unicode_int>
    Returns a dict: {label (int): character (str)}
    """
    mapping = {}
    with open(path) as f:
        for line in f:
            label, unicode_int = line.strip().split()
            mapping[int(label)] = chr(int(unicode_int))
    return mapping


def load_hasy_mapping(path):
    """Loads the HASYv2 symbol_id-to-LaTeX mapping from hasy-symbols.csv.
    CSV columns: symbol_id,symbol,latex
    Returns a dict: {symbol_id (str): latex (str)}
    """
    mapping = {}
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mapping[row["symbol_id"]] = row["latex"]
    return mapping


def map_symbol_value(symbol, emnist_map, hasy_map):
    """Given a Symbol object and the loaded mappings, returns the human-readable value.

    - For TEXT: uses EMNIST mapping (label as int)
    - For MATH: uses HASY mapping (symbol_id as str)
    """
    symbol_type = getattr(symbol.type, "name", None) or str(symbol.type)
    val = symbol.value
    result = "?"
    if symbol_type == "TEXT":
        try:
            emnist_val = emnist_map.get(int(val), None)
        except Exception:
            emnist_val = None
        hasy_val = hasy_map.get(str(val), None)
        if emnist_val is not None:
            result = emnist_val
        elif hasy_val is not None:
            result = hasy_val
    else:
        hasy_val = hasy_map.get(str(val), None)
        if hasy_val is not None:
            result = hasy_val

    # Always return a string
    return str(result) if result is not None else "?"
