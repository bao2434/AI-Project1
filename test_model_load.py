"""Test loading models/value_model.pt directly with zipfile and struct."""
import os
import sys
from pathlib import Path
import struct
import zipfile
import math
import time

def find_model_path():
    candidates = [
        "models/value_model.pt",
        "models/value_models.pt",
        Path(__file__).resolve().parent / "models" / "value_model.pt",
        Path(__file__).resolve().parent / "models" / "value_models.pt",
    ]
    for p in candidates:
        if Path(p).is_file():
            return Path(p)
    return None

def test_zip_load():
    path = find_model_path()
    assert path is not None, f"Model file not found in candidates"
    print(f"Found model at: {path}")
    
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        print(f"Archive files: {len(names)} items")
        prefix = ""
        for name in names:
            if name.endswith("data/0"):
                prefix = name[:-len("data/0")]
                break
        print(f"Detected prefix: '{prefix}'")
        
        parts = []
        for i in range(10):
            entry = f"{prefix}data/{i}"
            raw = z.read(entry)
            floats = struct.unpack(f"<{len(raw)//4}f", raw)
            parts.append(floats)
            print(f"Layer {i}: {len(floats)} floats")
            
    print("Successfully loaded all 10 layers!")
    return parts

if __name__ == "__main__":
    test_zip_load()
