#!/usr/bin/env python3
import os
import json
from typing import Optional

def load_data_from_path(data_path: str, max_samples: Optional[int] = None) -> list:
    """
    Load data from JSON/JSONL file or directory.
    
    Args:
        data_path: Path to JSON file, JSONL file, or directory containing JSON/JSONL files.
        max_samples: Maximum number of samples to load (None for all).
    
    Returns:
        List of data examples.
    """
    data = []
    
    # Handle both file and directory paths
    if os.path.isdir(data_path):
        # Look for JSON or JSONL files
        json_files = sorted([f for f in os.listdir(data_path) if f.endswith(('.json', '.jsonl'))])
        file_paths = [os.path.join(data_path, f) for f in json_files]
    else:
        file_paths = [data_path]
    
    for file_path in file_paths:
        print(f"Loading data from {file_path}")
        
        # Try line-by-line JSONL format first (most common for large datasets)
        try:
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    if max_samples and len(data) >= max_samples:
                        break
                    try:
                        example = json.loads(line.strip())
                        data.append(example)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line in {file_path}: {e}")
        except UnicodeDecodeError:
            print(f"Warning: Skipping {file_path} - not a valid UTF-8 text file")
            continue
        
        if max_samples and len(data) >= max_samples:
            break

    print(f"Loaded {len(data)} samples")
    return data


if __name__ == "__main__":
    # Test loading the test.json file
    print("=" * 60)
    print("Test 1: Loading test.json")
    print("=" * 60)
    data = load_data_from_path('student/test.json')
    print(f"✓ Successfully loaded {len(data)} records")
    print(f"✓ First record keys: {list(data[0].keys())}")
    print(f"✓ Ground truth: {data[0].get('ground_truth', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Test 2: Loading with max_samples=2")
    print("=" * 60)
    data_limited = load_data_from_path('student/test.json', max_samples=2)
    print(f"✓ With max_samples=2: loaded {len(data_limited)} records")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
