import os
import pickle


def save_cache(obj, filepath):
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

    print(f"[CACHE] Saved: {filepath}")


def load_cache(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    print(f"[CACHE] Loaded: {filepath}")
    return obj


def cache_exists(filepath):
    return os.path.exists(filepath)