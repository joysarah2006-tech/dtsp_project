# emotion_ravdess.py
import argparse
import numpy as np
import os
from collections import Counter
from ravdess_database import (
    build_database, load_database, read_audio, do_fft,
    extract_features, plot_signal, FEATURE_NAMES
)


K = 5

def train(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-9
    return mean, std

def knn_predict(new_sample, database, files, k=K):
    d = np.linalg.norm(database - new_sample, axis=1)
    idx = np.argsort(d)[:k]
    votes = Counter(files[i] for i in idx)
    return votes.most_common(1)[0][0], idx, d[idx]

import matplotlib.pyplot as plt
def predict(new_file, mean, std, database, emotions, files):
    sr, x = read_audio(new_file)
    freqs, mag = do_fft(x, sr)
    feats, smooth, top3, valley = extract_features(freqs, mag, x)
    new_sample = (np.array(feats) - mean) / std

    # Run k-NN
    best, idx, dists = knn_predict(new_sample, database, files)

    # Print features of the new audio
    print(f"\nNEW FILE: {new_file}")
    print("Extracted features:")
    for name, value in zip(FEATURE_NAMES, feats):
        print(f"  {name:<15}{value:10.3f}")

    # Print closest matches
    print("\nClosest matches in the database:")
    for i, d in zip(idx, dists):
        print(f"  {emotions[i]:<10} {os.path.basename(files[i]):<30} distance = {d:.2f}")

    # Print predicted emotion label
    predicted_emotion = emotions[idx[0]]
    print(f"\n>>> PREDICTED EMOTION: {predicted_emotion.upper()}")

    # Side-by-side plots
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_signal(new_file, sr, x, freqs, smooth, top3, valley,
                axes=(axes[0,0], axes[1,0]), title="New Audio")

    sr2, x2 = read_audio(best)
    f2, m2 = do_fft(x2, sr2)
    feats2, s2, t2, v2 = extract_features(f2, m2, x2)
    plot_signal(best, sr2, x2, f2, s2, t2, v2,
                axes=(axes[0,1], axes[1,1]), title="Closest Match")

    plt.suptitle(f"Predicted Emotion: {predicted_emotion.upper()}", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAVDESS Emotion Detection")
    parser.add_argument("cmd", choices=["build", "predict"], help="build database or predict emotion")
    parser.add_argument("file", nargs="?", help="path to audio file for prediction")
    args = parser.parse_args()

    if args.cmd == "build":
        build_database()
    elif args.cmd == "predict":
        if not args.file:
            parser.error("predict requires an audio file path")
        emotions, actors, files, data = load_database()
        mean, std = train(data)
        database = (data - mean) / std
        # Pass emotions AND files into predict
        predict(args.file, mean, std, database, emotions, files)
