# ravdess_database.py
import os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

DATASET_FOLDER = "ravdess"
DATABASE_FILE = "ravdess_features.csv"

FEATURE_NAMES = ["peak1_freq","peak2_freq","peak3_freq","peak2_height","peak3_height",
                 "valley_freq","centroid","zcr","pitch_mean","pitch_std","pitch_range",
                 "loudness","energy_change","duration"] + [f"band{i+1}" for i in range(9)]

def read_audio(path):
    sr, x = wavfile.read(path)
    x = x.astype(float) / 32768
    if x.ndim > 1: x = x.mean(axis=1)
    x = x - np.mean(x)
    return sr, x

def do_fft(x, sr):
    n = len(x)
    x = x * np.hanning(n)
    spectrum = np.fft.rfft(x)
    mag = np.abs(spectrum) / n
    freqs = np.fft.rfftfreq(n, 1/sr)
    keep = (freqs >= 50) & (freqs <= 4000)
    return freqs[keep], mag[keep]

def extract_features(freqs, mag, x):
    step = freqs[1]-freqs[0]
    smooth = np.convolve(mag, np.ones(int(50/step))/int(50/step), mode="same")
    peaks,_ = find_peaks(smooth, distance=int(100/step))
    if len(peaks)==0: peaks=[np.argmax(smooth)]
    top3 = peaks[np.argsort(smooth[peaks])[::-1]][:3]
    while len(top3)<3: top3 = np.append(top3, top3[-1])
    biggest = smooth[top3[0]]
    valleys,_ = find_peaks(-smooth, distance=int(100/step))
    valley = valleys[np.argmin(smooth[valleys])] if len(valleys)>0 else np.argmin(smooth)
    centroid = np.sum(freqs*smooth**2)/np.sum(smooth**2)
    zcr = np.mean(x[:-1]*x[1:]<0)
    return [freqs[top3[0]],freqs[top3[1]],freqs[top3[2]],
            smooth[top3[1]]/biggest,smooth[top3[2]]/biggest,
            freqs[valley],centroid,zcr], smooth, top3, valley

def plot_signal(path, sr, x, freqs, smooth, top3, valley, axes=None, title=""):
    time = np.arange(len(x)) / sr

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    ax0, ax1 = axes

    # Time domain
    ax0.plot(time, x)
    ax0.set_title("Time domain - " + os.path.basename(path) + " " + title)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.grid(alpha=0.3)

    # Frequency domain
    ax1.plot(freqs, smooth)
    ax1.plot(freqs[top3], smooth[top3], "ro", label="High points")
    ax1.plot(freqs[valley], smooth[valley], "bv", markersize=9, label="Valley")
    ax1.set_title("Frequency domain (FFT)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")
    ax1.grid(alpha=0.3)
    ax1.legend()

    if axes is None:
        plt.tight_layout()
        plt.show()


def build_database():
    rows=[]
    for folder,_,names in os.walk(DATASET_FOLDER):
        for name in names:
            if not name.endswith(".wav"): continue
            path=os.path.join(folder,name)
            sr,x=read_audio(path)
            freqs,mag=do_fft(x,sr)
            feats,smooth,top3,valley=extract_features(freqs,mag,x)
            rows.append([path]+feats)
    with open(DATABASE_FILE,"w",newline="") as f:
        writer=csv.writer(f); writer.writerow(["file"]+FEATURE_NAMES); writer.writerows(rows)
    print(f"Database built with {len(rows)} samples.")

def load_database():
    emotions, actors, files, data = [], [], [], []
    with open(DATABASE_FILE) as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            emotions.append(row[0])          # emotion
            actors.append(int(row[1]))       # actor
            files.append(row[2])             # file path
            data.append([float(v) for v in row[3:]])  # features only
    return np.array(emotions), np.array(actors), np.array(files), np.array(data)


