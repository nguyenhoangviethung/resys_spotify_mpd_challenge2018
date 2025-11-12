#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import string
import gzip
import csv
import sys

# ===================== CONFIG =====================
EPOCHS = 2
T_EPOCHS = 1
BATCH_SIZE = 128
HIDDEN_DIM = 1024
LR = 1e-3
NOISE_PROB = 0.3
MAX_LEN = 50

# Đường dẫn dữ liệu (cập nhật theo môi trường thực tế)
DATA_DIR = '/kaggle/input/spotify-million-playlist-dataset/data' # thay bằng đường dẫn đến ./spotify-million-playlist-dataset/data
CHALLENGE_PATH = '/kaggle/input/spotify-million-playlist-dataset-challenge/challenge_set.json'  # Thay bằng đường dẫn đến ./spotify-million-playlist-dataset-challenge/challenge_set.json
output_path = '/kaggle/working/submission.csv.gz' # thay bằng đường dẫn đến file submission, định dạng .csv.gz
output_path_csv = '/kaggle/working/submission.csv' # thay bằng đường dẫn đến file submission để verify, định dạng .csv
verify_submission_file = '/kaggle/input/spotify-million-playlist-dataset-challenge/verify_submission.py' # thay bằng đường dẫn đến ./spotify-million-playlist-dataset-challenge/verify_submission.py
# Số lượng slice để train (giảm để test nhanh)
NUM_SLICES = 4 #thay bằng số slices được nạp vào để huấn luyện (tối đa 1000)

# Team info
team_name = "HUST_RESYS_20251_G10_162306"
contact_email = "nguyenhoangviethung@gmail.com"

# ===================== DEVICE =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===================== LOAD DATA =====================
slices = [f for f in os.listdir(DATA_DIR) if f.startswith('mpd.slice')]
slices = slices[:NUM_SLICES]
playlists = []

print(f"Loading {len(slices)} MPD slices...")
for slice_file in tqdm(slices, desc="Reading MPD slices"):
    with open(os.path.join(DATA_DIR, slice_file), 'r') as f:
        data = json.load(f)
        playlists.extend(data['playlists'])

print(f"Loaded {len(playlists)} playlists")

# ===================== BUILD VOCAB =====================
track_counter = Counter()
artist_counter = Counter()
title_list = []

for pl in playlists:
    tracks = [t['track_uri'] for t in pl['tracks']]
    artists = [t['artist_uri'] for t in pl['tracks']]
    track_counter.update(tracks)
    artist_counter.update(artists)
    title_list.append(pl['name'].lower())

MIN_TRACK_COUNT = 5
MIN_ARTIST_COUNT = 3

valid_tracks = {t for t, c in track_counter.items() if c >= MIN_TRACK_COUNT}
valid_artists = {a for a, c in artist_counter.items() if c >= MIN_ARTIST_COUNT}

print(f"Valid tracks: {len(valid_tracks)}, Valid artists: {len(valid_artists)}")

track2id = {t: i for i, t in enumerate(sorted(valid_tracks))}
artist2id = {a: i for i, a in enumerate(sorted(valid_artists))}
id2track = {i: t for t, i in track2id.items()}

N_TRACKS = len(track2id)
N_ARTISTS = len(artist2id)
INPUT_DIM = N_TRACKS + N_ARTISTS

# ===================== PREPROCESS PLAYLISTS =====================
def playlist_to_vector(pl):
    track_ids = []
    artist_ids = []
    for t in pl['tracks']:
        track_uri = t['track_uri']
        artist_uri = t['artist_uri']
        if track_uri in track2id and artist_uri in artist2id:
            track_ids.append(track2id[track_uri])
            artist_ids.append(artist2id[artist_uri])
    return track_ids, artist_ids, pl['name'].lower()

data = []
titles = []
for pl in playlists:
    track_ids, artist_ids, title = playlist_to_vector(pl)
    if len(track_ids) >= 2:
        data.append((track_ids, artist_ids))
        titles.append(title)

print(f"Final training samples: {len(data)}")

# ===================== DAE DATASET & MODEL =====================
class DAEDataset(Dataset):
    def __init__(self, data, n_tracks, n_artists, noise_prob=0.3):
        self.data = data
        self.n_tracks = n_tracks
        self.n_artists = n_artists
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        track_ids, artist_ids = self.data[idx]
        input_vec = np.zeros(self.n_tracks + self.n_artists)
        for t in track_ids:
            input_vec[t] = 1
        for a in artist_ids:
            input_vec[self.n_tracks + a] = 1

        noisy_vec = input_vec.copy()
        mask = np.random.random(len(noisy_vec)) < self.noise_prob
        noisy_vec[mask] = 0

        return torch.tensor(noisy_vec, dtype=torch.float), torch.tensor(input_vec, dtype=torch.float)

class DAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

# ===================== TRAIN DAE =====================
dataset = DAEDataset(data, N_TRACKS, N_ARTISTS, noise_prob=NOISE_PROB)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = DAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Training DAE...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for noisy, clean in tqdm(dataloader, desc=f"DAE Epoch {epoch+1}/{EPOCHS}"):
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"DAE Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.encoder.state_dict(), 'dae_encoder.pth')

# ===================== TITLE CNN =====================
all_chars = set()
for title in titles:
    all_chars.update(title.lower())
all_chars = sorted(all_chars)
char2id = {c: i+1 for i, c in enumerate(all_chars)}
char2id[''] = 0
VOCAB_SIZE = len(char2id)

def title_to_tensor(title):
    seq = [char2id.get(c, 0) for c in title.lower()[:MAX_LEN]]
    seq += [0] * (MAX_LEN - len(seq))
    return torch.tensor(seq)

class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in [3, 4, 5]
        ])
        self.fc = nn.Linear(num_filters * 3, 1024)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        convs = [torch.relu(conv(x)).max(2)[0] for conv in self.convs]
        x = torch.cat(convs, dim=1)
        return self.fc(x)

char_cnn = CharCNN(VOCAB_SIZE).to(device)

# ===================== MMCF MODEL =====================
class MMCF(nn.Module):
    def __init__(self, dae_encoder, char_cnn, input_dim):
        super().__init__()
        self.dae_encoder = dae_encoder
        self.char_cnn = char_cnn
        self.final = nn.Linear(1024 * 2, input_dim)

    def forward(self, noisy_vec, title_tensor):
        dae_h = self.dae_encoder(noisy_vec)
        title_h = self.char_cnn(title_tensor)
        h = torch.cat([dae_h, title_h], dim=1)
        return torch.sigmoid(self.final(h))

# Load pre-trained DAE encoder
dae_encoder = nn.Sequential(
    nn.Linear(INPUT_DIM, 1024),
    nn.ReLU(),
    nn.Dropout(0.5)
).to(device)
dae_encoder.load_state_dict(torch.load('dae_encoder.pth'))
dae_encoder.eval()

mmcf = MMCF(dae_encoder, char_cnn, INPUT_DIM).to(device)
optimizer = optim.Adam(mmcf.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# ===================== MMCF DATASET =====================
class MMCFDataset(Dataset):
    def __init__(self, data, titles):
        self.data = data
        self.titles = titles

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        track_ids, artist_ids = self.data[idx]
        input_vec = np.zeros(INPUT_DIM)
        for t in track_ids: input_vec[t] = 1
        for a in artist_ids: input_vec[N_TRACKS + a] = 1
        noisy_vec = input_vec.copy()
        mask = np.random.random(INPUT_DIM) < 0.3
        noisy_vec[mask] = 0

        title_tensor = title_to_tensor(self.titles[idx])
        return (torch.tensor(noisy_vec, dtype=torch.float),
                title_tensor,
                torch.tensor(input_vec, dtype=torch.float))

mmcf_dataset = MMCFDataset(data, titles)
mmcf_loader = DataLoader(mmcf_dataset, batch_size=64, shuffle=True)

# ===================== TRAIN MMCF =====================
print("Training MMCF...")
mmcf.train()
for epoch in range(T_EPOCHS):
    for noisy, title, clean in tqdm(mmcf_loader, desc=f"MMCF Epoch {epoch+1}/{T_EPOCHS}"):
        noisy, title, clean = noisy.to(device), title.to(device), clean.to(device)
        pred = mmcf(noisy, title)
        loss = criterion(pred, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"MMCF Epoch {epoch+1} completed")

# ===================== INFERENCE ON CHALLENGE SET =====================
print("Loading challenge set...")
with open(CHALLENGE_PATH, 'r') as f:
    challenge_data = json.load(f)
challenge_playlists = challenge_data['playlists']
print(f"Loaded {len(challenge_playlists)} challenge playlists")

def get_seed_tracks(pl, scenario):
    tracks = pl['tracks']
    n = len(tracks)
    title = pl.get('name', '').lower()
    if scenario == 1:  # Title only
        return [], title
    elif scenario == 2:  # Title + first 1
        return tracks[:1], title
    elif scenario == 3:  # Title + first 5
        return tracks[:5], title
    elif scenario == 4:  # First 5 only
        return tracks[:5], ''
    elif scenario == 5:  # Title + first 10
        return tracks[:10], title
    elif scenario == 6:  # First 10 only
        return tracks[:10], ''
    elif scenario == 7:  # Title + first 25
        return tracks[:25], title
    elif scenario == 8:  # Title + 25 random
        k = min(25, n)
        idx = random.sample(range(n), k)
        return [tracks[i] for i in idx], title
    elif scenario == 9:  # Title + first 100
        return tracks[:100], title
    elif scenario == 10:  # Title + 100 random
        k = min(100, n)
        idx = random.sample(range(n), k)
        return [tracks[i] for i in idx], title
    return [], title

print("Generating predictions...")
mmcf.eval()
all_predictions = {}

with torch.no_grad():
    for pl in tqdm(challenge_playlists, desc="Predicting"):
        pid = pl['pid']
        all_seed_uris = {t['track_uri'] for t in pl['tracks']}
        title = pl.get('name', '').lower()
        candidate_uris = set()

        for scenario in range(1, 11):
            seed_tracks, _ = get_seed_tracks(pl, scenario)
            input_vec = np.zeros(INPUT_DIM)
            for t in seed_tracks:
                uri = t['track_uri']
                a_uri = t['artist_uri']
                if uri in track2id:
                    input_vec[track2id[uri]] = 1
                if a_uri in artist2id:
                    input_vec[N_TRACKS + artist2id[a_uri]] = 1

            noisy = torch.tensor(input_vec, dtype=torch.float).unsqueeze(0).to(device)
            title_tensor = title_to_tensor(title).unsqueeze(0).to(device)

            pred_probs = mmcf(noisy, title_tensor).cpu().numpy()[0]
            top_idx = np.argsort(pred_probs)[-1500:][::-1]
            for i in top_idx:
                if i < N_TRACKS:
                    uri = id2track[i]
                    if uri not in all_seed_uris and uri not in candidate_uris:
                        candidate_uris.add(uri)
                    if len(candidate_uris) >= 500:
                        break
            if len(candidate_uris) >= 500:
                break

        final_pred = list(candidate_uris)[:500]
        all_predictions[pid] = final_pred

print(f"Done! {len(all_predictions)} playlists ready.")

# ===================== SAVE SUBMISSION =====================
print("Saving submission...")
with gzip.open(output_path, 'wt', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['team_info', team_name, contact_email])
    for pid, tracks in all_predictions.items():
        row = [str(pid)] + tracks
        writer.writerow(row)

print(f"Submission saved: {output_path}")

# ===================== VERIFY SUBMISSION =====================
print("Verifying submission...")
os.system(f"gunzip -k {output_path}")
os.system(f"python {verify_submission_file} {CHALLENGE_PATH} {output_path_csv}")
print("Verification completed.")

print("All done!")