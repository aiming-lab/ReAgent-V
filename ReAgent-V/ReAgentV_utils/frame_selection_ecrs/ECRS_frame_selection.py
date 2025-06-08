from PIL import Image
import math
import numpy as np
import torch

import torch.nn.functional as F

def compute_entropy(image: Image.Image) -> float:
    image_np = np.array(image)
    entropy_sum = 0.0
    for c in range(3):
        channel = image_np[..., c]
        histogram, _ = np.histogram(channel, bins=256, range=(0, 255), density=True)
        histogram += 1e-9
        entropy = -np.sum(histogram * np.log2(histogram))
        entropy_sum += entropy
    return entropy_sum / 3.0

def normalize_array(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min + 1e-9)

def select_keyframes(frames, query_text, clip_model, clip_processor, base_threshold=0.03, 
                      alpha=1.01, max_iter=10, min_frames=32):
    device = clip_model.device

    with torch.no_grad():
        text_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True).to(device)
        text_feature = F.normalize(clip_model.get_text_features(**text_inputs), dim=-1).cpu()

    frame_features = []
    for frame in frames:
        inputs = clip_processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = F.normalize(clip_model.get_image_features(**inputs), dim=-1).cpu()
        frame_features.append(feat)
    frame_features = torch.stack(frame_features).squeeze(1)

    selected_indices = list(range(len(frames)))

    for m in range(max_iter):
        print(f'---the current iteration is {m}---')
        sel_features = frame_features[selected_indices]
        sel_frames = [frames[i] for i in selected_indices]

        with torch.no_grad():
            similarities = torch.mm(sel_features, text_feature.T).squeeze(1).numpy()

        entropies = np.array([compute_entropy(f) for f in sel_frames])
        similarities = normalize_array(similarities)
        entropies = normalize_array(entropies)

        ecrs_scores = similarities * entropies
        ecrs_scores = normalize_array(ecrs_scores)

        threshold = base_threshold * (alpha ** m)
        candidates = [selected_indices[i] for i, score in enumerate(ecrs_scores) if score > threshold]

        if not candidates or set(candidates) == set(selected_indices):
            break
        selected_indices = candidates

    if len(selected_indices) < min_frames:
        all_entropies = np.array([compute_entropy(f) for f in frames])
        with torch.no_grad():
            all_similarities = torch.mm(frame_features, text_feature.T).squeeze(1).numpy()
        all_ecrs = normalize_array(all_similarities) * normalize_array(all_entropies)
        sorted_indices = np.argsort(all_ecrs)[::-1]
        selected_indices = sorted(set(selected_indices + sorted_indices[:min_frames].tolist()))
    
    selected_frames = [frames[i] for i in selected_indices]
    
    print(f'------the selected_frames is {selected_indices}------')
    return selected_frames, selected_indices
