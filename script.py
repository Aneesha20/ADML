# Main script (script.py)
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import RoadDataset
from model import UNet
from train import train_model
from evaluate import evaluate_model
import torch
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

# Set local data path
data_dir = r"/content/Mi/thinning_data/data/thinning"

# Prepare datasets and loaders
train_dataset = RoadDataset(data_dir, split="train", augment=True, misalign_max=0)
val_dataset   = RoadDataset(data_dir, split="val", augment=False)
test_dataset  = RoadDataset(data_dir, split="test", augment=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=1, out_channels=1, init_features=64)
print(f"Using device: {device}")

# Train model
model, history = train_model(model, train_loader, val_loader, device, epochs=30, learning_rate=1e-3, alpha=0.1)
print("Training complete.")

# Evaluate on test set
metrics = evaluate_model(model, test_loader, device)

print("\n📊 Test Set Metrics:")
print(f"- Cross-Entropy Loss: {metrics['cross_entropy']:.4f}")
print(f"- Distance Transform MSE: {metrics['dist_mse']:.4f}")
for val in [1, 2, 3, 4]:
    try:
        prec = metrics['precision'][val]
        rec  = metrics['recall'][val]
        prec_str = f"{prec:.2f}" if prec is not None else "N/A"
        rec_str  = f"{rec:.2f}" if rec is not None else "N/A"
        print(f"- Valence {val} Nodes: Precision = {prec_str}, Recall = {rec_str}")
    except KeyError:
        print(f"- Valence {val}: No instances in truth/prediction")

# Create output folder if it doesn't exist
os.makedirs("/content/Mi", exist_ok=True)

# Generate qualitative comparisons
sample_indices = [2, 4] if len(test_dataset) > 4 else [0]
for idx in sample_indices:
    img_tensor, tgt_tensor, _ = test_dataset[idx]
    img = img_tensor.unsqueeze(0).to(device, dtype=torch.float32)
    tgt = tgt_tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(img)
        prob = torch.sigmoid(logits)

    pred_mask = (prob.cpu().numpy() >= 0.5).astype(np.uint8)
    pred_skel = skeletonize(pred_mask.squeeze().astype(bool)).astype(np.uint8)

    input_image = img_tensor.squeeze().numpy() * 255
    gt_image    = tgt_tensor.squeeze().numpy() * 255
    pred_image  = pred_skel * 255

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(input_image, cmap='gray'); plt.title("Input"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(gt_image, cmap='gray');    plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(pred_image, cmap='gray');  plt.title("Prediction"); plt.axis('off')
    plt.tight_layout()

    save_path = f"/content/Mi/result_example_{idx}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved result image for test index {idx} → {save_path}")



    # --- ABLATION STUDY ---
print("\n Running Ablation Study...\n")
ablations = [
    {"name": "alpha=0.0", "alpha": 0.0, "lr": 1e-3},
    {"name": "alpha=0.05", "alpha": 0.05, "lr": 1e-3},
    {"name": "alpha=0.1", "alpha": 0.1, "lr": 1e-3},
    {"name": "alpha=0.1_lr=1e-4", "alpha": 0.1, "lr": 1e-4},
]

results = []

for config in ablations:
    print(f"\n--- {config['name']} ---")
    model = UNet(in_channels=1, out_channels=1, init_features=64)
    model, _ = train_model(model, train_loader, val_loader, device,
                           epochs=10, learning_rate=config['lr'], alpha=config['alpha'])  
    metrics = evaluate_model(model, test_loader, device)
    results.append({
        "config": config['name'],
        "xent": metrics['cross_entropy'],
        "mse": metrics['dist_mse'],
        "p2": metrics['precision'][2],
        "r2": metrics['recall'][2],
    })

# Print Ablation Summary
print("\n📋 Ablation Summary (Valence-2 Example):")
print(f"{'Config':<20} {'XENT':<8} {'MSE':<10} {'Prec@2':<8} {'Rec@2':<8}")
for r in results:
    p2 = f"{r['p2']:.2f}" if r['p2'] is not None else "N/A"
    r2 = f"{r['r2']:.2f}" if r['r2'] is not None else "N/A"
    print(f"{r['config']:<20} {r['xent']:<8.4f} {r['mse']:<10.2f} {p2:<8} {r2:<8}")

