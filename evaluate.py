import torch
import numpy as np
import scipy.ndimage
from skimage.morphology import skeletonize

def evaluate_model(model, data_loader, device):
    """
    Evaluate model on the given data_loader. Returns a dict of metrics.
    """
    model.eval()
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    total_loss = 0.0
    total_pixels = 0
    # Counters for node metrics
    true_node_counts = {1:0, 2:0, 3:0, 4:0}
    pred_node_counts = {1:0, 2:0, 3:0, 4:0}
    match_counts = {1:0, 2:0, 3:0, 4:0}
    total_dist_sq = 0.0
    total_points = 0
    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            outputs = model(images)  # logits
            # accumulate cross-entropy
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_pixels += targets.numel()
            # Threshold prediction to binary mask
            prob = torch.sigmoid(outputs)
            pred_mask = (prob.cpu().numpy() >= 0.5).astype(np.uint8)
            true_mask = targets.cpu().numpy().astype(np.uint8)
            # Skeletonize prediction to ensure 1-pixel width
            pred_skel_bool = skeletonize(pred_mask.squeeze(0).astype(bool))
            pred_skel = pred_skel_bool.astype(np.uint8)
            true_mask = true_mask.squeeze(0).astype(np.uint8)
            # Distance transform metrics
            if pred_skel.any():
                dist_to_true = scipy.ndimage.distance_transform_edt(1 - true_mask)
                # for each predicted skeleton pixel, distance to nearest true
                pred_coords = np.argwhere(pred_skel == 1)
                if pred_coords.size > 0:
                    pred_dists = dist_to_true[pred_coords[:,0], pred_coords[:,1]]
                    total_dist_sq += np.sum(pred_dists**2)
                    total_points += pred_coords.shape[0]
            if true_mask.any():
                dist_to_pred = scipy.ndimage.distance_transform_edt(1 - pred_skel)
                true_coords = np.argwhere(true_mask == 1)
                if true_coords.size > 0:
                    true_dists = dist_to_pred[true_coords[:,0], true_coords[:,1]]
                    total_dist_sq += np.sum(true_dists**2)
                    total_points += true_coords.shape[0]
            # Node valences
            true_val_map = get_valence_map(true_mask)
            pred_val_map = get_valence_map(pred_skel)
            for val in [1, 2, 3, 4]:
                true_positions = np.argwhere(true_val_map == val)
                pred_positions = np.argwhere(pred_val_map == val)
                true_node_counts[val] += true_positions.shape[0]
                pred_node_counts[val] += pred_positions.shape[0]
                # Match true and predicted nodes of this valence
                matched_pred_idx = set()
                for t in true_positions:
                    ty, tx = t
                    for j, p in enumerate(pred_positions):
                        if j in matched_pred_idx:
                            continue
                        py, px = p
                        if np.hypot(py - ty, px - tx) <= 3.0:  # within 3px
                            match_counts[val] += 1
                            matched_pred_idx.add(j)
                            break
    # Aggregate metrics
    avg_cross_entropy = total_loss / total_pixels if total_pixels > 0 else 0.0
    dist_mse = total_dist_sq / total_points if total_points > 0 else 0.0
    metrics = {"cross_entropy": avg_cross_entropy, "dist_mse": dist_mse,
               "precision": {}, "recall": {}}
    for val in [1, 2, 3, 4]:
        t_count = true_node_counts[val]
        p_count = pred_node_counts[val]
        m_count = match_counts[val]
        precision = (m_count / p_count) if p_count > 0 else None
        recall    = (m_count / t_count) if t_count > 0 else None
        metrics["precision"][val] = precision
        metrics["recall"][val] = recall
    return metrics

def get_valence_map(skel_mask):
    if isinstance(skel_mask, torch.Tensor):
        skel_mask = skel_mask.cpu().numpy()

    skel_mask = skel_mask.squeeze()  # Make sure it's 2D now

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    neighbor_count = scipy.ndimage.convolve(skel_mask.astype(int), kernel, mode='constant', cval=0)
    
    val_map = np.zeros_like(skel_mask)
    val_map[skel_mask == 1] = neighbor_count[skel_mask == 1] - 1  # Subtract 1 for the pixel itself

    return val_map

