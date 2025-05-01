import torch
import torch.nn as nn
import numpy as np
import copy

def train_model(model, train_loader, val_loader, device, 
                epochs=30, learning_rate=1e-3, alpha=0.1):
    """
    Train the model for a given number of epochs. Returns the best model (by val loss) and loss history.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    best_val_loss = float('inf')
    best_model_wts = None
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for images, targets, dist_maps in train_loader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            dist_maps = dist_maps.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(images)                 # forward pass (logits)
            bce_loss = criterion(outputs, targets)  # BCE loss
            prob = torch.sigmoid(outputs)           # convert to [0,1] probability
            dist_loss = ((prob * dist_maps) ** 2).mean()  # distance-based loss
            loss = bce_loss + alpha * dist_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, targets, dist_maps in val_loader:
                images = images.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                dist_maps = dist_maps.to(device, dtype=torch.float32)
                outputs = model(images)
                bce_loss = criterion(outputs, targets)
                prob = torch.sigmoid(outputs)
                dist_loss = ((prob * dist_maps) ** 2).mean()
                loss = bce_loss + alpha * dist_loss
                val_loss_sum += loss.item() * images.size(0)
        epoch_val_loss = val_loss_sum / len(val_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        # Checkpoint best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model, history
