import os
import torch

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def save_model(model, version):
    """Saves the model to the models directory."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    save_path = os.path.join(MODELS_DIR, f"{version}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")