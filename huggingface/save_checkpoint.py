from trainer import SpeechLLMLightning
import os
import torch

ckpt_path = "best_checkpoint.ckpt"
model = SpeechLLMLightning.load_from_checkpoint(ckpt_path)
model = model.eval()

# Directory to save the models
save_dir = "checkpoints/pth"
os.makedirs(save_dir, exist_ok=True)

model.llm_model = model.llm_model.merge_and_unload()
torch.save(model.state_dict(), 'torch_model_best_checkpoints.pth')
print("Models saved successfully.")
