import os
import sys
import csv
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import optuna

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ==========================================
# 1. Fix the adapter by wrapping the attention module
# ==========================================
class AttentionWrapper(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention = attention_module
    
    def forward(self, x):
        result = self.attention(x)
        if isinstance(result, tuple):
            return result[0]
        return result

# Patch adapter classes before importing the model
def fix_adapter_classes():
    from src.models import adapter
    
    original_w2v2_conv_init = adapter.w2v2_adapter_conv.__init__
    original_w2v2_nlp_init = adapter.w2v2_adapter_nlp.__init__
    
    def patched_w2v2_conv_init(self, transformer_encoder):
        super(adapter.w2v2_adapter_conv, self).__init__()
        
        self.attention = nn.Sequential(
            AttentionWrapper(transformer_encoder.attention),
            transformer_encoder.dropout,
            transformer_encoder.layer_norm,
        )
        
        self.mhsa_conv_pass = adapter.Efficient_Conv_Pass()
        self.ffn_conv_pass = adapter.Efficient_Conv_Pass()
        
        self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )
    
    def patched_w2v2_nlp_init(self, transformer_encoder):
        super(adapter.w2v2_adapter_nlp, self).__init__()
        
        self.attention = nn.Sequential(
            AttentionWrapper(transformer_encoder.attention),
            transformer_encoder.dropout,
            transformer_encoder.layer_norm,
        )
        
        self.adapter = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=768, bias=True),
            nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        )
        
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )
    
    adapter.w2v2_adapter_conv.__init__ = patched_w2v2_conv_init
    adapter.w2v2_adapter_nlp.__init__ = patched_w2v2_nlp_init

fix_adapter_classes()

from src.models.audio_model import W2V2_Model 

# ==========================================
# 2. Parsing DOLOS Filenames
# ==========================================
class DolosAudioDataset(Dataset):
    def __init__(self, data_dir, target_sr=16000, max_duration=5):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.max_samples = target_sr * max_duration

        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        
        if len(self.file_list) == 0:
            print(f"Error, in {data_dir} no .wav file found！")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        label = 1 if 'lie' in file_name.lower() else 0

        # Use soundfile to load audio
        waveform, sr = sf.read(file_path)
        waveform = torch.from_numpy(waveform).float()
        
        # Convert to mono if stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            # Average channels to convert to mono
            waveform = waveform.mean(dim=1, keepdim=True).T

        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform.squeeze(0), torch.tensor(label)

# ==========================================
# 3. Leaderboard and CSV Management Functions
# ==========================================
def update_top_5_leaderboard(trial_data, model_state, model_dir):
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    csv_path = os.path.join(ckpt_dir, "top_5_models.csv")
    os.makedirs(ckpt_dir, exist_ok=True)

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            rows = list(csv.DictReader(f))

    current_entry = {
        'trial_id': str(trial_data['trial_id']),
        'accuracy': f"{trial_data['acc']:.4f}",
        **{k: str(v) for k, v in trial_data['params'].items()}
    }
    rows.append(current_entry)
    rows.sort(key=lambda x: float(x['accuracy']), reverse=True)
    top_5 = rows[:5]

    if top_5:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=top_5[0].keys())
            writer.writeheader()
            writer.writerows(top_5)

    if any(r['trial_id'] == str(trial_data['trial_id']) for r in top_5):

        e = trial_data['params']['num_encoders']
        a = trial_data['params']['adapter_type']
        
        save_name = f"trial_{trial_data['trial_id']}_e{e}_{a}_acc_{trial_data['acc']:.4f}.pth"
        save_path = os.path.join(ckpt_dir, save_name)
        torch.save(model_state, save_path)

        print(f"  --> Top 5 Model Saved: {save_path}")

# ==========================================
# 4. Training Function Definition
# ==========================================
def train_audio_branch(data_dir, epochs, batch_size, lr, num_encoders, adapter_type, is_tuning=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = DolosAudioDataset(data_dir)
    train_size = int(0.9 * len(full_dataset))
    train_loader = DataLoader(Subset(full_dataset, range(train_size)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=batch_size, shuffle=False)

    model = W2V2_Model(num_encoders=num_encoders, adapter=True, adapter_type=adapter_type).to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=is_tuning)
        for wavs, labels in pbar:
            wavs, labels = wavs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(wavs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for wavs, labels in val_loader:
                wavs, labels = wavs.to(device), labels.to(device)
                logits = model(wavs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    del model
    torch.cuda.empty_cache()
    return best_acc, best_state


# ==========================================
# 5. Hyper parameter fine-tuning
# ==========================================
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 5e-5, 3e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8]),
        "num_encoders": trial.suggest_int("num_encoders", 2, 6),
        "adapter_type": trial.suggest_categorical("adapter_type", ["conv", "nlp"])
    }
    
    print(f"\n[Trial {trial.number}] Parameters: {params}")
    
    acc, state = train_audio_branch(
        data_dir="data/DOLOS/audio_files",
        epochs=10,
        is_tuning=True,
        **params
    )
    
    update_top_5_leaderboard(
        {'acc': acc, 'trial_id': trial.number, 'params': params},
        state,
        "src/models/audio"
    )
    
    return acc

# ==========================================
# 6. Main Function
# ==========================================
def main():
    AUDIO_PATH = "data/DOLOS/audio_files"
    MODEL_ROOT = "src/models/audio"
    BEST_DIR = os.path.join(MODEL_ROOT, "best")
    MODEL_SAVE_PATH = os.path.join(BEST_DIR, "audio_adapter_m2d2.pth")

    os.makedirs(BEST_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_ROOT, "checkpoints"), exist_ok=True)

    if not os.path.exists(AUDIO_PATH):
        print(f"Error: {AUDIO_PATH} not found.")
        return

    print("--- Starting Hyperparameter Optimization (Optuna) ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\n" + "="*30)
    print("Optimization Finished!")
    print(f"Best Params: {study.best_params}")
    
    print("\n--- Starting Final Training with Best Params ---")
    acc, state = train_audio_branch(
        data_dir=AUDIO_PATH,
        epochs=50,
        is_tuning=False,
        num_encoders=study.best_params['num_encoders'],
        adapter_type=study.best_params['adapter_type'],
        lr=study.best_params['lr'],
        batch_size=study.best_params['batch_size']
    )

    best_params = study.best_params
    e = best_params['num_encoders']
    a = best_params['adapter_type']

    MODEL_NAME = f"audio_best_e{e}_{a}.pth"
    MODEL_SAVE_PATH = os.path.join(MODEL_ROOT, "best", MODEL_NAME)
    if state is not None:
        torch.save(state, MODEL_SAVE_PATH)
    print(f"Final Best Model saved to: {MODEL_SAVE_PATH}")
    print("="*30)

if __name__ == "__main__":
    main()