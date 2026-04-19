import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import soundfile as sf
import torchaudio.transforms as T
import random
import re

class AttentionWrapper(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention = attention_module
    
    def forward(self, x):
        result = self.attention(x)
        if isinstance(result, tuple):
            return result[0]
        return result

def fix_adapter_classes():
    from src.models import adapter
    
    def patched_init(self, transformer_encoder, is_conv=True):
        torch.nn.Module.__init__(self) 
        
        self.attention = nn.Sequential(
            AttentionWrapper(transformer_encoder.attention),
            transformer_encoder.dropout,
            transformer_encoder.layer_norm,
        )
        
        if is_conv:
            self.mhsa_conv_pass = adapter.Efficient_Conv_Pass()
            self.ffn_conv_pass = adapter.Efficient_Conv_Pass()
            self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        else:
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

    def conv_init_wrapper(self, transformer_encoder):
        return patched_init(self, transformer_encoder, is_conv=True)

    def nlp_init_wrapper(self, transformer_encoder):
        return patched_init(self, transformer_encoder, is_conv=False)

    adapter.w2v2_adapter_conv.__init__ = conv_init_wrapper
    adapter.w2v2_adapter_nlp.__init__ = nlp_init_wrapper

fix_adapter_classes()

from src.models.audio_model import W2V2_Model

def load_audio(file_path, target_sr=16000, max_duration=5):
    """Load and preprocess audio file"""

    waveform, sr = sf.read(file_path)
    waveform = torch.from_numpy(waveform).float()
    
    if waveform.dim() == 1:
        pass
    else:
        waveform = waveform.mean(dim=1)
    
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
    
    max_samples = target_sr * max_duration
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    else:
        padding = max_samples - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform

def predict_single_file(model, file_path, device):
    """Predict whether a single audio file contains truth or lie"""
    model.eval()
    
    waveform = load_audio(file_path)
    waveform = waveform.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
    
    truth_prob = probs[0][0].item()
    lie_prob = probs[0][1].item()
    
    return pred, truth_prob, lie_prob

def parse_model_params(file_name):
    encoders_match = re.search(r'e(\d+)', file_name)
    type_match = re.search(r'(nlp|conv)', file_name)
    num_encoders = int(encoders_match.group(1)) if encoders_match else 4
    adapter_type = type_match.group(1) if type_match else 'conv'
    
    return num_encoders, adapter_type

def test_model(model_path, test_dir, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    file_name = os.path.basename(model_path)
    num_encoders,adapter_type = parse_model_params(file_name)
    
    print(f"Auto-detected architecture: {num_encoders} encoders, {adapter_type} adapter")
    print("Loading model...")
    model = W2V2_Model(num_encoders=num_encoders, adapter=True, adapter_type=adapter_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    
    audio_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    
    if len(audio_files) == 0:
        print(f"No audio files found in {test_dir}")
        return
    
    sample_files = random.sample(audio_files, min(num_samples, len(audio_files)))
    print(f"Testing on {len(sample_files)} random samples:\n")
    print("=" * 80)
    
    correct = 0
    total = 0
    
    for file_name in sample_files:
        file_path = os.path.join(test_dir, file_name)
        true_label = 1 if 'lie' in file_name.lower() else 0
        true_label_str = "LIE" if true_label == 1 else "TRUTH"
        
        pred, truth_prob, lie_prob = predict_single_file(model, file_path, device)
        pred_label_str = "LIE" if pred == 1 else "TRUTH"
        
        is_correct = (pred == true_label)
        correct += int(is_correct)
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} File: {file_name}")
        print(f"  Ground Truth: {true_label_str}")
        print(f"  Prediction:   {pred_label_str}")
        print(f"  Confidence:   Truth={truth_prob:.2%}, Lie={lie_prob:.2%}")
        print("-" * 80)
    
    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    print("=" * 80)

def test_specific_file(model_path, audio_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file_name = os.path.basename(model_path)
    num_encoders, adapter_type = parse_model_params(file_name)
    
    print(f"Auto-detected: {num_encoders} encoders, {adapter_type} adapter")
    print("Loading model...")
    model = W2V2_Model(num_encoders=num_encoders, adapter=True, adapter_type=adapter_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    pred, truth_prob, lie_prob = predict_single_file(model, audio_file, device)
    pred_label_str = "LIE" if pred == 1 else "TRUTH"
    
    print("=" * 80)
    print(f"File: {audio_file}")
    print(f"Prediction: {pred_label_str}")
    print(f"Confidence: Truth={truth_prob:.2%}, Lie={lie_prob:.2%}")
    print("=" * 80)

if __name__ == "__main__":
    MODEL_ROOT = "src/models/audio/best"
    TEST_DIR = "data/DOLOS/audio_files"
    
    if os.path.exists(MODEL_ROOT):
        model_files = [f for f in os.listdir(MODEL_ROOT) if f.endswith('.pth')]
        if not model_files:
            print(f"Error: No .pth files found in {MODEL_ROOT}")
            sys.exit(1)
        MODEL_NAME = model_files[0] 
        MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_NAME)
    else:
        print(f"Error: Directory {MODEL_ROOT} not found.")
        sys.exit(1)

    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory not found at {TEST_DIR}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"Audio Lie Detection Model - Testing")
    print(f"Model File: {MODEL_NAME}")
    print("=" * 80)
    print()
    
    test_model(MODEL_PATH, TEST_DIR, num_samples=20)
    
    # Uncomment to test a specific file:
    # test_specific_file(MODEL_PATH, "data/DOLOS/audio_files/AN_WILTY_EP15_lie10.wav")
