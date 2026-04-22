import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import whisper
from scipy import signal
from scipy.io import wavfile
from transformers import BertTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

from src.models.fusion_model import Fusion
from src.utils.face.tools import FaceAlignmentTools
from src.utils.logging import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class VideoDeceptionDetector:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        whisper_model: str = "medium",
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")

        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint.get("config", {})
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        logger.info(f"Model config: {self.config}")

        self.num_frames = self.config.get("num_frames", 16)
        self.num_encoders = self.config.get("num_encoders", 4)
        self.adapter = self.config.get("adapter", True)
        self.adapter_type = self.config.get("adapter_type", "nlp")
        self.fusion_type = self.config.get("fusion_type", "cross2")
        self.multi = self.config.get("multi", False)

        self.model = Fusion(
            fusion_type=self.fusion_type,
            num_encoders=self.num_encoders,
            adapter=self.adapter,
            adapter_type=self.adapter_type,
            multi=self.multi,
            num_frames=self.num_frames,
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded and set to eval mode")

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.video_transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.face_aligner = FaceAlignmentTools()
        logger.info("Loading Whisper model for ASR...")
        self.whisper_model = whisper.load_model(whisper_model)
        logger.info("Whisper model loaded")

    def extract_audio_from_video(self, video_path: str, output_path: str) -> str:
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            raise ValueError(f"No audio track found in video: {video_path}")
        audio.write_audiofile(output_path, verbose=False, logger=None)
        video.close()
        return output_path

    def extract_face_frames(self, video_path: str) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")

        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        face_frames = []
        frame_count = 0

        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {target_idx} from {video_path}")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aligned_face = self.face_aligner.align(frame)

            if aligned_face is not None:
                aligned_face = cv2.resize(aligned_face, (160, 160))
                face_frames.append(aligned_face)
            else:
                logger.warning(f"No face detected in frame {target_idx}")
                face_frames.append(np.zeros((160, 160, 3), dtype=np.uint8))

            frame_count += 1

        cap.release()

        if len(face_frames) == 0:
            logger.error(f"No valid frames extracted from {video_path}")
            return None

        return np.array(face_frames)

    def preprocess_audio(self, audio_path: str, num_tokens: int = 64) -> torch.Tensor:
        sample_rate, waveform = wavfile.read(audio_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.dim() == 2:
            waveform = waveform.mean(dim=1)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        clip_duration = waveform.size(1) / sample_rate
        new_sample_rate = int(321.893491124260 * num_tokens / clip_duration)
        num_samples = int(clip_duration * new_sample_rate)

        waveform = waveform.squeeze(0).numpy()
        waveform = signal.resample(waveform, num_samples)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        return waveform.float()

    def preprocess_video(self, face_frames: np.ndarray) -> torch.Tensor:
        transformed = [self.video_transform(frame) for frame in face_frames]
        return torch.stack(transformed)

    def preprocess_text(self, text: str, max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def transcribe_audio(self, audio_path: str) -> str:
        result = self.whisper_model.transcribe(audio_path, language="en")
        return result["text"]

    def predict(self, video_path: str) -> Dict:
        start_time = time.time()
        video_name = Path(video_path).stem
        logger.info(f"Processing video: {video_name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, f"{video_name}.wav")

            logger.info("  Extracting audio...")
            self.extract_audio_from_video(video_path, audio_path)

            logger.info("  Extracting face frames...")
            face_frames = self.extract_face_frames(video_path)
            if face_frames is None:
                raise ValueError(f"Could not extract face frames from {video_path}")

            logger.info("  Transcribing audio...")
            transcript = self.transcribe_audio(audio_path)
            logger.info(f"  Transcript: {transcript[:100]}...")

            logger.info("  Preprocessing audio...")
            audio_tensor = self.preprocess_audio(audio_path).unsqueeze(0).to(self.device)

            logger.info("  Preprocessing video...")
            video_tensor = self.preprocess_video(face_frames).unsqueeze(0).to(self.device)

            logger.info("  Preprocessing text...")
            input_ids, attention_mask = self.preprocess_text(transcript)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            logger.info("  Running inference...")
            with torch.no_grad():
                audio_input = audio_tensor.squeeze(1)
                logits, _, _, _ = self.model(audio_input, video_tensor, input_ids, attention_mask)

                probs = torch.softmax(logits, dim=-1)
                pred_class: int = torch.argmax(logits, dim=-1).item()
                confidence = probs[0, pred_class].item()

        elapsed_time = time.time() - start_time
        prediction = "deceptive" if pred_class == 1 else "truthful"

        result = {
            "video": Path(video_path).name,
            "prediction": prediction,
            "confidence": confidence,
            "transcript": transcript,
            "processing_time_seconds": elapsed_time,
        }

        logger.info(f"  Prediction: {prediction} (confidence: {confidence:.4f})")
        logger.info(f"  Processing time: {elapsed_time:.2f}s")

        return result


def write_result_to_file(result: Dict, output_dir: str) -> str:
    video_name = Path(result["video"]).stem
    output_path = os.path.join(output_dir, f"{video_name}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"video: {result['video']}\n")
        f.write(f"prediction: {result['prediction']}\n")
        f.write(f"confidence: {result['confidence']:.4f}\n")
        f.write(f"processing_time_seconds: {result['processing_time_seconds']:.2f}\n")
        f.write(f"transcript: {result['transcript']}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run deception detection inference on videos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="inference_video/videos",
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_video/results",
        help="Directory to save prediction results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for ASR",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        os.makedirs(args.input_dir, exist_ok=True)
        print(f"Created input directory: {args.input_dir}")
        print("Place your videos in this directory and run again.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    video_files = []
    for f in os.listdir(args.input_dir):
        if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(os.path.join(args.input_dir, f))

    if not video_files:
        print(f"No video files found in {args.input_dir}")
        print(f"Supported formats: {VIDEO_EXTENSIONS}")
        return

    print(f"Found {len(video_files)} video(s) to process")
    print(f"Loading model from: {args.checkpoint}")

    detector = VideoDeceptionDetector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        whisper_model=args.whisper_model,
    )

    print("\nProcessing videos...")
    print("=" * 60)

    for video_path in sorted(video_files):
        try:
            result = detector.predict(video_path)
            output_path = write_result_to_file(result, args.output_dir)
            print(f"Saved result to: {output_path}")
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            video_name = Path(video_path).stem
            error_path = os.path.join(args.output_dir, f"{video_name}.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"video: {Path(video_path).name}\n")
                f.write(f"error: {str(e)}\n")
            print(f"Error saved to: {error_path}")
        print("-" * 60)

    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
