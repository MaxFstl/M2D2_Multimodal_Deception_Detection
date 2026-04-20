import os
import pandas as pd
import pytest

from src.data.validation import validate_clips


class TestValidateClips:
    @pytest.fixture
    def setup_dirs(self, tmp_path):
        annos = pd.DataFrame(
            {
                "video_id": ["vid001", "vid001", "vid002", "vid002"],
                "file_name": ["clip_001", "clip_002", "clip_003", "clip_004"],
                "label": ["truth", "deception", "truth", "deception"],
            }
        )
        annos_file = tmp_path / "annotations.csv"
        annos.to_csv(str(annos_file), index=False)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        img_dir = tmp_path / "video"
        img_dir.mkdir()

        return {
            "annotations_file": str(annos_file),
            "audio_dir": str(audio_dir),
            "img_dir": str(img_dir),
        }

    def _create_audio_file(self, audio_dir, clip_name):
        import soundfile as sf
        import numpy as np

        audio_path = os.path.join(audio_dir, f"{clip_name}.wav")
        waveform = np.random.randn(16000).astype(np.float32)
        sf.write(audio_path, waveform, 16000)

    def _create_video_dir(self, img_dir, clip_name, num_frames=4):
        import numpy as np
        from PIL import Image

        clip_dir = os.path.join(img_dir, clip_name)
        os.makedirs(clip_dir, exist_ok=True)
        for i in range(num_frames):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(clip_dir, f"frame_{i:04d}.jpg"))

    def test_all_clips_exist(self, setup_dirs):
        for clip in ["clip_001", "clip_002", "clip_003", "clip_004"]:
            self._create_audio_file(setup_dirs["audio_dir"], clip)
            self._create_video_dir(setup_dirs["img_dir"], clip)

        valid_clips, missing_clips = validate_clips(
            setup_dirs["annotations_file"],
            setup_dirs["audio_dir"],
            setup_dirs["img_dir"],
        )

        assert len(valid_clips) == 4
        assert len(missing_clips) == 0
        assert "clip_001" in valid_clips
        assert "clip_004" in valid_clips

    def test_missing_audio_only(self, setup_dirs):
        self._create_video_dir(setup_dirs["img_dir"], "clip_001")
        self._create_video_dir(setup_dirs["img_dir"], "clip_002")
        self._create_video_dir(setup_dirs["img_dir"], "clip_003")
        self._create_video_dir(setup_dirs["img_dir"], "clip_004")
        self._create_audio_file(setup_dirs["audio_dir"], "clip_001")
        self._create_audio_file(setup_dirs["audio_dir"], "clip_002")

        valid_clips, missing_clips = validate_clips(
            setup_dirs["annotations_file"],
            setup_dirs["audio_dir"],
            setup_dirs["img_dir"],
        )

        assert len(valid_clips) == 2
        assert len(missing_clips) == 2
        assert "clip_001" in valid_clips
        assert "clip_002" in valid_clips
        assert "clip_003" in missing_clips
        assert "clip_004" in missing_clips

    def test_missing_video_only(self, setup_dirs):
        self._create_audio_file(setup_dirs["audio_dir"], "clip_001")
        self._create_audio_file(setup_dirs["audio_dir"], "clip_002")
        self._create_audio_file(setup_dirs["audio_dir"], "clip_003")
        self._create_audio_file(setup_dirs["audio_dir"], "clip_004")
        self._create_video_dir(setup_dirs["img_dir"], "clip_001")
        self._create_video_dir(setup_dirs["img_dir"], "clip_003")

        valid_clips, missing_clips = validate_clips(
            setup_dirs["annotations_file"],
            setup_dirs["audio_dir"],
            setup_dirs["img_dir"],
        )

        assert len(valid_clips) == 2
        assert len(missing_clips) == 2
        assert "clip_001" in valid_clips
        assert "clip_003" in valid_clips

    def test_missing_both(self, setup_dirs):
        self._create_audio_file(setup_dirs["audio_dir"], "clip_001")
        self._create_video_dir(setup_dirs["img_dir"], "clip_001")
        self._create_audio_file(setup_dirs["audio_dir"], "clip_002")

        valid_clips, missing_clips = validate_clips(
            setup_dirs["annotations_file"],
            setup_dirs["audio_dir"],
            setup_dirs["img_dir"],
        )

        assert len(valid_clips) == 1
        assert len(missing_clips) == 3
        assert "clip_001" in valid_clips

    def test_empty_annotations(self, tmp_path):
        annos = pd.DataFrame(
            {
                "video_id": [],
                "file_name": [],
                "label": [],
            }
        )
        annos_file = tmp_path / "empty_annotations.csv"
        annos.to_csv(str(annos_file), index=False)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        img_dir = tmp_path / "video"
        img_dir.mkdir()

        valid_clips, missing_clips = validate_clips(
            str(annos_file), str(audio_dir), str(img_dir)
        )

        assert len(valid_clips) == 0
        assert len(missing_clips) == 0

    def test_empty_directories(self, setup_dirs):
        valid_clips, missing_clips = validate_clips(
            setup_dirs["annotations_file"],
            setup_dirs["audio_dir"],
            setup_dirs["img_dir"],
        )

        assert len(valid_clips) == 0
        assert len(missing_clips) == 4
