from src.data.loaders import create_dataloaders
from src.utils.load_config import load_config

if __name__ == "__main__":
    config = load_config()
    annotations = config["paths"]["annotations"]
    audio_dir = config["paths"]["audio_dir"]
    img_dir = config["paths"]["image_dir"]

    train, val, test = create_dataloaders(
        annotations_file=annotations,
        audio_dir=audio_dir,
        img_dir=img_dir,
        config=config,
    )

    for batch in train:
        print("Audio:", batch["audio"].shape)
        print("Video:", batch["video"].shape)
        print("Text IDs:", batch["text_ids"].shape)
        print("Text Mask:", batch["text_mask"].shape)
        print("Labels:", batch["label"])
        break
