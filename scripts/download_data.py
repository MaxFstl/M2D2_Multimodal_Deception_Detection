import argparse
import os

from src.utils.YT_video_downloader import download, VidInfo
from multiprocessing.pool import ThreadPool
from src.utils.load_config import load_config
from src.utils.face.extract_face_frames import extract_face
from src.utils.video_to_audio import extract_audio

config = load_config()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cookies",
    type=str,
    default=None,
    help="Path to cookies file for YouTube downloads",
)
parser.add_argument(
    "--sleep",
    type=float,
    default=2,
    help="Seconds to sleep between video requests to avoid rate limiting",
)
args = parser.parse_args()

video_dir = config["paths"]["video_dir"]
csv_file = config["paths"]["timestamps"]
out_dir = video_dir
save_dir_face = config["paths"]["image_dir"]
save_dir_audio = config["paths"]["audio_dir"]


def start_download(video_dir, csv_file, cookiefile=None, sleep=0):
    if os.path.exists(video_dir):
        existing_files = [f for f in os.listdir(video_dir)]
        print(f"Found {len(existing_files)} existing videos in {video_dir}")

        if len(existing_files) > 0:
            print("Videos already exist. Skipping download.")
            return
        else:
            print("Directory exists but empty. Preprocessing the images...")

    os.makedirs(out_dir, exist_ok=True)

    with open(csv_file, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [x.split(",") for x in lines]
        print(lines[:5])
        vidinfos = [
            VidInfo(x[0], x[1], x[2], x[3], video_dir, cookiefile, sleep) for x in lines
        ]

    with open(
        os.path.join(config["paths"]["data_dir"], "bad_files.txt"), "w"
    ) as bad_files:
        results = ThreadPool(5).imap_unordered(download, vidinfos)
        cnt = 0
        for r in results:
            cnt += 1
            print(cnt, "/", len(vidinfos), r)
            if "ERROR" in r:
                bad_files.write(r + "\n")
        bad_files.close()


start_download(video_dir, csv_file, args.cookies, args.sleep)
print("Preprocessing...")
extract_face(root_dir=video_dir, save_dir=save_dir_face)
extract_audio(video_dir=video_dir, save_dir=save_dir_audio)
