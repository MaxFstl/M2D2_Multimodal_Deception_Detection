# FROM: https://github.com/NMS05/AV-Data-Processing/blob/main/AV%20Data%20Processing/Audio/video_to_audio.py

import os

# from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract_audio(video_dir, save_dir):
    if os.path.exists(save_dir):
        existing_files = [f for f in os.listdir(save_dir)]
        print(f"Found {len(existing_files)} existing files in {save_dir}")

        if len(existing_files) > 0:
            print("Already extracted audio. Skipping this step.")
            return
        else:
            print("Directory exists but empty. Preprocessing the audio...")
    os.makedirs(save_dir, exist_ok=True)
    for f_name in os.listdir(video_dir):

        # load video
        f_path = os.path.join(video_dir, f_name)
        video = VideoFileClip(f_path)

        # save audio
        audio = video.audio
        save_name = save_dir + f_name.split(".")[0] + ".wav"
        audio.write_audiofile(save_name)
