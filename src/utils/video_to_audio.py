# FROM: https://github.com/NMS05/AV-Data-Processing/blob/main/AV%20Data%20Processing/Audio/video_to_audio.py

import os
from moviepy.editor import *

target_folder = "audio_files/"

for f_name in os.listdir("videos/"):

    # load video
    f_path = os.path.join("videos/", f_name)
    video = VideoFileClip(f_path)

    # save audio
    audio = video.audio
    save_name = target_folder + f_name.split(".")[0] + ".wav"
    audio.write_audiofile(save_name)
