# FROM https://github.com/NMS05/AV-Data-Processing/blob/main/AV%20Data%20Processing/Face/extract_face_frames.py

import cv2
from src.utils.face.tools import FaceAlignmentTools
import os
import numpy as np


def extract_face(root_dir, save_dir):
    if os.path.exists(save_dir):
        existing_files = [f for f in os.listdir(save_dir)]
        print(f"Found {len(existing_files)} existing images in {save_dir}")

        if len(existing_files) > 0:
            print("Already extracted images. Skipping this step.")
            return
        else:
            print("Directory exists but empty. Preprocessing the images...")

    tool = FaceAlignmentTools()

    total_vids = len(os.listdir(root_dir))
    vid_num = 1
    os.makedirs(save_dir, exist_ok=True)

    for file_name in os.listdir(root_dir):

        num_frames = 0
        save_name = file_name.split(".mp4")[0]
        os.mkdir(save_dir + save_name)
        save_path = save_dir + save_name

        print("Processing....", file_name, "\tProgress", round(vid_num / total_vids, 2))
        vid_num += 1

        cap = cv2.VideoCapture(root_dir + file_name)

        while True:
            ret, frame = cap.read()
            if ret == False:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aligned_img = tool.align(frame)

            if aligned_img is None:
                continue  # if no face detected, do not increase the num-frames

            aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
            aligned_img = cv2.resize(aligned_img, (224, 224))
            num_frames += 1

            s = "%04d" % num_frames
            save_image = save_path + "/frame_" + s + ".jpg"

            cv2.imwrite(save_image, aligned_img)
