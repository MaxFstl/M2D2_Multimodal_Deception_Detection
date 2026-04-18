# FROM: https://github.com/NMS05/AV-Data-Processing/blob/main/AV%20Data%20Processing/YT_video_downloader.py

import os
from typing import Optional

from yt_dlp import YoutubeDL
import ffmpeg

# COMMAND = python YT_video_downloader.py videos/ csv_file_name.csv
# csv file format must be 'yt_video_id, file_name, start_time (in seconds), end_time (in seconds)'


class VidInfo:
    def _parse_time(self, time_str: str) -> float:
        if ":" not in time_str:
            return float(time_str * 60)
        parts = time_str.split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return float(time_str)

    def __init__(
        self,
        yt_id,
        file_name,
        start_time: str,
        end_time: str,
        outdir,
        cookiefile: Optional[str] = None,
        sleep: float = 0,
    ):
        self.yt_id = yt_id
        self.start_time = self._parse_time(start_time)
        self.end_time = self._parse_time(end_time)
        self.out_filename = os.path.join(outdir, file_name + ".mp4")
        self.cookiefile = cookiefile
        self.sleep = sleep


def download(vidinfo):
    yt_base_url = "https://www.youtube.com/watch?v="

    yt_url = yt_base_url + vidinfo.yt_id

    ydl_opts = {
        "format": "22/18",
        "quiet": True,
        "ignoreerrors": True,
        "no_warnings": True,
    }
    if vidinfo.cookiefile:
        ydl_opts["cookiefile"] = vidinfo.cookiefile
    if vidinfo.sleep:
        ydl_opts["sleep_requests"] = vidinfo.sleep
    try:
        with YoutubeDL(ydl_opts) as ydl:
            download_url = ydl.extract_info(url=yt_url, download=False)["url"]
    except Exception as e:
        return_msg = "{}, ERROR (youtube): {}".format(vidinfo.yt_id, str(e))
        return return_msg
    try:
        (
            ffmpeg.input(download_url, ss=vidinfo.start_time, to=vidinfo.end_time)
            .output(
                vidinfo.out_filename,
                format="mp4",
                r=25,
                vcodec="libx264",
                crf=18,
                preset="veryfast",
                pix_fmt="yuv420p",
                acodec="aac",
                audio_bitrate=128000,
                strict="experimental",
            )
            .global_args("-y")
            .global_args("-loglevel", "error")
            .run()
        )
    except Exception as e:
        return_msg = "{}, ERROR (ffmpeg): {}".format(vidinfo.yt_id, str(e))
        return return_msg

    return "{}, DONE!".format(vidinfo.yt_id)
