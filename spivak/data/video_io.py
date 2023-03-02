# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from pathlib import Path
from typing import List

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv"}


def list_video_paths(input_videos_dir: Path) -> List[Path]:
    return sorted(
        [file_path
         for file_path in input_videos_dir.iterdir()
         if is_video_path(file_path)]
    )


def is_video_path(video_path: Path) -> bool:
    return video_path.suffix in VIDEO_EXTENSIONS
