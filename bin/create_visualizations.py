#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, List, TextIO

import av
import numpy as np
from av.video.frame import VideoFrame
from pandas import DataFrame

from spivak.data.label_map import LabelMap
from spivak.data.video_io import is_video_path
from spivak.html_visualization.result_data import \
    VideoSoccerRecognizedActions, FrameRecognizedActions, \
    SOCCERNET_FEATURES_FREQUENCY, SPOTTING_COLUMN_NAMES, \
    convert_deltas_to_timestamps, Video, read_videos
from spivak.html_visualization.segmentation_visualization import \
    add_segmentation_graph, add_segmentation_scores_graphs
from spivak.html_visualization.spotting_visualization import \
    add_detections_graph, add_detection_scores_graphs, \
    add_predictions_graphs
from spivak.html_visualization.utils import add_video, add_click_code, \
    create_category_settings, create_custom_category_settings, ColorMapChoice
from spivak.video_visualization.recognition_visualization import \
    FrameRecognizedActionsView, cv2_draw_labels, get_multi_labels, Label
from spivak.video_visualization.video_drawing import \
    create_video_from_containers, TransformationSizes, apply_draw_to_av_frame, \
    add_canvas_border, run_timed

INDEX_HTML = "index.html"
SPOTTING_HTML = "spotting.html"
SEGMENTATION_HTML = "segmentation.html"
OUT_VIDEO_EXTENSION = ".mp4"
RECOGNITION_THRESHOLD = 0.5
COARSE_ACTIONS_CHUNK_SIZE = 6


class Args:
    INPUT_VIDEOS_DIR = "input_dir"
    VISUALIZATIONS_DIR = "output_dir"
    RESULTS_DIR = "results_dir"
    LABEL_MAP = "label_map"
    CREATE_VIDEOS = "create_videos"
    DEBUG_GRAPHS = "debug_graphs"


def main() -> None:
    args = _get_command_line_arguments()
    should_create_videos = args[Args.CREATE_VIDEOS]
    debug_graphs = args[Args.DEBUG_GRAPHS]
    input_videos_dir = Path(args[Args.INPUT_VIDEOS_DIR])
    visualizations_dir = Path(args[Args.VISUALIZATIONS_DIR])
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    label_map = LabelMap.read_label_map(Path(args[Args.LABEL_MAP]))
    videos = read_videos(Path(args[Args.RESULTS_DIR]), label_map)
    video_relative_paths = {k for k in list(videos.keys())}
    for video_relative_path in sorted(video_relative_paths):
        video = videos[video_relative_path]
        in_video_path = _get_in_video_path(
            input_videos_dir, video_relative_path)
        video_dir = visualizations_dir / video_relative_path
        video_dir.mkdir(exist_ok=True, parents=True)
        out_video_path = video_dir / (
                video_relative_path.name + OUT_VIDEO_EXTENSION)
        _create_out_video(
            video, in_video_path, out_video_path, should_create_videos,
            label_map)
        # Create HTML file and plots separately for action spotting and
        # segmentation, since the HTML files can get pretty big.
        if (video.spotting_results is not None or
                video.segmentation_results is not None):
            html_path = video_dir / INDEX_HTML
            with html_path.open("w") as html_file:
                if video.spotting_results is not None:
                    html_file.write(
                        f'<a href="{SPOTTING_HTML}">Spotting</a><br>\n')
                    _write_spotting_html(
                        video, out_video_path, video_dir, label_map,
                        debug_graphs)
                if video.segmentation_results is not None:
                    html_file.write(
                        f'<a href="{SEGMENTATION_HTML}">Segmentation</a><br>\n')
                    _write_segmentation_html(
                        video, out_video_path, video_dir, label_map,
                        debug_graphs)


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.INPUT_VIDEOS_DIR, help='Input directory containing videos',
        required=True)
    parser.add_argument(
        "--" + Args.VISUALIZATIONS_DIR, help="Output directory", required=True)
    parser.add_argument(
        "--" + Args.RESULTS_DIR, required=True,
        help="Directory containing the saved result files")
    parser.add_argument(
        "--" + Args.LABEL_MAP, help="CSV file containing label names and ids",
        required=True)
    parser.add_argument(
        '--' + Args.CREATE_VIDEOS, dest=Args.CREATE_VIDEOS,
        help="Create videos with visualizations overlaid on them",
        action='store_true', default=False)
    parser.add_argument(
        '--no_' + Args.CREATE_VIDEOS, dest=Args.CREATE_VIDEOS,
        help="Do not create videos with visualizations overlaid on them, "
             "instead just copy the original video into the output folder.",
        action='store_false')
    parser.add_argument(
        '--' + Args.DEBUG_GRAPHS, dest=Args.DEBUG_GRAPHS, action='store_true',
        help="Include graphs with detailed intermediate outputs in the HTML "
             "visualization.", default=True)
    parser.add_argument(
        '--no_' + Args.DEBUG_GRAPHS, dest=Args.DEBUG_GRAPHS,
        help="Do not include graphs with detailed intermediate outputs in "
             "the HTML visualization.", action='store_false')
    args_dict = vars(parser.parse_args())
    return args_dict


def _get_in_video_path(
        input_videos_dir: Path, video_relative_path: Path) -> Path:
    in_video_dir = input_videos_dir / video_relative_path.parent
    in_video_short_name = video_relative_path.name
    in_video_paths = [
        p for p in in_video_dir.glob(in_video_short_name + "*")
        if is_video_path(p)]
    if len(in_video_paths) != 1:
        raise ValueError(
            f"There should be exactly one input video path matching the "
            f"video's relative path. The video_relative_path is: "
            f"{video_relative_path}, input paths: {in_video_paths}, "
            f"input video dir: {input_videos_dir}")
    return in_video_paths[0]


def _coarsen_video_recognized_actions(
        video_recognized_actions: VideoSoccerRecognizedActions) -> \
        VideoSoccerRecognizedActions:
    action_times = sorted(video_recognized_actions.keys())
    time_chunks = [
        action_times[start:(start + COARSE_ACTIONS_CHUNK_SIZE)]
        for start in range(0, len(action_times), COARSE_ACTIONS_CHUNK_SIZE)]
    return {
        np.mean(time_chunk): _max_recognized_action(
            [video_recognized_actions[time] for time in time_chunk])
        for time_chunk in time_chunks}


def _max_recognized_action(
        frame_recognized_actions_list: List[FrameRecognizedActions]) -> \
        FrameRecognizedActions:
    return FrameRecognizedActions(
        _concat_and_max(
            [frame_recognized_actions.scores for
             frame_recognized_actions in frame_recognized_actions_list]),
        _concat_and_max(
            [frame_recognized_actions.detections for
             frame_recognized_actions in frame_recognized_actions_list]
        ),
        _concat_and_max(
            [frame_recognized_actions.labels for
             frame_recognized_actions in frame_recognized_actions_list]
        )
    )


def _concat_and_max(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    return np.max(list_of_arrays, axis=0)


def _create_out_video(
        video: Video, in_video_path: Path, out_video_path: Path,
        should_create_videos: bool, label_map: LabelMap) -> None:
    if out_video_path.exists():
        print(f"Skipping creation of new {OUT_VIDEO_EXTENSION} video file for "
              f"visualization as it already exists at \n{out_video_path}")
    elif should_create_videos:
        print(f"Reading input video \n{in_video_path}\nand creating "
              f"output visualization at \n{out_video_path}")
        if video.recognized_actions:
            coarse_video_recognized_actions = \
                _coarsen_video_recognized_actions(video.recognized_actions)
            _run_timed_create_a_video(
                str(in_video_path), str(out_video_path),
                coarse_video_recognized_actions, label_map)
    else:
        # TODO: add some logic that creates new videos instead of copying if
        #  the input and output formats are incompatible.
        print(f"Copying input video from \n{in_video_path} to\n"
              f"{out_video_path}")
        shutil.copy(str(in_video_path), str(out_video_path))


def _run_timed_create_a_video(
        in_video_path: str, out_video_path: str,
        video_recognized_actions: VideoSoccerRecognizedActions,
        recognition_label_map: LabelMap) -> None:
    def create_a_video() -> None:
        _create_video(
            in_video_path, out_video_path, video_recognized_actions,
            recognition_label_map)
    run_timed(create_a_video)


def _create_video(
        in_video_path: str, out_video_path: str,
        video_recognized_actions: VideoSoccerRecognizedActions,
        recognition_label_map: LabelMap) -> None:

    def get_frame_labels(frame_time: float) -> Optional[List[Label]]:
        frame_actions_view = _get_frame_recognized_actions_view_from_video(
            frame_time, video_recognized_actions, recognition_label_map)
        if not frame_actions_view:
            return None
        return get_multi_labels(frame_actions_view, RECOGNITION_THRESHOLD)

    def draw_actions(
            np_image: np.ndarray, transformation_sizes: TransformationSizes,
            frame_time: float) -> np.ndarray:
        np_image = add_canvas_border(
            np_image, transformation_sizes.canvas_image_size)
        action_labels = get_frame_labels(frame_time)
        cv2_draw_labels(np_image, frame_time, action_labels)
        return np_image

    def draw_on_av_frame(
            in_frame: VideoFrame,
            transformation_sizes: TransformationSizes) -> VideoFrame:
        return apply_draw_to_av_frame(
            in_frame, transformation_sizes, draw_actions)

    with av.open(in_video_path) as in_container, \
            av.open(out_video_path, mode='w') as out_container:
        create_video_from_containers(
            in_container, out_container, draw_on_av_frame, add_border=True)
        # Close the files. Is this still needed, given the with statement?
        in_container.close()
        out_container.close()


def _get_frame_recognized_actions_view_from_video(
        frame_time: float,
        video_recognized_actions: VideoSoccerRecognizedActions,
        label_map: LabelMap) -> Optional[FrameRecognizedActionsView]:
    action_times = video_recognized_actions.keys()
    time_diff, best_time = min(
        (abs(action_time - frame_time), action_time)
        for action_time in action_times)
    if (time_diff >
            0.5 * COARSE_ACTIONS_CHUNK_SIZE * SOCCERNET_FEATURES_FREQUENCY):
        return None
    scores = video_recognized_actions[best_time].scores
    return FrameRecognizedActionsView(best_time, scores, label_map)


def _write_segmentation_html(
        video: Video, out_video_path: Path,
        video_dir: Path, label_map: LabelMap, debug_graphs: bool) -> None:
    html_path = video_dir / SEGMENTATION_HTML
    class_names = [
        label_map.int_to_label[label_index]
        for label_index in sorted(label_map.int_to_label.keys())]
    with html_path.open("w") as html_file:
        html_file.write('<div style="height:25%;text-align:center;">\n')
        add_video(
            html_file, out_video_path.relative_to(video_dir), video.video_id)
        html_file.write('</div>\n')
        html_file.write(
            '<div style="overflow-y:auto;height:74%;border:1px '
            'solid black;">')
        html_file.write('<table width="99%"><tr><td>\n')
        _add_segmentation_graphs(
            html_file, video.segmentation_results, class_names, video.video_id,
            debug_graphs)
        html_file.write('</td></tr></table>\n')
        html_file.write('</div>\n')


def _add_segmentation_graphs(
        html_file: TextIO, data_frame: DataFrame, class_names: List[str],
        video_id: str, debug_graphs: bool) -> None:
    convert_deltas_to_timestamps(data_frame)
    category_settings = create_custom_category_settings(
        class_names, ColorMapChoice.DARK24)
    add_segmentation_graph(html_file, data_frame, category_settings)
    if debug_graphs:
        add_segmentation_scores_graphs(html_file, data_frame, category_settings)
    add_click_code(html_file, video_id)


def _write_spotting_html(
        video: Video, out_video_path: Path, video_dir: Path,
        label_map: LabelMap, debug_graphs: bool) -> None:
    # class_names might be helpful when creating the intervals below,
    # but not currently used there.
    class_names = [
        label_map.int_to_label[label_index]
        for label_index in sorted(label_map.int_to_label.keys())]
    html_path = video_dir / SPOTTING_HTML
    with html_path.open("w") as html_file:
        html_file.write('<div style="height:25%;text-align:center;">\n')
        add_video(
            html_file, out_video_path.relative_to(video_dir), video.video_id)
        html_file.write('</div>\n')
        html_file.write(
            '<div style="overflow-y:auto;height:74%;border:1px '
            'solid black;">')
        html_file.write('<table width="99%"><tr><td>\n')
        _add_spotting_graphs(
            html_file, video.spotting_results, class_names, video.video_id,
            debug_graphs)
        html_file.write('</td></tr></table>\n')
        html_file.write('</div>\n')


def _add_spotting_graphs(
        html_file: TextIO, data_frame: DataFrame, class_names: List[str],
        video_id: str, debug_graphs: bool) -> None:
    category_settings = create_category_settings(class_names)
    convert_deltas_to_timestamps(data_frame)
    add_detections_graph(html_file, data_frame, category_settings)
    if debug_graphs:
        add_detection_scores_graphs(html_file, data_frame, category_settings)
        columns = [column for column in SPOTTING_COLUMN_NAMES
                   if column in data_frame.columns]
        add_predictions_graphs(
            html_file, data_frame, columns, category_settings)
    add_click_code(html_file, video_id)


if __name__ == "__main__":
    main()
