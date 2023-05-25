# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from heapq import heappush, heappop
from typing import Optional, Callable

import av
import cv2
import numpy as np
from av.audio.frame import AudioFrame
from av.audio.stream import AudioStream
from av.container import InputContainer, OutputContainer
from av.filter import Graph
from av.stream import Stream
from av.video.frame import VideoFrame
from av.video.stream import VideoStream

# Constants for PyAV decoding/encoding.
# aac is a popular decent audio codec that comes with ffmpeg.
OUT_AUDIO_CODEC = "aac"
OUT_SILENT_AUDIO_NUMPY_TYPE = "int16"
OUT_SILENT_AUDIO_RATE = 48000
OUT_SILENT_AUDIO_FORMAT = "s16p"
OUT_SILENT_AUDIO_LAYOUT = "stereo"
OUT_SILENT_AUDIO_NUMPY_SHAPE = (2, 1152)
# This codec (libx264) seems to work better with Chrome and also Jupyter
# notebook inside Chrome. I think I don't need to choose a frame-rate
# here now, as I will be setting the timestamps directly on the frames
# below.
OUT_VIDEO_CODEC = "libx264"
# Seems like yuv420p is the most common/popular format
OUT_STREAM_PIX_FMT = 'yuv420p'
OUT_VIDEO_BIT_RATE = 2e6
NP_IMAGE_FORMAT = 'rgb24'
TARGET_HEIGHT = 720
MAX_VIDEO_FRAME_QUEUE_LEN = 5
RECOGNITION_BORDER_RELATIVE_SIZE = 0.08
FILTER_NAME_SCALE = "scale"
FILTER_NAME_BUFFER_SINK = "buffersink"


class ImageSize:

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width


class TransformationSizes:

    def __init__(self, resizing_image_size: Optional[ImageSize],
                 canvas_image_size: Optional[ImageSize]) -> None:
        self.resizing_image_size = resizing_image_size
        self.canvas_image_size = canvas_image_size


DrawOnAvFrame = Callable[[VideoFrame, TransformationSizes], VideoFrame]
DrawActions = Callable[
    [np.ndarray, TransformationSizes, float], np.ndarray]
ApplyFilters = Callable[[VideoFrame], VideoFrame]


def create_video_from_containers(
        in_container: InputContainer, out_container: OutputContainer,
        draw_on_av_frame: DrawOnAvFrame, add_border: bool) -> None:
    # This function decodes both audio and video at once, in order to
    # better handle longer videos.
    in_video_stream = in_container.streams.video[0]
    in_video_stream.thread_type = 'AUTO'  # Decode faster
    transformation_sizes = _compute_transformation_sizes(
        in_video_stream, add_border)
    out_video_stream = _set_up_out_video_stream(
        out_container, transformation_sizes, in_video_stream)
    if in_container.streams.audio:
        in_audio_stream = in_container.streams.audio[0]
    else:
        in_audio_stream = None
    out_audio_stream = out_container.add_stream(OUT_AUDIO_CODEC)
    apply_filters = _create_filters(
        transformation_sizes.resizing_image_size, in_video_stream)
    # A priority queue is needed for the output in order to deal with frames
    # with pts that are not in increasing order.
    out_video_frame_queue = []
    for in_frame in in_container.decode(in_video_stream, in_audio_stream):
        if isinstance(in_frame, av.VideoFrame):
            filtered_frame = apply_filters(in_frame)
            out_frame = draw_on_av_frame(filtered_frame, transformation_sizes)
            heappush(out_video_frame_queue, (out_frame.pts, out_frame))
            while len(out_video_frame_queue) > MAX_VIDEO_FRAME_QUEUE_LEN:
                popped_frame = heappop(out_video_frame_queue)[1]
                add_video_frame(popped_frame, out_video_stream, out_container)
        elif isinstance(in_frame, av.AudioFrame):
            in_frame.pts = None
            add_audio_frame(in_frame, out_audio_stream, out_container)
    # Empty the output video queue.
    while out_video_frame_queue:
        popped_frame = heappop(out_video_frame_queue)[1]
        add_video_frame(popped_frame, out_video_stream, out_container)
    if not in_audio_stream:
        add_silent_audio(out_audio_stream, out_container)
    flush_stream(out_video_stream, out_container)
    flush_stream(out_audio_stream, out_container)


def _compute_transformation_sizes(
        in_video_stream: VideoStream, add_border: bool) -> TransformationSizes:
    input_width = in_video_stream.width
    input_height = in_video_stream.height
    scaling = TARGET_HEIGHT / float(input_height)
    resized_width = _make_even(scaling * input_width)
    resized_height = TARGET_HEIGHT
    if resized_height != input_height or resized_width != input_width:
        resizing_image_size = ImageSize(resized_height, resized_width)
    else:
        resizing_image_size = None
    if not add_border:
        canvas_image_size = None
    else:
        extra_canvas_height = RECOGNITION_BORDER_RELATIVE_SIZE * resized_height
        extra_canvas_height = _make_even(extra_canvas_height)
        canvas_image_size = ImageSize(
            resized_height + extra_canvas_height, resized_width)
    return TransformationSizes(resizing_image_size, canvas_image_size)


def _make_even(dimension: float) -> int:
    """The x264 codec requires the video width and height to be multiples of
    2."""
    return 2 * (round(dimension) // 2)


def _set_up_out_video_stream(
        out_container: OutputContainer,
        transformation_sizes: TransformationSizes,
        in_video_stream: VideoStream) -> VideoStream:
    out_video_stream = out_container.add_stream(OUT_VIDEO_CODEC)
    out_video_stream.thread_type = 'AUTO'  # Encode faster
    out_video_stream.pix_fmt = OUT_STREAM_PIX_FMT
    if transformation_sizes.canvas_image_size:
        out_video_stream.width = transformation_sizes.canvas_image_size.width
        out_video_stream.height = transformation_sizes.canvas_image_size.height
    elif transformation_sizes.resizing_image_size:
        out_video_stream.width = transformation_sizes.resizing_image_size.width
        out_video_stream.height = \
            transformation_sizes.resizing_image_size.height
    else:
        out_video_stream.width = in_video_stream.width
        out_video_stream.height = in_video_stream.height
    out_video_stream.time_base = in_video_stream.time_base
    out_video_stream.bit_rate = OUT_VIDEO_BIT_RATE
    return out_video_stream


def add_silent_audio(out_audio_stream: AudioStream,
                     out_container: OutputContainer) -> None:
    # This is pretty hacky, as it's adding a single audio frame for the whole
    # video. The idea is just to make sure all output videos have an audio
    # track, so that it is easier to concatenate them later.
    silent_frame = create_silent_frame()
    add_audio_frame(silent_frame, out_audio_stream, out_container)


def create_silent_frame() -> AudioFrame:
    silence = np.zeros(
        OUT_SILENT_AUDIO_NUMPY_SHAPE, dtype=OUT_SILENT_AUDIO_NUMPY_TYPE)
    audio_frame = AudioFrame.from_ndarray(
        silence, format=OUT_SILENT_AUDIO_FORMAT, layout=OUT_SILENT_AUDIO_LAYOUT)
    audio_frame.pts = None
    audio_frame.rate = OUT_SILENT_AUDIO_RATE
    return audio_frame


def add_audio_frame(out_frame: AudioFrame, out_audio_stream: AudioStream,
                    out_container: OutputContainer) -> None:
    for packet in out_audio_stream.encode(out_frame):
        if packet.dts is None:
            continue
        # We need to assign the packet to the output stream.
        packet.stream = out_audio_stream
        out_container.mux(packet)


def add_video_frame(
        out_frame: VideoFrame, out_video_stream: VideoStream,
        out_container: OutputContainer) -> None:
    for packet in out_video_stream.encode(out_frame):
        packet.stream = out_video_stream
        out_container.mux(packet)


def flush_stream(out_stream: Stream, out_container: OutputContainer) -> None:
    for packet in out_stream.encode():
        out_container.mux(packet)


def apply_draw_to_av_frame(
        in_frame: VideoFrame, transformation_sizes: TransformationSizes,
        draw_actions: DrawActions) -> VideoFrame:
    np_image = in_frame.to_ndarray(format=NP_IMAGE_FORMAT)
    np_image = draw_actions(np_image, transformation_sizes, in_frame.time)
    out_frame = av.VideoFrame.from_ndarray(np_image, format=NP_IMAGE_FORMAT)
    out_frame.pts = in_frame.pts
    out_frame.time_base = in_frame.time_base
    return out_frame


def add_canvas_border(
        np_image: np.ndarray,
        canvas_image_size: Optional[ImageSize]) -> np.ndarray:
    if not canvas_image_size:
        return np_image
    extra_height = (canvas_image_size.height - np_image.shape[0])
    return cv2.copyMakeBorder(np_image, 0, extra_height, 0, 0,
                              cv2.BORDER_CONSTANT, 0)


def _create_filters(
        resizing_image_size: Optional[ImageSize],
        in_video_stream: VideoStream) -> ApplyFilters:
    if not resizing_image_size:
        def apply_no_filters(in_frame: VideoFrame) -> VideoFrame:
            return in_frame
        return apply_no_filters
    graph = Graph()
    buffer = graph.add_buffer(template=in_video_stream)
    size_string = (str(resizing_image_size.width) + ":" +
                   str(resizing_image_size.height))
    scale = graph.add(FILTER_NAME_SCALE, size_string)
    sink = graph.add(FILTER_NAME_BUFFER_SINK)
    buffer.link_to(scale)
    scale.link_to(sink)
    graph.configure()

    def apply_graph_filters(in_frame: VideoFrame) -> VideoFrame:
        graph.push(in_frame)
        filtered_frame = graph.pull()
        filtered_frame.time_base = in_frame.time_base
        return filtered_frame
    return apply_graph_filters
