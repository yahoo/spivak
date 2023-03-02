# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from enum import Enum
from math import ceil
from pathlib import Path
from typing import TextIO, List, Dict

import numpy as np
import plotly.express as px
from pandas import DataFrame
from plotly.graph_objs import Figure

from spivak.html_visualization.result_data import COLUMN_CLASS, COLUMN_TIME

SOURCE_FLOAT_LABEL = 5.0
SOURCE_FLOAT_PREDICTION = 1.0
# HTML and JS code templates.
VIDEO_TEMPLATE = """<video id="{}" controls style="outline:none; height:100%;">
<source src="{}">
Your browser does not support the video tag.
</video>"""
PLOTLY_CLICK_SCRIPT_TEMPLATE = """
<script type="text/javascript">
elements = document.getElementsByClassName('plotly-graph-div');
console.log(elements);
function video_jump(data){{
    for(var i=0; i < data.points.length; i++){{
        // The Z denotes UTC in the date string.
        clickedDate = new Date(data.points[i].x + 'Z');
        clickedSeconds = clickedDate.getTime() / 1000.0;
    }}
    document.getElementById('{}').currentTime = clickedSeconds;
}}
for (const element of elements) {{
    element.on('plotly_click', video_jump);
}}
</script>
"""


class CategorySettings:

    def __init__(self, category_order: Dict[str, List[str]],
                 discrete_color_map: Dict[str, str]) -> None:
        self.category_order = category_order
        self.discrete_color_map = discrete_color_map


class ColorMapChoice(Enum):

    PLOTLY = 0
    DARK24 = 1
    LIGHT24 = 2
    ALPHABET = 3


def create_category_settings(categories: List[str]) -> CategorySettings:
    return create_custom_category_settings(categories, ColorMapChoice.DARK24)


def create_custom_category_settings(
        categories: List[str],
        colormap_choice: ColorMapChoice) -> CategorySettings:
    plotly_colors = _colormap_from_choice(colormap_choice)
    if len(categories) > len(plotly_colors):
        print(
            f"There are not enough colors in the color sequence as to support "
            f"the number of categories. Will repeat colors. Number of "
            f"categories: {len(categories)}, number of colors: "
            f"{len(plotly_colors)}")
        num_repetitions = ceil(len(categories) / len(plotly_colors))
        repeated_plotly_colors = num_repetitions * plotly_colors
        plotly_colors = repeated_plotly_colors[:len(categories)]
    category_order = {COLUMN_CLASS: categories}
    discrete_color_map = {
        category: plotly_colors[category_index]
        for category_index, category in enumerate(categories)}
    return CategorySettings(category_order, discrete_color_map)


def add_video(
        html_file: TextIO, video_html_relative_path: Path,
        video_id: str) -> None:
    _add_video_to_file(html_file, video_html_relative_path, video_id)
    html_file.write('\n<br>\n')


def add_click_code(html_file: TextIO, video_id: str) -> None:
    html_file.write(PLOTLY_CLICK_SCRIPT_TEMPLATE.format(video_id))


def adjust_subplot_xaxes(fig: Figure, n_categories: int) -> None:
    fig.update_xaxes(tickformat='%M:%S.%L', showticklabels=False)
    fig.update_xaxes(showticklabels=True, row=n_categories, col=1)


def extract_locations(
        category_data_frame: DataFrame, column: str, y_value: float):
    x = category_data_frame[COLUMN_TIME][category_data_frame[column] == 1]
    y = y_value * np.ones(len(x))
    return x, y


def _add_video_to_file(
        html_file: TextIO, video_html_relative_path: Path, video_id: str) -> None:
    html_file.write(VIDEO_TEMPLATE.format(video_id, video_html_relative_path))


def _colormap_from_choice(colormap_choice: ColorMapChoice):
    if colormap_choice == ColorMapChoice.DARK24:
        return px.colors.qualitative.Dark24
    if colormap_choice == ColorMapChoice.LIGHT24:
        return px.colors.qualitative.Light24
    elif colormap_choice == ColorMapChoice.PLOTLY:
        return px.colors.qualitative.Plotly
    elif colormap_choice == ColorMapChoice.ALPHABET:
        return px.colors.qualitative.Alphabet
    else:
        raise ValueError(f"Unknown colormap choice: {colormap_choice}")
