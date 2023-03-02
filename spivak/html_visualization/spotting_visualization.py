# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from typing import TextIO, List, Dict

import pandas
import plotly.express as px
from pandas import DataFrame
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from spivak.html_visualization.result_data import COLUMN_TIME, \
    COLUMN_DETECTION, COLUMN_SPOTTING_LABEL, COLUMN_SOURCE, COLUMN_CLASS, \
    COLUMN_SOURCE_FLOAT, COLUMN_DETECTION_SCORE, COLUMN_DELTA, \
    COLUMN_DETECTION_SCORE_NMS
from spivak.html_visualization.utils import CategorySettings, \
    SOURCE_FLOAT_PREDICTION, SOURCE_FLOAT_LABEL, adjust_subplot_xaxes, \
    extract_locations

SCORES_RANGE = [0.0, 1.0]
SCORES_RANGE_EXTRA = [0.0, 1.5]
DETECTIONS_HEIGHT = 300
DETECTION_SCORES_HEIGHT = 1000
DETECTION_SCORES_TITLE = "Detection scores with detections and labels"
DETECTIONS_TITLE = "Detections and labels"
PREDICTIONS_HEIGHT = 2500
PREDICTIONS_TITLE = (
    "Different intermediate outputs along with detections and labels")
SUBPLOT_VERTICAL_SPACING = 0.004
LABEL_SUFFIX = "_t"
DETECTION_SUFFIX = "_d"
PREDICTIONS_Y_DETECTIONS = 1.25
PREDICTIONS_Y_LABELS = 1.25


def add_detection_scores_graphs(
        html_file: TextIO, data_frame: DataFrame,
        category_settings: CategorySettings) -> None:
    n_categories = len(category_settings.discrete_color_map)
    fig = make_subplots(
        rows=n_categories, cols=1, shared_xaxes=True,
        vertical_spacing=SUBPLOT_VERTICAL_SPACING)
    for category_index, category in enumerate(
            category_settings.category_order[COLUMN_CLASS]):
        row = category_index + 1
        category_color = category_settings.discrete_color_map[category]
        category_data_frame = data_frame[data_frame[COLUMN_CLASS] == category]
        _fig_add_detection_scores(
            fig, category_data_frame, category, category_color, row)
        _fig_add_detections(fig, category_data_frame,
                            category + DETECTION_SUFFIX, category_color, row)
        if COLUMN_SPOTTING_LABEL in category_data_frame.columns:
            _fig_add_labels(fig, category_data_frame, category + LABEL_SUFFIX,
                            category_color, row)
    fig.update_yaxes(range=SCORES_RANGE_EXTRA, showticklabels=False,
                     visible=True)
    adjust_subplot_xaxes(fig, n_categories)
    fig.update_layout(
        title_text=DETECTION_SCORES_TITLE, height=DETECTION_SCORES_HEIGHT)
    fig.write_html(html_file)


def add_predictions_graphs(
        html_file: TextIO, data_frame: DataFrame, columns: List[str],
        category_settings: CategorySettings) -> None:
    n_categories = len(category_settings.discrete_color_map)
    category_order = category_settings.category_order[COLUMN_CLASS]
    column_color_map = _create_column_color_map(
        columns + [COLUMN_SPOTTING_LABEL, COLUMN_DETECTION])
    detection_color = column_color_map[COLUMN_DETECTION]
    fig = make_subplots(
        rows=n_categories, cols=1, shared_xaxes=True,
        vertical_spacing=SUBPLOT_VERTICAL_SPACING,
        subplot_titles=category_order,
        specs=n_categories * [[{"secondary_y": True}]])
    for category_index, category in enumerate(category_order):
        row = category_index + 1
        show_legend = (category_index == 0)
        category_data_frame = data_frame[data_frame[COLUMN_CLASS] == category]
        _fig_add_predictions(
            fig, category_data_frame, columns, column_color_map, show_legend,
            row)
        _fig_add_detections_grouped(
            fig, category_data_frame, COLUMN_DETECTION, detection_color,
            show_legend, row)
        if COLUMN_SPOTTING_LABEL in category_data_frame.columns:
            label_color = column_color_map[COLUMN_SPOTTING_LABEL]
            _fig_add_labels_grouped(
                fig, category_data_frame, COLUMN_SPOTTING_LABEL, label_color,
                show_legend, row)
    # Move subplot labels to the right and make them smaller.
    fig.update_annotations(
        textangle=-90, x=1.0, yshift=-55, yanchor="middle", xanchor="left",
        font=dict(size=12))
    adjust_subplot_xaxes(fig, n_categories)
    fig.update_layout(
        title_text=PREDICTIONS_TITLE, height=PREDICTIONS_HEIGHT)
    fig.write_html(html_file)


def add_detections_graph(
        html_file: TextIO, data_frame: DataFrame,
        category_settings: CategorySettings) -> None:
    fig = _plot_all_detections_and_labels(
        data_frame, category_settings, DETECTIONS_HEIGHT)
    fig.update_layout(title_text=DETECTIONS_TITLE)
    time_range = [data_frame[COLUMN_TIME].head(1).item(),
                  data_frame[COLUMN_TIME].tail(1).item()]
    fig.update_xaxes(tickformat='%M:%S.%L', range=time_range)
    y_tick_values = [SOURCE_FLOAT_PREDICTION, SOURCE_FLOAT_LABEL]
    y_range = [min(y_tick_values) - 2.0, max(y_tick_values) + 2.0]
    y_tick_texts = [COLUMN_DETECTION, COLUMN_SPOTTING_LABEL]
    fig.update_yaxes(
        range=y_range, tickvals=y_tick_values, ticktext=y_tick_texts,
        title_text=COLUMN_SOURCE)
    fig.write_html(html_file)


def _fig_add_detection_scores(
        fig: Figure, category_data_frame: DataFrame, category: str,
        category_color: str, row: int) -> None:
    line_dict = dict(color=category_color)
    fig.add_scatter(
        x=category_data_frame[COLUMN_TIME],
        y=category_data_frame[COLUMN_DETECTION_SCORE], name=category,
        line=line_dict, col=1, row=row)


def _fig_add_predictions(
        fig: Figure, category_data_frame: DataFrame, columns: List[str],
        column_color_map: Dict[str, str],  show_legend: bool, row: int) -> None:
    for column in columns:
        if column == COLUMN_DELTA:
            marker_dict = dict(color=column_color_map[column], size=2)
            fig.add_scatter(
                x=category_data_frame[COLUMN_TIME],
                y=category_data_frame[column], name=column,
                legendgroup=column, mode="markers", marker=marker_dict,
                showlegend=show_legend, col=1, row=row, secondary_y=True)
        elif column == COLUMN_DETECTION_SCORE_NMS:
            marker_dict = _create_nms_marker()
            filtered_data_frame = category_data_frame[
                category_data_frame[column] != -1.0]
            fig.add_scatter(
                x=filtered_data_frame[COLUMN_TIME],
                y=filtered_data_frame[column], name=column,
                legendgroup=column, mode="markers", marker=marker_dict,
                showlegend=show_legend, col=1, row=row)
        else:
            line_dict = dict(color=column_color_map[column])
            fig.add_scatter(
                x=category_data_frame[COLUMN_TIME],
                y=category_data_frame[column], name=column,
                legendgroup=column, mode="lines", line=line_dict,
                showlegend=show_legend, col=1, row=row)


def _fig_add_labels_grouped(
        fig: Figure, category_data_frame: DataFrame, name: str, color: str,
        show_legend: bool, row: int) -> None:
    x, y = extract_locations(
        category_data_frame, COLUMN_SPOTTING_LABEL, PREDICTIONS_Y_LABELS)
    label_marker = _create_label_marker(color)
    fig.add_scatter(
        x=x, y=y, col=1,  row=row, showlegend=show_legend,
        name=name, legendgroup=name, mode="markers", marker=label_marker)


def _fig_add_labels(
        fig: Figure, category_data_frame: DataFrame, name: str, color: str,
        row: int) -> None:
    x, y = extract_locations(
        category_data_frame, COLUMN_SPOTTING_LABEL, PREDICTIONS_Y_LABELS)
    label_marker = _create_label_marker(color)
    fig.add_scatter(
        x=x, y=y, col=1, row=row, name=name, mode="markers",
        marker=label_marker)


def _fig_add_detections_grouped(
        fig: Figure, category_data_frame: DataFrame, name: str,
        color: str, show_legend: bool, row: int) -> None:
    x, y = extract_locations(
        category_data_frame, COLUMN_DETECTION, PREDICTIONS_Y_DETECTIONS)
    detection_marker = _create_detection_marker(color)
    fig.add_scatter(
        x=x, y=y, col=1, row=row, showlegend=show_legend, name=name,
        legendgroup=name, mode="markers", marker=detection_marker)


def _fig_add_detections(
        fig: Figure, category_data_frame: DataFrame, name: str,
        color: str, row: int) -> None:
    x, y = extract_locations(
        category_data_frame, COLUMN_DETECTION, PREDICTIONS_Y_DETECTIONS)
    detection_marker = _create_detection_marker(color)
    fig.add_scatter(
        x=x, y=y, col=1, row=row, name=name, mode="markers",
        marker=detection_marker)


def _create_column_color_map(columns) -> Dict[str, str]:
    plotly_colors = px.colors.qualitative.Dark24
    return {
        column: plotly_colors[column_index]
        for column_index, column in enumerate(columns)}


def _plot_all_detections_and_labels(
        data_frame: DataFrame, category_settings: CategorySettings,
        height: int) -> Figure:
    # Get detections from data frame
    detections_and_labels_data_frame = data_frame[
        data_frame[COLUMN_DETECTION] == 1].copy()
    detections_and_labels_data_frame[COLUMN_SOURCE_FLOAT] = \
        SOURCE_FLOAT_PREDICTION
    # Get labels from data frame
    if COLUMN_SPOTTING_LABEL in data_frame.columns:
        labels_data_frame = data_frame[
            data_frame[COLUMN_SPOTTING_LABEL] == 1].copy()
        labels_data_frame[COLUMN_SOURCE_FLOAT] = SOURCE_FLOAT_LABEL
        # Concatenate the detections with the labels, then plot.
        detections_and_labels_data_frame = pandas.concat(
            [detections_and_labels_data_frame, labels_data_frame])
    fig = px.scatter(
        detections_and_labels_data_frame, x=COLUMN_TIME,
        y=COLUMN_SOURCE_FLOAT, color=COLUMN_CLASS, height=height,
        category_orders=category_settings.category_order,
        color_discrete_map=category_settings.discrete_color_map)
    fig.update_traces(
        marker=dict(size=10, symbol="x"), selector=dict(mode='markers'))
    return fig


def _create_nms_marker() -> Dict:
    return dict(size=7, color="black", symbol="cross-thin", line_width=2)


def _create_detection_marker(color: str) -> Dict:
    return dict(
        size=7, color=color, line_color=color, symbol="x-thin", line_width=2)


def _create_label_marker(color: str) -> Dict:
    return dict(
        size=8, color=color, line_color=color, symbol="circle-open",
        line_width=2)
