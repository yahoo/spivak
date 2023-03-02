# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

from typing import TextIO, Dict

import numpy as np
import pandas
from pandas import DataFrame
from plotly import express as px
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from spivak.html_visualization.result_data import COLUMN_CLASS, \
    COLUMN_SOURCE_FLOAT, COLUMN_TIME, COLUMN_SOURCE, \
    COLUMN_SEGMENTATION_LABEL, COLUMN_SEGMENTATION_SCORE
from spivak.html_visualization.utils import CategorySettings, \
    adjust_subplot_xaxes, extract_locations

SEGMENTATION_HEIGHT = 350
SEGMENTATION_TITLE = "Segmentation prediction and labels"
SEGMENTATION_PREDICTION_TICK_LABEL = "Segmentation prediction"
SUBPLOT_VERTICAL_SPACING = 0.004
SEGMENTATION_SCORES_HEIGHT = 1000
SEGMENTATION_SCORES_TITLE = "Segmentation scores with predictions and labels"
LABEL_SUFFIX = "_t"
PREDICTION_SUFFIX = "_p"
SCORES_RANGE_EXTRA = [0.0, 1.5]
Y_LABELS = 1.35
Y_PREDICTIONS = 1.15
SOURCE_FLOAT_PREDICTION = 1.0
SOURCE_FLOAT_LABEL = 3.0


def add_segmentation_graph(
        html_file: TextIO, data_frame: DataFrame,
        category_settings: CategorySettings) -> None:
    fig = _plot_segmentation_and_labels(
        data_frame, category_settings, SEGMENTATION_HEIGHT)
    fig.update_layout(
        title_text=SEGMENTATION_TITLE, legend=dict(itemsizing="constant"))
    time_range = [data_frame[COLUMN_TIME].head(1).item(),
                  data_frame[COLUMN_TIME].tail(1).item()]
    fig.update_xaxes(tickformat='%M:%S.%L', range=time_range)
    y_tick_values = [SOURCE_FLOAT_PREDICTION, SOURCE_FLOAT_LABEL]
    y_range = [min(y_tick_values) - 3.0, max(y_tick_values) + 3.0]
    y_tick_texts = [
        SEGMENTATION_PREDICTION_TICK_LABEL, COLUMN_SEGMENTATION_LABEL]
    fig.update_yaxes(
        range=y_range, tickvals=y_tick_values, ticktext=y_tick_texts,
        title_text=COLUMN_SOURCE)
    fig.write_html(html_file)


def add_segmentation_scores_graphs(
        html_file: TextIO, data_frame: DataFrame,
        category_settings: CategorySettings) -> None:
    n_categories = len(category_settings.discrete_color_map)
    predictions_data_frame = _max_prediction_per_time_instant(data_frame)
    fig = make_subplots(
        rows=n_categories, cols=1, shared_xaxes=True,
        vertical_spacing=SUBPLOT_VERTICAL_SPACING)
    for category_index, category in enumerate(
            category_settings.category_order[COLUMN_CLASS]):
        row = category_index + 1
        category_color = category_settings.discrete_color_map[category]
        category_data_frame = data_frame[data_frame[COLUMN_CLASS] == category]
        category_predictions_data_frame = predictions_data_frame[
            predictions_data_frame[COLUMN_CLASS] == category]
        _fig_add_segmentation_scores(
            fig, category_data_frame, category, category_color, row)
        _fig_add_predictions(
            fig, category_predictions_data_frame, category + PREDICTION_SUFFIX,
            category_color, row)
        _fig_add_labels(
            fig, category_data_frame, category + LABEL_SUFFIX, "black", row)
    fig.update_yaxes(
        range=SCORES_RANGE_EXTRA, showticklabels=False, visible=True)
    adjust_subplot_xaxes(fig, n_categories)
    fig.update_layout(
        title_text=SEGMENTATION_SCORES_TITLE, height=SEGMENTATION_SCORES_HEIGHT,
        legend=dict(itemsizing="constant"))
    fig.write_html(html_file)


def _plot_segmentation_and_labels(
        data_frame: DataFrame, category_settings: CategorySettings,
        height: int) -> Figure:
    # Get predictions from data frame
    predictions_data_frame = _max_prediction_per_time_instant(data_frame)
    predictions_data_frame[COLUMN_SOURCE_FLOAT] = SOURCE_FLOAT_PREDICTION
    # Get labels from data frame
    label_idx = data_frame[COLUMN_SEGMENTATION_LABEL] == 1
    labels_data_frame = data_frame.loc[label_idx].copy()
    labels_data_frame[COLUMN_SOURCE_FLOAT] = SOURCE_FLOAT_LABEL
    # Concatenate the segmentation predictions with the labels, then plot.
    predictions_and_labels_data_frame = pandas.concat(
        [predictions_data_frame, labels_data_frame])
    fig = px.scatter(
        predictions_and_labels_data_frame, x=COLUMN_TIME,
        y=COLUMN_SOURCE_FLOAT, color=COLUMN_CLASS, height=height,
        category_orders=category_settings.category_order,
        color_discrete_map=category_settings.discrete_color_map)
    fig.update_traces(
        marker=dict(symbol="line-ns-open", size=35, line=dict(width=1)),
        selector=dict(mode="markers"))
    return fig


def _max_prediction_per_time_instant(data_frame: DataFrame) -> DataFrame:
    max_score_per_frame_idx = (
        data_frame.groupby([COLUMN_TIME])[COLUMN_SEGMENTATION_SCORE].idxmax())
    return data_frame.loc[max_score_per_frame_idx]


def _fig_add_segmentation_scores(
        fig: Figure, category_data_frame: DataFrame, category: str,
        category_color: str, row: int) -> None:
    line_dict = dict(color=category_color)
    fig.add_scatter(
        x=category_data_frame[COLUMN_TIME],
        y=category_data_frame[COLUMN_SEGMENTATION_SCORE], name=category,
        line=line_dict, col=1, row=row)


def _fig_add_labels(
        fig: Figure, category_data_frame: DataFrame, name: str, color: str,
        row: int) -> None:
    x, y = extract_locations(
        category_data_frame, COLUMN_SEGMENTATION_LABEL, Y_LABELS)
    label_marker = _create_label_marker(color)
    fig.add_scatter(
        x=x, y=y, col=1, row=row, name=name, mode="markers",
        marker=label_marker)


def _fig_add_predictions(
        fig: Figure, category_predictions_data_frame: DataFrame, name: str,
        color: str, row: int) -> None:
    x = category_predictions_data_frame[COLUMN_TIME]
    y = Y_PREDICTIONS * np.ones(len(x))
    prediction_marker = _create_prediction_marker(color)
    fig.add_scatter(
        x=x, y=y, col=1, row=row, name=name, mode="markers",
        marker=prediction_marker)


def _create_label_marker(color: str) -> Dict:
    return dict(
        size=2, color=color, line_color=color, symbol="circle", line_width=0)


def _create_prediction_marker(color: str) -> Dict:
    return dict(
        size=2, color=color, line_color=color, symbol="circle", line_width=0)
