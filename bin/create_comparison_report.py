#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import pickle
from itertools import cycle
from pathlib import Path
from typing import Dict, List, TextIO, Optional

import numpy as np
import pandas
import plotly.io as pio
from pandas import DataFrame

# This fixes the issue with the pdf plots having some text in the bottom,
# though I think it also blocks the use of math symbols.
pio.kaleido.scope.mathjax = None
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure

from spivak.data.label_map import LabelMap
from spivak.evaluation.aggregate import EVALUATION_AGGREGATE_PICKLE_FILE_NAME, \
    EvaluationAggregate
from spivak.evaluation.spotting_evaluation import SpottingEvaluation
from spivak.application.validation import EvaluationRun, \
    EVALUATION_RUN_PICKLE_FILE_NAME

INDEX_HTML = "index.html"
CREATE_PAPER_PDFS = True
METHOD_SYMBOL_SEQUENCE = [
    "circle-open", "square-open", "diamond-open", "cross-thin-open",
    "x-thin-open"]
# Average mAP graph
AVERAGE_MAP_GRAPH_TITLE = "Average mAP"
AVERAGE_MAP_GRAPH_HEIGHT = 400
AVERAGE_MAP_GRAPH_WIDTH = 500
# Class graph
CLASS_GRAPH_TITLE = "Average AP"
CLASS_GRAPH_HEIGHT = 500
CLASS_GRAPH_WIDTH = 1200
# AP per tolerance graph
AP_PER_TOLERANCE_GRAPH_TITLE = \
    "AP as a function of class (specific to the chosen matching tolerance)"
AP_PER_CLASS_TOLERANCE_GRAPH_HEIGHT = 600
AP_PER_TOLERANCE_GRAPH_WIDTH = 1300
# Tolerance graph
TOLERANCE_GRAPH_TITLE = "mAP as a function of matching tolerance"
TOLERANCE_MEDIUM_GRAPH_TITLE = TOLERANCE_GRAPH_TITLE
TOLERANCE_SMALL_GRAPH_TITLE = "mAP over small matching tolerances"
TOLERANCE_GRAPH_HEIGHT = 500
TOLERANCE_GRAPH_WIDTH = 800
TOLERANCE_GRAPH_PDF_HEIGHT = 400
TOLERANCE_GRAPH_PDF_WIDTH = 550
TOLERANCE_MEDIUM_GRAPH_PDF_HEIGHT = 430
TOLERANCE_MEDIUM_GRAPH_PDF_WIDTH = 600
TOLERANCE_GRAPH_PDF_FILE = "tolerance_graph.pdf"
TOLERANCE_MEDIUM_GRAPH_PDF_FILE = "tolerance_medium_graph.pdf"
TOLERANCE_SMALL_GRAPH_PDF_FILE = "tolerance_small_graph.pdf"
PDF_TOLERANCE_GRAPH_SETTINGS = [
    (TOLERANCE_GRAPH_TITLE, TOLERANCE_GRAPH_PDF_FILE, None, True,
     TOLERANCE_GRAPH_PDF_HEIGHT, TOLERANCE_GRAPH_PDF_WIDTH),
    (TOLERANCE_MEDIUM_GRAPH_TITLE, TOLERANCE_MEDIUM_GRAPH_PDF_FILE, 20.0,
     True, TOLERANCE_MEDIUM_GRAPH_PDF_HEIGHT, TOLERANCE_MEDIUM_GRAPH_PDF_WIDTH),
    (TOLERANCE_SMALL_GRAPH_TITLE, TOLERANCE_SMALL_GRAPH_PDF_FILE, 10.0,
     True, TOLERANCE_GRAPH_PDF_HEIGHT, TOLERANCE_GRAPH_PDF_WIDTH)]
# AP per class graph
AP_PER_CLASS_GRAPH_TITLE = \
    "AP as a function of matching tolerance (specific to the chosen class)"
AP_PER_CLASS_CLASS_GRAPH_HEIGHT = 500
AP_PER_CLASS_GRAPH_WIDTH = 900
# Precision-recall graphs
PR_GRAPH_TITLE = "Precision-recall curves"
PR_GRAPH_HEIGHT = 600
PR_GRAPH_WIDTH = 1000
# Precision-recall graphs per tolerance, aggregated over classes.
PR_PER_TOLERANCE_GRAPH_TITLE = \
    "Precision-recall curves, micro-averaged over classes"
PR_PER_TOLERANCE_GRAPH_HEIGHT = 600
PR_PER_TOLERANCE_GRAPH_WIDTH = 800
# Shared
MATCHING_TOLERANCE_AXIS_TITLE = "Matching tolerance (seconds)"
CLASS_AXIS_TITLE = "Class"
AP_AXIS_TITLE = "AP"
MAP_AXIS_TITLE = "mAP"
AVERAGE_AP_AXIS_TITLE = "Average AP"
PRECISION_AXIS_TITLE = "Precision"
RECALL_AXIS_TITLE = "Recall"
TOLERANCE_TICKS = [0.0, 5.0, 10.0, 20.0, 40.0, 60.0]
PR_RECALL_RANGE = [-0.1, 1.1]
PR_PRECISION_RANGE = [-0.05, 1.05]
# New DataFrame columns
AVERAGE_MAP = "Average mAP"
AVERAGE_AP = "Average AP"
METHOD = "Method"
TOLERANCES = "Tolerances"


class Args:
    OUT_DIR = "out_dir"
    PICKLES = "pickles"
    NAMES = "names"


def main() -> None:
    args = _get_command_line_arguments()
    out_dir = Path(args[Args.OUT_DIR])
    pickle_paths = [Path(p) for p in args[Args.PICKLES]]
    methods = args[Args.NAMES]
    assert len(methods) == len(pickle_paths)
    evaluation_runs = _read_evaluation_runs(pickle_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / INDEX_HTML
    first_spotting_evaluation = \
        evaluation_runs[0].evaluation.spotting_evaluation
    ordered_class_names = (
        first_spotting_evaluation.label_map.display_ordered_labels)
    with html_path.open("w") as html_file:
        html_file.write("Classes used when computing means (mAP)")
        _write_selected_classes(
            html_file, first_spotting_evaluation.selected_classes,
            first_spotting_evaluation.label_map)
        _write_tolerances(
            html_file, first_spotting_evaluation.tolerances_in_seconds_dict)
        average_map_fig = _create_average_map_graph(evaluation_runs, methods)
        average_map_fig.write_html(html_file)
        class_fig = _create_class_graph(
            evaluation_runs, methods, ordered_class_names)
        class_fig.write_html(html_file)
        ap_data_frame = _create_ap_data_frame(evaluation_runs, methods)
        ap_per_tolerance_fig = _create_ap_per_tolerance_graph(
            ap_data_frame, methods, ordered_class_names)
        ap_per_tolerance_fig.write_html(html_file)
        tolerance_fig = _create_tolerance_graph(
            evaluation_runs, methods, None, TOLERANCE_GRAPH_HEIGHT,
            TOLERANCE_GRAPH_WIDTH, True, TOLERANCE_GRAPH_TITLE,
            pdf=False)
        tolerance_fig.write_html(html_file)
        if CREATE_PAPER_PDFS:
            for (title, file_name, tolerance_limit, show_legend, height,
                 width) in PDF_TOLERANCE_GRAPH_SETTINGS:
                tolerance_pdf_fig = _create_tolerance_graph(
                    evaluation_runs, methods, tolerance_limit, height, width,
                    show_legend, title, pdf=True)
                tolerance_pdf_fig.write_image(out_dir / file_name)
        ap_per_class_fig = _create_ap_per_class_graph(
            ap_data_frame, methods, ordered_class_names)
        ap_per_class_fig.write_html(html_file)
        pr_data_frame = _create_pr_data_frame(evaluation_runs, methods)
        pr_per_class_and_tolerance_fig = \
            _create_pr_per_class_and_tolerance_graph(
                pr_data_frame, methods, ordered_class_names)
        pr_per_class_and_tolerance_fig.write_html(html_file)
        if first_spotting_evaluation.confusion_data_frame is not None:
            label_map = first_spotting_evaluation.label_map
            int_to_label = label_map.int_to_label
            n_labels = len(int_to_label)
            pr_selected_class_sets = [{
                int_to_label[i] for i in range(n_labels)}]
            for pr_class_set in pr_selected_class_sets:
                pr_selected_classes = [
                    int_to_label[i] in pr_class_set for i in range(n_labels)]
                html_file.write(
                    "Classes used when computing precision-recall curves")
                _write_selected_classes(
                    html_file, pr_selected_classes, label_map)
                pr_per_tolerance_data_frame = \
                    _create_pr_per_tolerance_data_frame(
                        evaluation_runs, methods, pr_selected_classes)
                pr_per_tolerance_fig = _create_pr_per_tolerance_graph(
                    pr_per_tolerance_data_frame, methods)
                pr_per_tolerance_fig.write_html(html_file)


def _write_selected_classes(
        html_file: TextIO, selected_classes: List[bool],
        label_map: LabelMap) -> None:
    selected_label_set = {
        label_map.int_to_label[class_index]
        for class_index, class_is_selected in enumerate(selected_classes)
        if class_is_selected}
    ordered_selected_labels = [
        label for label in label_map.display_ordered_labels
        if label in selected_label_set]
    html_file.write("<br>Classes used: ")
    html_file.write(", ".join(ordered_selected_labels))
    ordered_excluded_labels = [
        label for label in label_map.display_ordered_labels
        if label not in selected_label_set]
    html_file.write("<br>Classes excluded: ")
    html_file.write(", ".join(ordered_excluded_labels))


def _write_tolerances(
        html_file: TextIO,
        tolerances_in_seconds_dict: Dict[str, np.ndarray]) -> None:
    for tolerances_name, tolerances_in_seconds in \
            tolerances_in_seconds_dict.items():
        html_file.write(
            f"<br>{tolerances_name} tolerances (in seconds) used when "
            f"computing averages: ")
        html_file.write(str(tolerances_in_seconds))


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.OUT_DIR, help="Output directory", required=True)
    parser.add_argument(
        "--" + Args.PICKLES,
        help="Input pickle files containing EvaluationRun objects",
        required=True, nargs="+")
    parser.add_argument(
        "--" + Args.NAMES,
        help="Names of each of the runs, to be used in the graph legends",
        required=True, nargs="+")
    args_dict = vars(parser.parse_args())
    return args_dict


def _read_evaluation_runs(pickle_paths: List[Path]) -> List[EvaluationRun]:
    return [_read_evaluation_run(pickle_path) for pickle_path in pickle_paths]


def _read_evaluation_run(pickle_path: Path) -> EvaluationRun:
    if pickle_path.name == EVALUATION_RUN_PICKLE_FILE_NAME:
        with pickle_path.open("rb") as pickle_file:
            evaluation_run: EvaluationRun = pickle.load(pickle_file)
    elif pickle_path.name == EVALUATION_AGGREGATE_PICKLE_FILE_NAME:
        with pickle_path.open("rb") as pickle_file:
            evaluation_aggregate: EvaluationAggregate = pickle.load(pickle_file)
            evaluation_run = EvaluationRun(evaluation_aggregate, None)
    else:
        raise ValueError(
            f"Could not determine the type of the evaluation pickle from file"
            f" name: {pickle_path}")
    # Ad-hoc step to rename labels for plots, if needed.
    # evaluation_run.evaluation.spotting_evaluation.label_map = _rename_labels(
    #     evaluation_run.evaluation.spotting_evaluation.label_map)
    return evaluation_run


def _create_ap_data_frame(
        evaluation_runs: List[EvaluationRun], methods: List[str]) -> DataFrame:
    data_frames = []
    for evaluation_run, method_name in zip(evaluation_runs, methods):
        spotting_evaluation = evaluation_run.evaluation.spotting_evaluation
        ap_data_frame = spotting_evaluation.ap_data_frame
        ap_data_frame = _add_class_name_mapping(
            ap_data_frame, spotting_evaluation.label_map)
        ap_data_frame[METHOD] = method_name
        data_frames.append(ap_data_frame)
    return pandas.concat(data_frames)


def _create_pr_data_frame(
        evaluation_runs: List[EvaluationRun], methods: List[str]) -> DataFrame:
    data_frames = []
    for evaluation_run, name in zip(evaluation_runs, methods):
        spotting_evaluation = evaluation_run.evaluation.spotting_evaluation
        pr_data_frame = spotting_evaluation.pr_data_frame
        pr_data_frame = _add_class_name_mapping(
            pr_data_frame, spotting_evaluation.label_map)
        pr_data_frame[METHOD] = name
        data_frames.append(pr_data_frame)
    concatenated = pandas.concat(data_frames)
    # Remove the (0, 0) points so the plots look nicer.
    drop_selection = (
            (concatenated[SpottingEvaluation.PRECISION] == 0) &
            (concatenated[SpottingEvaluation.RECALL] == 0)
    )
    concatenated.drop(concatenated[drop_selection].index, inplace=True)
    # Sort the precision and recall values according to their respective
    # thresholds so that it is easier to plot them later.
    concatenated.sort_values(SpottingEvaluation.THRESHOLD, inplace=True)
    return concatenated


def _create_pr_per_tolerance_data_frame(
        evaluation_runs: List[EvaluationRun], methods: List[str],
        selected_classes: List[bool]) -> DataFrame:
    data_frames = []
    for evaluation_run, name in zip(evaluation_runs, methods):
        confusion_data_frame = \
            evaluation_run.evaluation.spotting_evaluation.confusion_data_frame
        method_pr_per_tolerance_data_frame = \
            _create_method_pr_per_tolerance_data_frame(
                confusion_data_frame, selected_classes)
        method_pr_per_tolerance_data_frame[METHOD] = name
        data_frames.append(method_pr_per_tolerance_data_frame)
    pr_per_tolerance_data_frame = pandas.concat(data_frames)
    # Sort the precision and recall values according to their respective
    # thresholds so that it is easier to plot them later.
    pr_per_tolerance_data_frame.sort_values(
        SpottingEvaluation.THRESHOLD, inplace=True)
    return pr_per_tolerance_data_frame


def _create_method_pr_per_tolerance_data_frame(
        confusion_data_frame: DataFrame,
        selected_classes: List[bool]) -> DataFrame:
    # Restrict classes according to selected_classes.
    drop_selection = confusion_data_frame[SpottingEvaluation.CLASS_INDEX].map(
        lambda index: not selected_classes[index])
    confusion_data_frame = confusion_data_frame.drop(
        confusion_data_frame[drop_selection].index, inplace=False)
    # Sum counts across all classes, grouping by tolerance and threshold.
    aggregate_data_frame = confusion_data_frame.groupby(
        [SpottingEvaluation.TOLERANCE, SpottingEvaluation.THRESHOLD]).sum()
    # Drop rows with zero predicted positive, since the precision is undefined
    # in those cases.
    drop_selection = (
            aggregate_data_frame[SpottingEvaluation.PREDICTED_POSITIVE] == 0)
    aggregate_data_frame.drop(
        aggregate_data_frame[drop_selection].index, inplace=True)
    # Add columns with precision and recall
    aggregate_data_frame[SpottingEvaluation.PRECISION] = (
        aggregate_data_frame[SpottingEvaluation.TRUE_POSITIVE] /
        aggregate_data_frame[SpottingEvaluation.PREDICTED_POSITIVE])
    aggregate_data_frame[SpottingEvaluation.RECALL] = (
        aggregate_data_frame[SpottingEvaluation.TRUE_POSITIVE] /
        aggregate_data_frame[SpottingEvaluation.CONDITION_POSITIVE])
    # Transforms the tolerance and threshold into regular columns, as opposed
    # to indexes from the groupby result.
    return aggregate_data_frame.reset_index()


def _add_class_name_mapping(
        data_frame: DataFrame, label_map: LabelMap) -> DataFrame:
    data_frame[SpottingEvaluation.CLASS_NAME] = data_frame[
        SpottingEvaluation.CLASS_INDEX].map(label_map.int_to_label)
    return data_frame


def _create_average_map_graph(
        evaluation_runs: List[EvaluationRun], methods: List[str]) -> Figure:
    rows = []
    for evaluation_run, method in zip(evaluation_runs, methods):
        average_map_dict = evaluation_run.evaluation.spotting_evaluation\
            .average_map_dict
        rows.extend([
            {AVERAGE_MAP: a_map, METHOD: method, TOLERANCES: tolerances_name}
            for tolerances_name, a_map in average_map_dict.items()])
    data_frame = DataFrame(rows)
    tolerances_names = sorted(
        evaluation_runs[0].evaluation.spotting_evaluation.average_map_dict
        .keys())
    category_orders = {TOLERANCES: tolerances_names}
    fig = px.bar(
        data_frame, x=TOLERANCES, y=AVERAGE_MAP, color=METHOD,
        barmode="group", category_orders=category_orders)
    fig.update_layout(
        title_text=AVERAGE_MAP_GRAPH_TITLE, height=AVERAGE_MAP_GRAPH_HEIGHT,
        width=AVERAGE_MAP_GRAPH_WIDTH)
    return fig


def _create_class_graph(
        evaluation_runs: List[EvaluationRun], methods: List[str],
        ordered_class_names: List[str]) -> Figure:
    rows = []
    for evaluation_run, method in zip(evaluation_runs, methods):
        spotting_evaluation = evaluation_run.evaluation.spotting_evaluation
        class_average_ap_dict = spotting_evaluation.class_average_ap_dict
        label_map = spotting_evaluation.label_map
        rows.extend(_create_method_class_average_ap_rows(
            class_average_ap_dict, method, label_map))
    average_ap_data_frame = DataFrame(rows)
    tolerances_names = _extract_tolerances_names(average_ap_data_frame, methods)
    active_tolerances_index = 0
    active_tolerances = tolerances_names[active_tolerances_index]
    active_tolerances_average_ap_data_frame = average_ap_data_frame[
        average_ap_data_frame[TOLERANCES] == active_tolerances]
    active_tolerances_average_ap_values = [
        _create_method_class_average_ap_values(
            active_tolerances_average_ap_data_frame, method,
            ordered_class_names)
        for method in methods
    ]
    fig = px.bar(
        x=ordered_class_names, y=active_tolerances_average_ap_values,
        barmode="group")
    _update_trace_names(fig, methods)
    buttons = []
    # button with one option for each dataframe
    for tolerances_name in tolerances_names:
        tolerances_average_ap_data_frame = average_ap_data_frame[
            average_ap_data_frame[TOLERANCES] == tolerances_name]
        tolerance_average_ap_values = [
            _create_method_class_average_ap_values(
                tolerances_average_ap_data_frame, method, ordered_class_names)
            for method in methods
        ]
        args = [{"y": tolerance_average_ap_values}]
        new_button = dict(
            method="restyle", visible=True, args=args,
            label=f"{TOLERANCES}: {tolerances_name}")
        buttons.append(new_button)
    update_menus = [
        dict(buttons=buttons, direction="down", showactive=True,
             active=active_tolerances_index)]
    fig.update_layout(
        title_text=CLASS_GRAPH_TITLE, xaxis_title=CLASS_AXIS_TITLE,
        yaxis_title=AVERAGE_AP_AXIS_TITLE, height=CLASS_GRAPH_HEIGHT,
        width=CLASS_GRAPH_WIDTH, legend_title_text=METHOD,
        updatemenus=update_menus)
    return fig


def _create_ap_per_tolerance_graph(
        ap_data_frame: DataFrame, methods: List[str],
        ordered_class_names: List[str]) -> Figure:
    tolerances = _extract_tolerances(ap_data_frame, methods)
    active_tolerance_index = len(tolerances) - 1
    active_tolerance = tolerances[active_tolerance_index]
    active_tolerance_ap_data_frame = ap_data_frame[
        ap_data_frame[SpottingEvaluation.TOLERANCE] == active_tolerance]
    active_tolerance_ap_values = [
        _create_method_tolerance_ap_values(
            active_tolerance_ap_data_frame, method, ordered_class_names)
        for method in methods
    ]
    fig = px.bar(
        x=ordered_class_names, y=active_tolerance_ap_values, barmode="group")
    _update_trace_names(fig, methods)
    buttons = []
    # button with one option for each dataframe
    for tolerance in tolerances:
        tolerance_ap_data_frame = ap_data_frame[
            ap_data_frame[SpottingEvaluation.TOLERANCE] == tolerance]
        tolerance_ap_values = [
            _create_method_tolerance_ap_values(
                tolerance_ap_data_frame, method, ordered_class_names)
            for method in methods
        ]
        args = [{"y": tolerance_ap_values}]
        new_button = dict(
            method="restyle", visible=True, args=args,
            label=f"{SpottingEvaluation.TOLERANCE}: {tolerance}")
        buttons.append(new_button)
    update_menus = [
        dict(buttons=buttons, direction="down", showactive=True,
             active=active_tolerance_index)]
    fig.update_layout(
        title_text=AP_PER_TOLERANCE_GRAPH_TITLE,
        xaxis_title=CLASS_AXIS_TITLE, yaxis_title=AP_AXIS_TITLE,
        height=AP_PER_CLASS_TOLERANCE_GRAPH_HEIGHT,
        width=AP_PER_TOLERANCE_GRAPH_WIDTH, legend_title_text=METHOD,
        updatemenus=update_menus)
    return fig


def _extract_tolerances_names(
        data_frame: DataFrame, methods: List[str]) -> List[str]:
    tolerances_names_set = set(data_frame[TOLERANCES])
    for method in methods:
        method_tolerances_names_set = set(
            data_frame[data_frame[METHOD] == method][TOLERANCES])
        tolerances_names_set = tolerances_names_set.intersection(
            method_tolerances_names_set)
    return sorted(tolerances_names_set)


def _extract_tolerances(
        data_frame: DataFrame, methods: List[str]) -> List[float]:
    tolerance_set = set(data_frame[SpottingEvaluation.TOLERANCE])
    for method in methods:
        method_tolerance_set = set(
            data_frame[data_frame[METHOD] == method][
                SpottingEvaluation.TOLERANCE])
        tolerance_set = tolerance_set.intersection(method_tolerance_set)
    tolerances = sorted(tolerance_set)
    return tolerances


def _create_tolerance_graph(
        evaluation_runs: List[EvaluationRun], methods: List[str],
        tolerance_limit: Optional[float], height: int, width: int,
        show_legend: bool, title: str, pdf: bool) -> Figure:
    fig = go.Figure()
    for i, evaluation_run in enumerate(evaluation_runs):
        name = methods[i]
        symbol = METHOD_SYMBOL_SEQUENCE[i]
        tolerance_average_map = (
            evaluation_run.evaluation.spotting_evaluation.tolerance_map)
        sorted_tolerances = sorted(tolerance_average_map.keys())
        if tolerance_limit:
            sorted_tolerances = [
                tolerance for tolerance in sorted_tolerances
                if tolerance <= tolerance_limit]
        sorted_average_map = [
            tolerance_average_map[tolerance] for tolerance in sorted_tolerances]
        marker_dict = dict(symbol=symbol)
        fig.add_scatter(
            x=sorted_tolerances, y=sorted_average_map, mode="markers+lines",
            name=name, marker=marker_dict)
    if pdf:
        template = "none"
        legend_dict = dict(
            yanchor="bottom", y=0.07, xanchor="right", x=0.995, borderwidth=1)
    else:
        template = "plotly"  # the default
        legend_dict = dict()
        fig.update_xaxes(tickvals=TOLERANCE_TICKS)
    fig.update_layout(
        title_text=title, xaxis_title=MATCHING_TOLERANCE_AXIS_TITLE,
        yaxis_title=MAP_AXIS_TITLE, height=height, width=width,
        template=template, legend=legend_dict, showlegend=show_legend)
    return fig


def _create_ap_per_class_graph(
        ap_data_frame: DataFrame, methods: List[str],
        ordered_class_names: List[str]) -> Figure:
    ap_data_frame.sort_values(SpottingEvaluation.TOLERANCE, inplace=True)
    active_class_index = 0
    active_class_name = ordered_class_names[active_class_index]
    active_class_data_frame = ap_data_frame[
        ap_data_frame[SpottingEvaluation.CLASS_NAME] == active_class_name]
    # To simplify things, choose only the tolerances that are common to all
    # methods.
    tolerances = _extract_tolerances(active_class_data_frame, methods)
    # Now, we need to filter the data frame to only keep the rows for the
    # relevant tolerances.
    active_class_data_frame = active_class_data_frame[
        active_class_data_frame[SpottingEvaluation.TOLERANCE].isin(tolerances)]
    # Finally, we get the relevant ap values from the data frame.
    active_class_ap_values = _all_methods_ap_values(
        active_class_data_frame, methods)
    fig = px.line(y=active_class_ap_values, x=tolerances, labels=methods)
    _update_trace_names(fig, methods)
    buttons = []
    for class_name in ordered_class_names:
        class_ap_values = _class_ap_values(ap_data_frame, class_name, methods)
        args = [{"y": class_ap_values, "labels": methods}]
        new_button = dict(
            method="restyle", label=class_name, visible=True, args=args)
        buttons.append(new_button)
    fig.update_traces(mode="markers+lines")
    fig.update_xaxes(tickvals=TOLERANCE_TICKS)
    update_menus = [dict(
        buttons=buttons, direction="down", showactive=True,
        active=active_class_index)]
    fig.update_layout(
        title_text=AP_PER_CLASS_GRAPH_TITLE,
        xaxis_title=MATCHING_TOLERANCE_AXIS_TITLE, yaxis_title=AP_AXIS_TITLE,
        height=AP_PER_CLASS_CLASS_GRAPH_HEIGHT, width=AP_PER_CLASS_GRAPH_WIDTH,
        legend_title_text="", updatemenus=update_menus)
    return fig


def _class_ap_values(ap_data_frame, class_name, methods):
    class_data_frame = ap_data_frame[
        ap_data_frame[SpottingEvaluation.CLASS_NAME] == class_name]
    return _all_methods_ap_values(class_data_frame, methods)


def _all_methods_ap_values(ap_data_frame, methods):
    return [
        ap_data_frame[ap_data_frame[METHOD] == method][
            SpottingEvaluation.AVERAGE_PRECISION].values
        for method in methods]


def _create_method_class_average_ap_rows(
        class_average_ap_dict: Dict[str, List[float]], method: str,
        label_map: LabelMap) -> List[Dict]:
    return [
        {
            AVERAGE_AP: average_ap,
            TOLERANCES: tolerances_name,
            SpottingEvaluation.CLASS_NAME: label_map.int_to_label[class_index],
            METHOD: method
        }
        for tolerances_name, class_average_ap in class_average_ap_dict.items()
        for class_index, average_ap in enumerate(class_average_ap)
    ]


def _create_method_class_average_ap_values(
        tolerances_ap_data_frame: DataFrame, method: str,
        ordered_class_names: List[str]):
    method_tolerances_ap_data_frame = tolerances_ap_data_frame[
        tolerances_ap_data_frame[METHOD] == method]
    return [
        method_tolerances_ap_data_frame[
            method_tolerances_ap_data_frame[SpottingEvaluation.CLASS_NAME] ==
            class_name][AVERAGE_AP].values[0]
        for class_name in ordered_class_names
    ]


def _create_method_tolerance_ap_values(
        tolerance_ap_data_frame, method, ordered_class_names: List[str]):
    method_tolerance_ap_data_frame = tolerance_ap_data_frame[
        tolerance_ap_data_frame[METHOD] == method]
    return [
        method_tolerance_ap_data_frame[
            method_tolerance_ap_data_frame[SpottingEvaluation.CLASS_NAME] ==
            class_name][SpottingEvaluation.AVERAGE_PRECISION].values[0]
        for class_name in ordered_class_names
    ]


def _create_pr_per_class_and_tolerance_graph(
        pr_data_frame: DataFrame, methods: List[str],
        ordered_class_names: List[str]) -> Figure:
    tolerances = _extract_tolerances(pr_data_frame, methods)
    active_tolerance = tolerances[0]
    active_class_name = ordered_class_names[0]
    active_pr_data_frame = pr_data_frame[
        (pr_data_frame[SpottingEvaluation.CLASS_NAME] == active_class_name) &
        (pr_data_frame[SpottingEvaluation.TOLERANCE] == active_tolerance)]
    # For some reason, I'm not able to directly provide x, y and text as data
    # here, so I have to use provide the DataFrame instead into px.line().
    fig = px.line(
        active_pr_data_frame, y=SpottingEvaluation.PRECISION,
        x=SpottingEvaluation.RECALL, text=SpottingEvaluation.THRESHOLD,
        color=METHOD, category_orders={METHOD: methods})
    buttons = []
    for tolerance in tolerances:
        pr_tolerance_data_frame = pr_data_frame[
            pr_data_frame[SpottingEvaluation.TOLERANCE] == tolerance]
        for class_name in ordered_class_names:
            class_precisions, class_recalls, class_thresholds = \
                _class_precision_recall(
                    pr_tolerance_data_frame, class_name, methods)
            args = [{
                "y": class_precisions, "x": class_recalls,
                "text": class_thresholds, "labels": methods}]
            button_label = f"Tolerance: {tolerance}, Class: {class_name}"
            new_button = dict(
                method="restyle", label=button_label, visible=True, args=args)
            buttons.append(new_button)
    fig.update_traces(mode="markers+lines")
    update_menus = [dict(
        buttons=buttons, direction="down", showactive=True, active=0)]
    fig.update_layout(
        title_text=PR_GRAPH_TITLE, xaxis_title=RECALL_AXIS_TITLE,
        yaxis_title=PRECISION_AXIS_TITLE, height=PR_GRAPH_HEIGHT,
        width=PR_GRAPH_WIDTH, legend_title_text="", updatemenus=update_menus,
        xaxis_range=PR_RECALL_RANGE, yaxis_range=PR_PRECISION_RANGE)
    return fig


def _create_pr_per_tolerance_graph(
        pr_data_frame: DataFrame, methods: List[str]) -> Figure:
    tolerances = _extract_tolerances(pr_data_frame, methods)
    active_tolerance = tolerances[0]
    active_pr_data_frame = pr_data_frame[
        pr_data_frame[SpottingEvaluation.TOLERANCE] == active_tolerance]
    # For some reason, I'm not able to directly provide x, y and text as data
    # here, so I have to use provide the DataFrame instead into px.line().
    fig = px.line(
        active_pr_data_frame, y=SpottingEvaluation.PRECISION,
        x=SpottingEvaluation.RECALL, text=SpottingEvaluation.THRESHOLD,
        color=METHOD, category_orders={METHOD: methods})
    buttons = []
    for tolerance in tolerances:
        tolerance_precisions, tolerance_recalls, tolerance_thresholds = \
            _tolerance_precision_recall(pr_data_frame, tolerance, methods)
        args = [{
            "y": tolerance_precisions, "x": tolerance_recalls,
            "text": tolerance_thresholds, "labels": methods}]
        button_label = f"Tolerance: {tolerance}"
        new_button = dict(
            method="restyle", label=button_label, visible=True, args=args)
        buttons.append(new_button)
    fig.update_traces(mode="markers+lines")
    update_menus = [dict(
        buttons=buttons, direction="down", showactive=True, active=0)]
    fig.update_layout(
        title_text=PR_PER_TOLERANCE_GRAPH_TITLE, xaxis_title=RECALL_AXIS_TITLE,
        yaxis_title=PRECISION_AXIS_TITLE, height=PR_PER_TOLERANCE_GRAPH_HEIGHT,
        width=PR_PER_TOLERANCE_GRAPH_WIDTH, legend_title_text="",
        updatemenus=update_menus, xaxis_range=PR_RECALL_RANGE,
        yaxis_range=PR_PRECISION_RANGE)
    return fig


def _tolerance_precision_recall(
        pr_data_frame: DataFrame, tolerance: float, methods: List[str]):
    pr_tolerance_data_frame = pr_data_frame[
        pr_data_frame[SpottingEvaluation.TOLERANCE] == tolerance]
    return _all_methods_precision_recall(pr_tolerance_data_frame, methods)


def _class_precision_recall(pr_data_frame, class_name, methods):
    class_data_frame = pr_data_frame[
        pr_data_frame[SpottingEvaluation.CLASS_NAME] == class_name]
    return _all_methods_precision_recall(class_data_frame, methods)


def _all_methods_precision_recall(pr_data_frame, methods):
    all_methods_precisions = []
    all_methods_recalls = []
    all_methods_thresholds = []
    for method in methods:
        method_pr_data_frame = pr_data_frame[pr_data_frame[METHOD] == method]
        all_methods_precisions.append(
            method_pr_data_frame[SpottingEvaluation.PRECISION].values)
        all_methods_recalls.append(
            method_pr_data_frame[SpottingEvaluation.RECALL].values)
        all_methods_thresholds.append(
            method_pr_data_frame[SpottingEvaluation.THRESHOLD].values)
    return all_methods_precisions, all_methods_recalls, all_methods_thresholds


def _update_trace_names(fig: Figure, methods: List[str]) -> None:
    legend_names = cycle(methods)

    def _update_trace(t):
        legend_name = next(legend_names)
        t.update(
            name=legend_name, legendgroup=legend_name,
            hovertemplate=t.hovertemplate.replace(t.name, legend_name)
        )

    fig.for_each_trace(_update_trace)


if __name__ == "__main__":
    main()
