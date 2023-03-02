#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import argparse
import csv
from pathlib import Path
from typing import Dict, Set, List

from spivak.data.dataset import Task
from spivak.data.dataset_splits import SPLIT_KEY_VALIDATION, SPLIT_KEY_TEST, \
    SPLIT_KEY_TRAIN
from spivak.data.soccernet_reader import GamePathsReader, \
    GAMES_CSV_COLUMN_GAME, GAMES_CSV_COLUMN_SPLIT_KEY

OUT_CSV_PATH = Path("SoccerNetSpottingAndCameraChangesLarge.csv")
ARG_SPLITS_DIR = "splits_dir"
RELEVANT_SPLITS = [SPLIT_KEY_TRAIN, SPLIT_KEY_VALIDATION, SPLIT_KEY_TEST]
OUT_FIELDS = [
    GAMES_CSV_COLUMN_GAME, GAMES_CSV_COLUMN_SPLIT_KEY, Task.SPOTTING.name,
    Task.SEGMENTATION.name]


def main() -> None:
    args = _get_command_line_arguments()
    splits_dir = Path(args[ARG_SPLITS_DIR])
    spotting_game_paths = _read_spotting_game_paths_dict(splits_dir)
    segmentation_game_paths_set = _read_segmentation_game_paths_set(splits_dir)
    out_rows = _prepare_out_rows(
        spotting_game_paths, segmentation_game_paths_set)
    _write_rows(out_rows)
    _print_summary(out_rows)


def _read_spotting_game_paths_dict(splits_dir: Path) -> Dict[str, List[Path]]:
    spotting_game_paths = dict()
    for split_key in RELEVANT_SPLITS:
        spotting_game_paths[split_key] = GamePathsReader.read_game_list_v2(
            splits_dir, split_key)
    return spotting_game_paths


def _read_segmentation_game_paths_set(splits_dir: Path) -> Set[Path]:
    segmentation_game_paths = set()
    for split_key in RELEVANT_SPLITS:
        split_game_paths = \
            GamePathsReader.read_game_list_v2_camera_segmentation(
                splits_dir, split_key)
        segmentation_game_paths.update(split_game_paths)
    return segmentation_game_paths


def _prepare_out_rows(
        spotting_game_paths: Dict[str, List[Path]],
        segmentation_game_paths_set: Set[Path]) -> List[Dict]:
    out_rows = []
    for split_key in RELEVANT_SPLITS:
        split_games = spotting_game_paths[split_key]
        for game_path in split_games:
            use_segmentation = (game_path in segmentation_game_paths_set)
            row = {
                GAMES_CSV_COLUMN_GAME: game_path,
                GAMES_CSV_COLUMN_SPLIT_KEY: split_key,
                Task.SEGMENTATION.name: int(use_segmentation),
                Task.SPOTTING.name: 1}
            out_rows.append(row)
    return out_rows


def _write_rows(out_rows: List[Dict]) -> None:
    print(f"Writing output to {OUT_CSV_PATH}")
    with OUT_CSV_PATH.open("w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)


def _print_summary(out_rows: List[Dict]) -> None:
    print("Summary of game counts per split:")
    for split_key in RELEVANT_SPLITS:
        split_rows = [
            row for row in out_rows
            if row[GAMES_CSV_COLUMN_SPLIT_KEY] == split_key]
        n_games = len(split_rows)
        n_segmentation = sum(row[Task.SEGMENTATION.name] for row in split_rows)
        n_spotting = sum(row[Task.SPOTTING.name] for row in split_rows)
        print(f"split: {split_key}, n_games: {n_games}, n_spotting: "
              f"{n_spotting}, n_segmentation: {n_segmentation}")


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + ARG_SPLITS_DIR, help="Directory containing the splits",
        required=True)
    args_dict = vars(parser.parse_args())
    return args_dict


if __name__ == "__main__":
    main()
