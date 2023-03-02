# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import csv
from pathlib import Path
from typing import Dict, List

FIELD_NAME = "name"
FIELD_ID = "id"
FIELD_DISPLAY_ORDER = "order"


class LabelMap:

    def __init__(
            self, int_to_label: Dict[int, str],
            display_order: List[int]) -> None:
        self.int_to_label = int_to_label
        self.label_to_int = {
            label: integer for integer, label in int_to_label.items()}
        # Map the ordered position of a label to its name.
        display_ordered_labels_dict = {
            ordered_position: int_to_label[label_int]
            for label_int, ordered_position in enumerate(display_order)}
        # Convert from dictionary to list.
        self.display_ordered_labels = [
            display_ordered_labels_dict[ordered_position]
            for ordered_position in sorted(display_order)]

    def num_classes(self) -> int:
        return len(self.int_to_label)

    def write(self, label_map_file_path: Path) -> None:
        with label_map_file_path.open("w") as csv_file:
            writer = csv.DictWriter(
                csv_file, [FIELD_ID, FIELD_NAME, FIELD_DISPLAY_ORDER])
            writer.writeheader()
            for order, label in enumerate(self.display_ordered_labels):
                row_dict = {
                    FIELD_NAME: label, FIELD_ID: self.label_to_int[label],
                    FIELD_DISPLAY_ORDER: order}
                writer.writerow(row_dict)

    @staticmethod
    def read_label_map(label_map_file_path: Path) -> "LabelMap":
        with label_map_file_path.open("r") as csv_file:
            reader = csv.DictReader(csv_file)
            int_to_label = {
                int(row[FIELD_ID]): row[FIELD_NAME] for row in reader}
        # It's hard to reset the reader, so I'll just open the file all over
        # again here.
        with label_map_file_path.open("r") as csv_file:
            reader = csv.DictReader(csv_file)
            display_order = [int(row[FIELD_DISPLAY_ORDER]) for row in reader]
        return LabelMap(int_to_label, display_order)
