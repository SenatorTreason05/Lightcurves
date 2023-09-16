"""Mihir Patankar [mpatankar06@gmail.com]"""
from io import StringIO
from tkinter.ttk import Checkbutton, Entry, Label
from typing import Callable, NamedTuple
from uuid import UUID


class ConfigField(NamedTuple):
    """Holds immutable data for a GUI configuration entry."""

    label: Label
    field: Entry | Checkbutton
    default_value: str | bool
    entry_type: Callable


class Message(NamedTuple):
    """Holds an optional uuid field, this is used for when you want to track and update a single
    message in the buffer rather than constantly appending."""

    content: str
    uuid: UUID = None


class ObservationHeaderInfo(NamedTuple):
    """Holds details derrived from the observation header."""

    instrument: str
    observation_id: int
    region_id: int
    start_time: str
    end_time: str


class ObservationData(NamedTuple):
    """Holds data derrived from the observation event list."""

    average_count_rate: float
    total_counts: int
    total_exposure_time: float
    raw_start_time: int


class LightcurveParseResults(NamedTuple):
    """Holds observation details and data along with the in-memory svg string representation for
    its plot image."""

    observation_header_info: ObservationHeaderInfo
    observation_data: ObservationData
    svg_data: StringIO

class ExportableObservationData(NamedTuple):
    """Combines all observation data into the form that will be sent to the templating engine."""

    columns: dict
    image_path: str
    