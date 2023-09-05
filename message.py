"""Mihir Patankar [mpatankar06@gmail.com]"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """Holds an optional uuid field, this is used for when you want to track and update a single
    message in the buffer rather than constantly appending."""

    content: str
    uuid: str = None
