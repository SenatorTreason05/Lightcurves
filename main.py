"""Mihir Patankar [mpatankar06@gmail.com]"""
import os
import sys

import search_config
from search import SourceManager


def main():
    """Entry point for the program."""
    os.system("clear")
    source_manager = SourceManager(search_config.get_config())
    source_manager.search_csc()
    if input("Proceed? (y/n) ") != "y":
        sys.exit(0)
    source_manager.download_and_process()


if __name__ == "__main__":
    print("Starting program.")
    main()
