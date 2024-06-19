"""Mihir Patankar [mpatankar06@gmail.com]"""
import argparse
import atexit
import subprocess
import time

import search
import search_config
import server


def main():
    """Entry point for the program."""
    subprocess.run(["clear"], check=False)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--no-gui", action="store_false")
    arguments = argument_parser.parse_args()
    source_manager = search.SourceManager(search_config.get_config(use_gui=arguments.no_gui))
    source_manager.search_csc()
    start_time = time.perf_counter()
    atexit.register(search.print_log_location)
    output_html_file = source_manager.download_and_process()
    print(f"\nAll sources finished in {time.perf_counter() - start_time:.3f}s")
    if source_manager.config["Auto Start Server"]:
        server.start(output_html_file)


if __name__ == "__main__":
    print("Starting program.")
    main()
