"""Mihir Patankar [mpatankar06@gmail.com]"""
import subprocess
import sys
from pathlib import Path

import yaml


def write_object_name_to_config(object_name):
    """Writes the object name to the config file, for use with each new object."""
    config_file = "./config.yaml"

    with open(file=config_file, mode="r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["Object Name"] = object_name

    with open(file=config_file, mode="w", encoding="utf-8") as file:
        yaml.dump(config, file)


def run_main_program():
    """Runs the main program."""
    program_path = "Lightcurves/main.py"
    subprocess.call(
        [
            sys.executable,
            program_path,
            "--no-gui",
        ]
    )


def main():
    """Main method."""
    progress_file = Path("Lightcurves/batch_run/batch_progress.txt")
    object_names = []
    with open(file="Lightcurves/batch_run/objects_list.txt", mode="r", encoding="utf-8") as file:
        for line in file:
            object_names.append(line.strip())
    for index, object_name in enumerate(object_names):
        try:
            write_object_name_to_config(object_name)
            run_main_program()
            with open(file=progress_file, mode="w", encoding="utf-8") as file:
                file.write(f"{index + 1}/{len(object_names)}")
        except KeyboardInterrupt:
            print("Batch run interrupted.")
            progress_file.unlink(missing_ok=True)
            sys.exit(0)


if __name__ == "__main__":
    main()
