"""Mihir Patankar [mpatankar06@gmail.com]"""
import sys
import tkinter
from tkinter.ttk import Button, Checkbutton, Entry, Label, Style

import yaml
from yaml.scanner import ScannerError

from data_structures import ConfigField

CONFIG_FILE_PATH = "./config.yaml"


class SearchConfigGUI:
    """Opens a simple GUI where the config can be edited and saved."""

    def __init__(self, config):
        self.config = config
        self.config_entries = []
        self.cleanup_function = None

    def __enter__(self):
        window = tkinter.Tk()
        window.geometry("350x400")
        window.title("Edit program configuration:")
        window.configure(menu=tkinter.Menu(window))
        window.resizable(width=False, height=False)

        # The default behavior, window.destroy(), confuses the context manager.
        window.protocol("WM_DELETE_WINDOW", sys.exit)

        window_style = Style()
        window_style.configure("TButton", foreground="black")
        window_style.configure("TEntry", foreground="black")
        window_style.configure("TLabel", padding=5, foreground="black")

        self.config_entries = [
            ConfigField(Label(window, text="Object Name"), Entry(window), "", str),
            ConfigField(Label(window, text="Search Radius (arcmin)"), Entry(window), "2", float),
            ConfigField(Label(window, text="Significance Threshold"), Entry(window), "50", float),
            ConfigField(Label(window, text="Minimum Counts"), Entry(window), "100", int),
            ConfigField(Label(window, text="Binsize"), Entry(window), "500", float),
            ConfigField(Label(window, text="Data Directory"), Entry(window), "./data", str),
            ConfigField(Label(window, text="Output Directory"), Entry(window), "./output", str),
            ConfigField(Label(window, text="Log Directory"), Entry(window), "./logs", str),
            ConfigField(Label(window, text="Enable Output"), Checkbutton(window), True, bool),
        ]

        self.populate_form_fields()

        btn_row = len(self.config_entries) + 1
        Label(text="").grid(row=btn_row - 1, column=0)
        Button(text="Save & Close", command=window.quit).grid(row=btn_row, column=0)
        Button(text="Reset", command=self.reset_form).grid(row=btn_row, column=1)
        Label(window, text="Warning: paths will not be validated.").grid(
            row=btn_row + 1, column=0, columnspan=2
        )
        print("Waiting for config window...")
        window.mainloop()
        self.cleanup_function = lambda: [window.withdraw(), window.destroy()]
        return self

    @staticmethod
    def change_input_value(input_field, value):
        """Change the text within a field widget."""
        if isinstance(input_field, Checkbutton):
            input_field.state(["!alternate"])
            input_field.state(["selected" if value is True else "!selected"])
        else:
            input_field.delete(0, "end")
            input_field.insert(0, value)

    def populate_form_fields(self):
        """Puts all labels and input fields onto the form, and loads the saved config data into the
        fields where possible, default values when neccesary."""
        for row, config_entry in enumerate(self.config_entries):
            input_label, input_field = (config_entry.label, config_entry.field)
            input_label.grid(row=row, column=0)
            input_field.grid(row=row, column=1)
            label_text = input_label["text"]
            self.change_input_value(
                input_field,
                self.config[label_text]
                if label_text in self.config
                else config_entry.default_value,
            )

    def reset_form(self):
        """Resets all input fields in the form to their default values."""
        for config_entry in self.config_entries:
            self.change_input_value(config_entry.field, config_entry.default_value)

    def finalize_config(self):
        """Get new configuration as a dictionary. Loosely validates fields by performing conversion
        with their corresponding type."""

        def get_value(field):
            is_checkbox = isinstance(field, Checkbutton)
            return field.instate(["selected"]) if is_checkbox else field.get()

        try:
            validated_config = {
                config_entry.label["text"]: config_entry.entry_type(get_value(config_entry.field))
                for config_entry in self.config_entries
            }
        except ValueError as error:
            print(f"Invalid input - {error}.\nConfig will not be overwritten.")
            sys.exit(1)
        return validated_config

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_value and exception_type != SystemExit:
            exception_info = (exception_type, exception_value, traceback)
            print("GUI context manager threw exception:", exception_info)
        self.cleanup_function()


def get_config(use_gui=False):
    """Writes, reads, and returns the config file."""
    with open(file=CONFIG_FILE_PATH, mode="a", encoding="utf-8"):
        # Creates file if it does not exist.
        pass

    with open(file=CONFIG_FILE_PATH, mode="r+", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file) or {}
        except ScannerError:
            print("Config file corrupted.")
            config = {}
        if not use_gui:
            return config
        with SearchConfigGUI(config) as config_gui:
            file.seek(0)
            finalized_config = config_gui.finalize_config()
            file.write(yaml.safe_dump(finalized_config))
            file.truncate()
    return finalized_config
