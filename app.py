import os
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from plots import Plot

DATA_DIRECTORY = '/Users/wayan/asiaa/tkinter_plotting/test_data'
PADX = 20
PADY = 20
F_WIDTH = 100
F_HEIGHT = 200

class DirectoryFrame(ttk.LabelFrame):
    '''
    Define DirectoryFrame for navigating and selecting directory items
    '''

    def __init__(self, parent, console_frame):
        super().__init__(parent, text='Directory')
        self.selected_files = []
        self.console_frame = console_frame
        self.directory_listbox = tk.Listbox(self, selectmode=tk.SINGLE,
                                       width=50, height=15)
        self.directory_listbox.grid(column=0, row=0)
        self.directory_listbox.bind("<<ListboxSelect>>", self.on_select_directory)

        self.selected_directory = None
        self.data_file = [None]
        self.data_path = '/Users/wayan/asiaa/tkinter_plotting/test_data'

        self.select_button = ttk.Button(self, text="Select Directory",
                                        command=lambda: self.select_directory())
        self.select_button.grid(column=0, row=1)
        # Populate listbox with files from the specified directory
        self.populate_file_list(self.data_path)

    def select_directory(self):
        target = os.path.join(self.data_path, self.selected_directory)
        file = filedialog.askopenfilename(initialdir=target)

        if file:
            self.console_frame.log(f'Selected file: {file}')
            self.data_file[0] = file

    def on_select_directory(self, event):
        selected_index = self.directory_listbox.curselection()
        if selected_index:
            self.selected_directory = self.directory_listbox.get(selected_index)
            self.console_frame.log(f"Selected directory: {self.selected_directory}")

    def populate_file_list(self, directory):
        self.console_frame.log("Setting directory")
        if os.path.isdir(directory):
            data_folders = os.listdir(directory)
            for folder in data_folders:
                if os.path.isdir(os.path.join(directory, folder)):
                    self.directory_listbox.insert(tk.END, folder)

    def select_files(self):
        files = filedialog.askopenfilenames()
        if files:
            self.selected_files = files
            self.update_file_list()

    def update_file_list(self):
        self.directory_listbox.delete(0, tk.END)
        for file in self.selected_files:
            self.directory_listbox.insert(tk.END, file)

    def get_datafile(self):
        return self.data_file

class ChannelFrame(ttk.LabelFrame):
    '''
    Frame for selecting specific channels
    '''

    def __init__(self, parent, console_frame):
        super().__init__(parent, text='Select Channel')
        self.console_frame = console_frame
        self.checkbox_vars = []

        self.setup()
        self.testButton = ttk.Button(self, text='Save Selection',
                                     command=lambda: self.get_channels())
        self.testButton.grid(row=1, column=0)

    def setup(self):
        total_channels = 16
        row = 0
        col = 0
        frame = ttk.Frame(self)
        frame.grid(row=0, column=0)

        for i in range(total_channels):
            var = tk.BooleanVar()
            self.checkbox_vars.append(var)
            tk.Checkbutton(frame, text=i,
                           variable=var).grid(
                                   column=col, row=row, sticky=tk.W)

            col += 1
            if i == 7:
                row += 1
                col = 0

    def get_channels(self):
        channels = []
        for i, var in enumerate(self.checkbox_vars):
            if var.get():
                channels.append(int(i))

        channel_string = ','.join(str(c) for c in channels)
        print(channel_string)
        self.console_frame.log(f'Selected Channels: {channel_string}')

        return channels

class PlotFrame(ttk.LabelFrame):
    '''
    Frame for selecting and generating plots
    '''

    def __init__(self, parent, console_frame):
        super().__init__(parent, text="Select Plot")
        self.plot_types = ["correlation_coefficent",
                           "correlation_magnitude", "correlation_phase",
                           "spectrum", "waterfall", "rms"]
        self.console_frame = console_frame
        self.selected_plot = None

        self.generate_plot_options()

    def generate_plot_options(self):
        row = 0
        col = 0
        selected_option = tk.StringVar()
        for plot in self.plot_types:
            # Create plot ui
            tk.Radiobutton(self, text=f"{plot}",
                           variable=selected_option,
                           command=lambda p=plot: self.set_plot(p),
                           value=plot).grid(row=row, column=0, sticky=tk.W)
            row += 1

    def set_plot(self, plot_type):
        self.selected_plot = plot_type

    def get_plot(self):
        return self.selected_plot

class ConsoleFrame(ttk.LabelFrame):
    '''
    Console frame for showing actions
    '''

    def __init__(self, parent):
        super().__init__(parent, text="Console")

        self.log_text = tk.Text(self, wrap="word", state="disabled",
                                font=("TkDefaultFont", 20),
                                width=40, height=10)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.clear_button = ttk.Button(self, text="Clear",
                                       command=lambda: self.clear_log())
        self.clear_button.grid(row=1, column=0)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


class Frame(ttk.Frame):
    '''
    Define the Frame that will be the notebooks first tab named "Tab"
    '''

    def __init__(self, parent):
        super().__init__(parent)

        self.Plot = None
        self.console_frame = ConsoleFrame(self)
        self.console_frame.grid(column=0, row=1, padx=PADX, pady=PADY)
        # Setup frames inside of main frame
        self.directory_frame = DirectoryFrame(self, self.console_frame)
        self.directory_frame.grid(column=0, row=0, padx=PADX, pady=PADY)

        self.plot_frame = PlotFrame(self, self.console_frame)
        self.plot_frame.grid(column=1, row=0)
        self.channel_frame = ChannelFrame(self, self.console_frame)
        self.channel_frame.grid(column=1, row=1)
        self.plot_button = ttk.Button(self, text='Plot',
                                      command=lambda: self.generate_plot())
        self.plot_button.grid(row=2)

    def generate_plot(self):
        '''
        Generate a pop up window of a selected plot
        '''
        data_file = self.directory_frame.get_datafile()
        plot_type = self.plot_frame.get_plot()
        channels = self.channel_frame.get_channels()

        if data_file and plot_type is not None or len(channels) == 0:
            string = ''
            for i in channels:
               string = f'{string}, {i}'

            self.console_frame.log(f'Selected channels: {string}')
            self.console_frame.log(f'Plot_type: {plot_type}')
            self.console_frame.log(f'filepath: {data_file}')
            if self.Plot is None:
                self.Plot = Plot(plot_type, data_file, channels, self.console_frame)
            else:
                self.Plot.update(plot_type, data_file, channels)
            self.Plot.plot_data()
        else:
            if len(channels) == 0:
                self.console_frame.log("Error: No channels are selected")
            if data_file[0] is None:
                self.console_frame.log("Error: Data File is not selected")
            if plot_type is None:
                self.console_frame.log("Error: Plot Type is not selected")


class App(tk.Tk):
    '''
    Top level application class
    meta information and app window settings are defined here
    The top level layout information is described here
    '''

    def __init__(self):
        super().__init__()

        # Configure the app
        self.title('App')
        self.geometry('1400x850')
        self.minsize(1400, 800)
        self.resizable(True, True)

        # Define the top level layout

        # create notebook and add frame
        notebook = ttk.Notebook(self)
        notebook.add(Frame(self), text='Tab')

        # grid notebook
        notebook.grid(column=0, row=0, padx=PADX, pady=PADY, sticky="nsew")  # sticky="nsew" to make it fill the entire window

        # configure row and column weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    args = parser.parse_args()

    app = App()
    app.mainloop()
