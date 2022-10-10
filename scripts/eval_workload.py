import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# =======================================================================================
# Analysis stats about the workload of a script, such as CPU load or memory consumption
# =======================================================================================


class Evalload:
    def __init__(self, args, input_path, output_path):
        self.args = args
        self.input_path = input_path
        self.output_path = output_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def readsignal(self, filename):

        data = {"virt": [], "res": [], "shr": [], "cpu": [], "mem": [], "time": []}
        with open(os.path.join(self.input_path, filename)) as f:
            for line in f:
                if self.args.user in line:
                    curr_data = " ".join(line.split()).split()
                    data["virt"].append(float(curr_data[4].replace(",", ".")))
                    data["res"].append(float(curr_data[5].replace(",", ".")))
                    data["shr"].append(float(curr_data[6].replace(",", ".")))
                    data["cpu"].append(float(curr_data[8].replace(",", ".")))
                    data["mem"].append(float(curr_data[9].replace(",", ".")))
                    data["time"].append(curr_data[10])

        return data

    def plotsignal(self, filename, signal):
        """
        Plots stats over time
        :param signallist: list of names of variables we want to plot
        """
        plt.rc("xtick", labelsize=14)
        plt.rc("ytick", labelsize=14)
        plt.rc("axes", labelsize=14)

        data = self.readsignal(filename)

        # width as measured in inkscape
        width = 1.5 * 3.487
        height = width / 1.618

        fig, ax = plt.subplots(figsize=(20, 15))
        # fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        plt.grid(linestyle="--", color="silver", which="both")

        plt.plot(data["time"], data[signal], "b-", linewidth=1.0, label=signal)

        # Axis
        ax.set_title("Performance Analysis")

        if signal == "cpu" or signal == "mem":
            ax.set_ylabel(signal + " [%]")
        elif signal == "shr":
            ax.set_ylabel(signal + " [kb]")
        else:
            ax.set_ylabel(signal)

        ax.set_xlabel("time")
        ax.xaxis.set_ticks(np.asarray(data["time"])[::50])
        ax.set_xticklabels(np.asarray(data["time"])[::50], rotation=40)

        # fig.set_size_inches(width, height)
        fig.tight_layout()

        plot_name = signal + ".jpg"
        fig.savefig(os.path.join(self.output_path, plot_name), dpi=200)

        print("figure created.")


if __name__ == "__main__":
    # Paths
    input_frame_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "input"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))

    # Get input arguments from shell
    parser = argparse.ArgumentParser("Visualize Workload")

    # General configs
    parser.add_argument(
        "--filename",
        default="top.txt",
        type=str,
        help="Specify filename of recorded performance load",
    )
    parser.add_argument("--user", default="pmmon", type=str, help="Specify username")

    # Video Configs
    parser.add_argument(
        "--signalname", default="cpu", type=str, help="Name of variable to analyze"
    )
    parser.add_argument(
        "--height", default=346, type=int, help="Specify height of image and video"
    )
    parser.add_argument(
        "--width", default=260, type=int, help="Specify width of image and video"
    )

    # Get arguments
    args = parser.parse_args()

    # Run Visualization
    tracker = Evalload(args, input_frame_path, output_path)
    tracker.plotsignal(args.filename, args.signalname)
