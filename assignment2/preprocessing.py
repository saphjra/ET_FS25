"""
Eye Tracking: Experiment design and machine learning methods
Assignment 2
Names:
Matriculation numbers:
"""

from argparse import ArgumentParser, FileType
from typing import TextIO
import polars as pl
import matplotlib.pyplot as plt


def read_file(infile: TextIO, trial_id: int):
    """
    Read in the raw eye tracking data
    :param infile: raw eye tracking file: trialID, pointID, time, x_left, y_left, pupil_left, x_right, y_right, pupil_right
    :param trial_id: the trial ID to focus on
    :return:
    """
    data = pl.read_csv(infile)
    data = data.filter(data['trialId'] == trial_id)
    print(data)
    return data


def ivt(gaze_data: pl.DataFrame, threshold: float, sampling_freq: int):
    """ Implementation of the velocity-based fixation detection algorithm.
    I-VT (gaze data, threshold)
    { Calculate the distance between consecutive points in gaze data
    Calculate velocities for each point in gaze data based on distance and sampling frequency
    if velocity(point) < threshold:
        label point fixation
    else: label point saccade

    glue consecutive fixation points together
    return fixation}"""
    last_point = None
    label = []
    gaze_data = gaze_data.with_columns(label=pl.col("x_left") > 0)

    for row in gaze_data.rows(named=True):
        if last_point is None:
            last_point = row
            continue
        else:
            current_point = row
            try:
                distance = ((current_point['x_right'] - last_point['x_right']) ** 2 + (
                            current_point['y_right'] - last_point['y_right']) ** 2) ** 0.5
            except TypeError:
                distance = 0  # pixels
            velocity = distance / (sampling_freq / 1000) # sampling frequency is in Hz, so we divide by 1000 to get ms
            print(distance, velocity)
            if velocity < threshold:  # velocity is in pixels/ms
                label.append("fixation")
            else:
                label.append("saccade")
            last_point = current_point
    label[0] = label[1]  # first point is changed to whatever second point was (or maybe we should remove it?)
    label.append(label[-1])  # last point is changed to whatever second last point was
    label = {"label": label}
    gaze_data = gaze_data.update(pl.DataFrame(label))
    print(gaze_data)
    return gaze_data


def idt(gaze_data: pl.DataFrame, dispersion_threshold: float, duration_threshold: int):
    """ Implementation of the duration-based fixation algorithm.
    I - DT (gaze data, dispersion threshold, duration threshold) {
     while points in gaze data initialize window in size of first duration threshold
     if dispersion(window) ≤ dispersion threshold:
         add additional points to window until dispersion(window) > dispersion threshold
         add fixation at centroid to fixations
         remove window points
     else
        Remove first point
     return fixations}"""
    pass


def plot_fixations(gaze_data: pl.DataFrame, trial_id: int, mode: str, sample_freq: int):
    """ Plot the gaze data and the detected fixations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    position_x = gaze_data['x_right']
    position_y = gaze_data['y_right']
    ax1.plot(position_x, color='blue', alpha=0.5, label='horizontal movement')
    ax1.plot(position_y, color='orange', alpha=0.5, label='vertical movement')
    # Detect start of a fixation: previous was "saccade", current is "fixation"

    df = gaze_data.with_row_index()

    # Detect start of a fixation
    fixation_start = df.filter(
        (df["label"] == "saccade") & (df["label"].shift(1) == "fixation")
    ).select(pl.col("index").alias("start"))

    # Detect end of a fixation
    fixation_end = df.filter(
        (df["label"] == "fixation") & (df["label"].shift(1) == "saccade")
    ).select(pl.col("index").alias("end"))

    # Combine start and end timestamps
    fixations = pl.concat([fixation_start, fixation_end], how="horizontal")[:-1]
    print(fixations)
    # Combine start and end timestamps
    #fixations = pl.concat([fixation_start, fixation_end])
    for start, end in zip(fixations["start"], fixations["end"]):
        #pass
        ax1.axvspan(start, end, color='grey', alpha=0.1)
    #ax1.axvspan(fixations["start"], fixations["end"], alpha=0.1, color='0.9')
    ax1.plot(position_x, color='blue', alpha=0.5, label='horizontal movement')
    ax1.plot(position_y, color='orange', alpha=0.5, label='vertical movement')

    ax1.set(xlim=(0 , len(gaze_data)),xlabel='sample num', ylabel='position')
    ax1.set(title=f"{mode} for trial {trial_id} with {sample_freq} Hz sampling frequency")
    plt.show()
    """
    ax1.axhspan(-1, 1, alpha=0.1)
    ax1.set(ylim=(-1.5, 1.5), title="fixation")

    mu = 8
    sigma = 2
    x = np.linspace(0, 16, 401)
    y = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    ax2.axvspan(mu - 2 * sigma, mu - sigma, color='0.95')
    ax2.axvspan(mu - sigma, mu + sigma, color='0.9')
    ax2.axvspan(mu + sigma, mu + 2 * sigma, color='0.95')
    ax2.axvline(mu, color='darkgrey', linestyle='--')
    ax2.plot(x, y)
    ax2.set(title="axvspan")

    plt.show()
    """


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Preprocess raw eye tracking data.')
    parser.add_argument('infile', type=FileType('r', encoding='utf-8'),
                        help='Input file with raw eye tracking data.')
    parser.add_argument('--mode', type=str, choices=['dispersion', 'velocity'], required=True,
                        help='The kind of algorithm to apply for fixation detection')
    parser.add_argument('--trial', type=int, help='the trial ID')
    parser.add_argument('--freq', type=int, choices=[60, 2000], required=True,
                        help='the sampling frequency')
    parser.add_argument('--vel_thres', type=float, help='the min velocity threshold for I-VT, in pixels/ms')
    parser.add_argument('--dis_thres', type=float, help='the max dispersion threshold for I-DT, in pixels')
    parser.add_argument('--dur_thres', type=float, default=200, help='the min duration threshold for I-DT in ms')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    gaze_data = read_file(args.infile, args.trial)
    if args.mode == 'velocity':
        gaze_data = ivt(gaze_data, args.vel_thres, args.freq)
    elif args.mode == 'dispersion':
        gaze_data = idt(gaze_data, args.dis_thres, args.dur_thres)
    plot_fixations(gaze_data, args.trial, args.mode, args.freq)


if __name__ == '__main__':
    main()
