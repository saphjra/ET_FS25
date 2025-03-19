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
#import defaultdict


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
#    fixations = defaultdict()
    for row in gaze_data.rows(named=True):
        if last_point is None:
            last_point = row
            continue
        else:
            current_point = row
            try:
                distance = ((last_point['x_right'] - current_point['x_right']) ** 2 + (
                            last_point['y_right'] - current_point['y_right']) ** 2) ** 0.5
            except TypeError:
                distance = 0  # pixels
            velocity = distance / (sampling_freq / 1000) # sampling frequency is in Hz, so we divide by 1000 to get ms
            #print(distance, velocity)
            if velocity < threshold:  # velocity is in pixels/ms
                label.append("fixation")
            else:
                label.append("saccade")
            last_point = current_point

    label[0] = label[1]  # first point is changed to whatever second point was (or maybe we should remove it?)
    label.append(label[-1])  # last point is changed to whatever second last point was
    label = {"label": label}
    gaze_data = gaze_data.update(pl.DataFrame(label))
    last_point = None
    fixations = {}
    current_fixation = 0
    for row in gaze_data.rows(named=True):
        if last_point is None:
            last_point = row
            continue
        else:
            current_point = row

        if last_point['label'] == "saccade" and current_point['label'] == "fixation":
            fixations[current_fixation] ={'fix_start': current_point['time'],
                                          'x_right': current_point['x_right'],
                                          'y_right': current_point['y_right'],
                                          'pointId': current_point['pointId'],
                                          'fix_end': None}
            current_fixation += 1
        if last_point['label'] == "fixation" and current_point['label'] == "saccade":
            fixations[current_fixation-1]['fix_end'] = current_point['time']

        last_point = current_point
    print(fixations)
    #df = gaze_data
    #fixations = df.filter(
    #    (df["label"] == "fixation") & (df["label"].shift(-1) == "saccade") or (df["label"] == "saccade") & (df["label"].shift(1) == "fixation")
    #)
    #points = df["pointId"].to_list()
    #for point in points:

    #print(fixations)
    return fixations


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
    def _comp_disp(current_window):
        try:
            duration = (max(current_window['time']) - min(current_window['time']))
            dispersion = 0.5 *(max(current_window['x_right'])-min(current_window['x_right'])) + (max(current_window['y_right'])-min(current_window['y_right']))
        except TypeError:
            duration = duration_threshold + 1
            dispersion = dispersion_threshold + 1
        return dispersion, duration

    current_window = {"time": [], 'x_right':[], 'y_right':[]}
    current_fixation = 0
    fixations = {}
    for row in gaze_data.rows(named=True):
        #window = gaze_data.filter(pl.col['time'] <= row['time'] + duration_threshold )

        current_window['time'].append(row['time'])
        current_window['x_right'].append(row['x_right'])
        current_window['y_right'].append(row['y_right'])

        disperison, duration =_comp_disp(current_window)

        if duration >= duration_threshold and disperison <= dispersion_threshold:
            fixations[min(current_window['time'])] = {'fix_start': min(current_window['time']),
                                           'x_right': row['x_right'],
                                           'y_right': row['y_right'],
                                           'pointId': row['pointId'],
                                           'fix_end': max(current_window['time'])}
        elif disperison >= dispersion_threshold:
            current_window = {"time": [], 'x_right':[], 'y_right':[]}

        elif duration < duration_threshold:
            continue

    print(fixations)
    return(fixations)



def plot_fixations(gaze_data: pl.DataFrame, trial_id: int, mode: str, sample_freq: int, fixations, threshold=None):
    """ Plot the gaze data and the detected fixations."""
    fig, ax1 = plt.subplots(figsize=(10, 5))


    position_x = gaze_data['x_right']
    position_y = gaze_data['y_right']
    time = gaze_data['time']
    # Detect start of a fixation: previous was "saccade", current is "fixation"


    for fixation in fixations.items():
        print(fixation)
        #pass
        ax1.axvspan(fixation[1]['fix_start'], fixation[1]['fix_end'], color='grey', alpha=0.1)
    #ax1.axvspan(fixations["start"], fixations["end"], alpha=0.1, color='0.9')
    ax1.plot(time, position_x,   color='blue', alpha=0.5, label='horizontal movement')
    ax1.plot(time, position_y, color='orange', alpha=0.5, label='vertical movement')

    ax1.set(xlabel='time', ylabel='coordinate position')
    plt.figlegend()
    ax1.set(title=f"{mode} for trial {trial_id} with {sample_freq} Hz sampling frequency")

    plt.show()
    plt.savefig(f"{mode}_trial_{trial_id}_sampling_{sample_freq}_{threshold}_100.png")



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
        fixations = ivt(gaze_data, args.vel_thres, args.freq)
        threshold = args.vel_thres
    elif args.mode == 'dispersion':
        fixations = idt(gaze_data, args.dis_thres, args.dur_thres)
        threshold = args.dis_thres
    plot_fixations(gaze_data, args.trial, args.mode, args.freq, fixations=fixations, threshold=threshold)


if __name__ == '__main__':
    main()
