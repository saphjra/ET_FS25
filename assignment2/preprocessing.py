"""
Eye Tracking: Experiment design and machine learning methods
Assignment 2
Names:
Matriculation numbers:
"""

from argparse import ArgumentParser, FileType
from typing import TextIO
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import os

def ivt(dataframe, velocity_thr: float, sampling_freq: int):
    """ Implementation of the velocity-based fixation detection algorithm.
        I-VT (gaze data, threshold)
        { Calculate the distance between consecutive points in gaze data
        Calculate velocities for each point in gaze data based on distance and sampling frequency
        if velocity(point) < threshold:
            label point fixation
        else: label point saccade

        glue consecutive fixation points together
        return fixation}"""

    points = dataframe[['time', 'x_right', 'y_right']].rename(
        columns={'x_right': 'x', 'y_right': 'y'}).to_dict('records')

    last_point = None
    fixations = []
    current_fixation = None
    for p in points:
        # if abs(8.055*1_000_000-p['time']) <= 20:
        #     print("here")
        if not last_point:
            last_point = p
            continue
#       to get the velocity, we first calc the (Euclidean) distance to the previous point
        try:
            distance = ((last_point['x'] - p['x']) ** 2 + (
                    last_point['y'] - p['y']) ** 2) ** 0.5
        except TypeError:
            distance = 0  # pixels


        dt_seconds = 1/sampling_freq
        dt_ms = dt_seconds * 1000
        velocity = distance / dt_ms

        if velocity < velocity_thr:  # velocity is in pixels/ms
            # part of a fixation
            if not current_fixation:
                current_fixation = {
                    'start': p['time'],
                }
        else:
            # otherwise it's a saccade
            # if there's an open fixation, close it (its stop time is the previous point's time)
            # unless the duration of the fixation is 0, then just ignore it!
            if current_fixation:
                duration = last_point['time'] - current_fixation['start']
                if duration>0:
                    current_fixation['duration'] = duration
                    fixations.append(current_fixation)
                current_fixation=None

        last_point = p
    return fixations

def idt(dataframe, DI_T, DU_T):
    points = dataframe[['time', 'x_right', 'y_right']].rename(
        columns={'x_right': 'x', 'y_right': 'y'}).to_dict('records')


    def calc_disp(window_points):
        keys = ['x', 'y']
        range_total = 0
        for k in keys:
            max_v = max(window_points, key=lambda x: x[k])[k]
            min_v = min(window_points, key=lambda x: x[k])[k]
            range_v = max_v - min_v
            range_total += range_v
        return range_total / len(keys)

    def init_window_of_duration(pts, dur):
        max_time = pts[0]['time'] + dur
        return [p for p in pts if p['time'] <= max_time]

    def excluding_window_points(w, pts):
        ids = [p['time'] for p in w]
        return [p for p in pts if p['time'] not in ids]

    def fixation_from_window(w):
        min_time = min(w, key=lambda x: x['time'])['time']
        max_time = max(w, key=lambda x: x['time'])['time']
        duration = max_time - min_time
        return {
            'start': min_time,
            'duration': duration,
        }

    fixations = []
    while len(points) > 0:
        window = init_window_of_duration(points, DU_T)
        disp = calc_disp(window)
        if disp <= DI_T:
            # now we need to find additional points to add to our window
            points_left = excluding_window_points(window, points)
            i = 0
            while disp <= DI_T and i < len(points_left):
                window.append(points_left[i])
                disp = calc_disp(window)
                i += 1
            if disp > DI_T:
                window.pop()


            fixations.append(fixation_from_window(window))
            #       now we have to remove the points of the window
            #       from the list we consider in the outer loop
            points = excluding_window_points(window, points)
        else:
            points.pop(0)
    return fixations




def plot_and_save(dataframe: pd.DataFrame, plot_title, plot_path, fixations):
    plt.figure(figsize=(12, 6))
    plt.plot(dataframe['time'], dataframe['x_right'], label='Horizontal Position (x_right)')
    plt.plot(dataframe['time'], dataframe['y_right'], label='Vertical Position (y_right)')

    # fixations
    for i, fixation in enumerate(fixations):
        start_time = fixation['start']
        end_time = fixation['start'] + fixation['duration']
        # add the label only for the first fixation to avoid duplicate legend entries
        label = 'Fixation' if i == 0 else None
        plt.axvspan(start_time, end_time, color='yellow', alpha=0.3, label=label)


    # labeling the plot
    plt.xlabel('Time (ms)')
    plt.ylabel('Position')
    plt.title(plot_title)
    plt.legend()
    plt.savefig(plot_path)
    plt.close()



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Preprocess raw eye tracking data.')
    parser.add_argument('--infile', type=str,
                        default='JumpingDots60.csv', help='Input CSV file with raw eye tracking data.')
    parser.add_argument('--mode', type=str, choices=['dispersion', 'velocity'],
                        help='The kind of algorithm to apply for fixation detection', default='dispersion')
    parser.add_argument('--trial', type=int, help='the trial ID', default=7)
    parser.add_argument('--freq', type=int, choices=[60, 2000],
                        help='the sampling frequency', default=60)
    parser.add_argument('--vel_thres', type=float, help='the min velocity threshold for I-VT, in pixels/ms')
    parser.add_argument('--dis_thres', type=float, default=12, help='the max dispersion threshold for I-DT, in pixels')
    parser.add_argument('--dur_thres', type=float, default=200, help='the min duration threshold for I-DT in ms')

    return parser



if __name__ == '__main__':

    # Use these flags to quickly plot all the combinations of each method.
    TRY_ALL_DISPERTION = False
    TRY_ALL_VELOCITY = False
    TRIAL_ID = 13
    if TRY_ALL_DISPERTION:
        path = 'all_dispertion_trial_{}'.format(TRIAL_ID)
        os.makedirs(path, exist_ok=True)
        durations = [
            100,
            200
        ]
        dispersions = [
            6,
            12,
            20
        ]
        fnames = {
            '60': 'JumpingDots60.csv',
            '2000': 'JumpingDots2000.csv',
        }
        for code, fname in fnames.items():
            freq_path = os.path.join(path, f"sampling_{code}hz")
            os.makedirs(freq_path, exist_ok=True)
            for d in durations:
                for disp in dispersions:
                    fig_name = f"sampling_{code}_DUR{d}_DIS{disp}.png"
                    fig_path = os.path.join(freq_path, fig_name)
                    df = pd.read_csv(fname)
                    df = df[(df['trialId'] == TRIAL_ID)]
                    fixations = idt(dataframe=df, DI_T=disp, DU_T=d)
                    fig_title = f"{code} Hz, Trial {TRIAL_ID} (Dispersion, {disp} pixels, {d}ms)"
                    plot_and_save(df, fig_title, fig_path, fixations)
    elif TRY_ALL_VELOCITY:
        thresholds = [
            1,
            2,
            5,
            10
        ]
        fnames = {
            '60': 'JumpingDots60.csv',
            '2000': 'JumpingDots2000.csv',
        }
        path = 'all_velocity_trial_{}'.format(TRIAL_ID)
        os.makedirs(path, exist_ok=True)
        for freq, fname in fnames.items():
            freq_path = os.path.join(path, f"sampling_{freq}hz")
            os.makedirs(freq_path, exist_ok=True)
            for thr in thresholds:
                fig_name = f"sampling_{freq}_THR{thr}.png"
                fig_path = os.path.join(freq_path, fig_name)
                df = pd.read_csv(fname)
                df = df[(df['trialId'] == TRIAL_ID)]
                fixations = ivt(dataframe=df, velocity_thr=thr, sampling_freq=int(freq))
                fig_title = f"{freq} Hz, Trial {TRIAL_ID} (Velocity, {thr} pixels/ms)"
                plot_and_save(df, fig_title, fig_path, fixations)



    parser = get_parser()
    args = parser.parse_args()
    gaze_data = pd.read_csv(args.infile)
    gaze_data = gaze_data[(gaze_data['trialId'] == args.trial)]

    # figures will be created in a(n aptly named) 'figures' directory
    path = 'figures'
    os.makedirs(path, exist_ok=True)

    if args.mode == 'velocity':
        fixations = ivt(gaze_data, args.vel_thres, args.freq)
        fname = f"{args.mode}_trial_{args.trial}_sampling_{args.freq}_VEL-THR{args.vel_thres}.png"
        fig_title = f"{args.mode} for trial {args.trial} with {args.freq} Hz sampling frequency"

    elif args.mode == 'dispersion':
        fixations = idt(gaze_data, args.dis_thres, args.dur_thres)
        fname = f"{args.mode}_trial_{args.trial}_sampling_{args.freq}_DUR{args.dur_thres}_DIS{args.dis_thres}.png"
        fig_title = f"{args.freq} Hz, Trial {args.trial} ({args.mode}, {args.dis_thres} pixels, {args.dur_thres}ms)"
    else:
        raise ValueError('Invalid mode')


    fig_path = os.path.join(path, fname)

    plot_and_save(gaze_data, fig_title, fig_path, fixations)
