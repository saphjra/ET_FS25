# Assignment 2 - Interpretation
**Eye Tracking: Experiment design and machine learning methods**

*Andreas Loizidis, Jana Hofmann*


## Effect of sampling frequency
The dispersion-based method seems to work better at the measurements taken at the lower sampling rate of 60Hz. As more frequent measures are taken, there tends to be higher variability in-between points, often exceeding the dispertion threshold.


### Dispertion-based method
As the **dispertion threshold** increases, the algorithm is less tolerant on data fluctuations, generating more fixations where it might not be useful.
This is exacerbated for the higher frequency of 2kHz (for example, a large 'hump' is included in the first fixation for duration threshold=100 and dispertion threshold=20. That is not the case for dispertion threshold=12 (see `sampling_2000_DUR100_DIS20.png` and `sampling_2000_DUR100_DIS12.png` in `all_dispertion_trial_13/sampling_2000hz/`. 
 

The effect of manipulating the minimum duration is inconclusive.

### Velocity-based method
This method doesn't seem too useful for either sampling frequency in the given data. While a higher sampling frequency should in theory reveal the ends of fixation points, in the given data, even with a low velocity threshold, high-speed movements register as many micro-fixation points.

