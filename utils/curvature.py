import numpy as np
import os,sys
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt


def extract_inflection_points(x: np.ndarray, y : np.ndarray) -> np.ndarray:
    """
    Get points with maximum curvature, before the curve maximum is reached

    Params
    ------
    x : time points to be processed
    y : curve to be processed (AIF)

    Returns
    -------
    t_inflect : computed inflection points through time

    """
    # Derive the time when the maximum value of Y is reached, since the inflection points 
    # should occur only before the maximum

    dy = np.gradient(y, x)        # First derivative
    d2y = np.gradient(dy, x)      # Second derivative

    # Inflection points are where the second derivative changes sign
    curvature = np.diff(np.sign(d2y))
    inflection_points = np.where(curvature)[0]  

    # filter inflection points before the curve's main maximum
    argmax = np.argmax(y)
    inflection_points = inflection_points[inflection_points < argmax]

    # Consider the inflection point with the biggest second derivative
    d2y_inflection = np.abs(d2y[inflection_points])
    ind_inflection = np.argmax(d2y_inflection)

    final_ind = inflection_points[ind_inflection]

    # Consider a couple of slices earlier than the inflection point
    # In case the AIF mask is being taken a bit later in the circulation
    if final_ind > 2:
        final_ind -= 2

    t_inflect = x[final_ind]

    return t_inflect


def second_cycle(x : np.ndarray, y : np.ndarray, thr_prominence : float = 20):
    """
    Determine if there is a second cycle in the AIF function

    Params
    ------
    x : time points to be processed
    y : curve to be processed (AIF)
    thr_prominence : threshold to consider a second peak found as prominent

    Returns
    -------
    t_prominent : time moment when a second peak happens
    
    """
    t_prominent = None # default values

    # Determine argmax
    argmax = np.argmax(y)

    # Determine peaks
    peaks = find_peaks(x = y, height=None)[0]

    if peaks.shape[0] > 0:
        # Consider only peaks after the argmax 
        peaks = peaks[peaks > argmax]
    
        if peaks.shape[0] > 0:
            # Determine prominences
            y_peaks = y[peaks] 
            prominence = peak_prominences(x = y, peaks=peaks)[0]

            # Determine relative prominences 
            prominence_rel = prominence*100/(y_peaks + np.finfo(float).eps)

            # Determine if some point exceeds the 
            # threshold of relative prominence 
            # If so, take the peak with the highest relative prominence
            inds = np.where(prominence_rel > thr_prominence)[0]
            if inds.shape[0] > 0:
                # Estimate prominence time 
                prominence_rel_argmax = np.argmax(prominence_rel)
                t_prominent = x[peaks[prominence_rel_argmax]] 
                

    return t_prominent


if __name__ == "__main__":
    file_time = "t.npy"
    file_aif = "aif.npy"
    t = np.load(file_time)
    aif = np.load(file_aif)

    t_inflect = extract_inflection_points(x=t, y=aif)
    t_prominent = second_cycle(x = t, y = aif)

    plt.figure()
    plt.plot(t,aif)
    plt.axvline(x = t_inflect, color="r")
    plt.axvline(x = t_prominent, color="r")
    plt.show()