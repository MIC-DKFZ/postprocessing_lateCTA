import numpy as np
import os,sys
import scipy.signal as signal
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


if __name__ == "__main__":
    file_time = "t.npy"
    file_aif = "aif.npy"
    t = np.load(file_time)
    aif = np.load(file_aif)

    t_inflect = extract_inflection_points(x=t, y=aif)

    print(t_inflect)

    plt.figure()
    plt.plot(t,aif)
    plt.scatter(t_inflect,0.)
    plt.show()