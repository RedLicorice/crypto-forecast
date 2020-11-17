from scipy.interpolate import UnivariateSpline
import numpy as np

def get_spline(y, nu, degree=3):
    # The number of data points must be larger than the spline degree k.
    result = []
    for i in range(y.shape[0]):
        if i < degree:
            result.append(np.nan)
            continue
        x_space = np.linspace(0, i, i + 1)
        _y = y.iloc[0:i + 1]
        spl = UnivariateSpline(x_space, _y, s=0, k=degree)
        result.append(spl(i, nu=nu))
    return result