import load as myload
import sys
import numpy as np
from pandas import *
import scipy.ndimage as im

all_data = myload.load_data_folder(sys.argv[1])

groups = all_data.groupby(level='sky')

n_skies = len(groups)
n_bins = 42

output = np.ndarray(shape=(n_skies, 3, n_bins, n_bins), dtype='float32')
output_90 = output.copy()
sky_names = []

hist_kwargs={"bins":n_bins,
             "range": ((0,4200),(0,4200))
            }

i = 0
for sky, data in groups:
    # keep track of the order we go through skies
    sky_names.append(sky)

    bin_occupancy, _, _ = np.histogram2d(data.x, data.y, **hist_kwargs)

    sum_e1, _, _ = np.histogram2d(data.x, data.y, weights=data.e1, **hist_kwargs)
    avg_e1 = np.nan_to_num(sum_e1/bin_occupancy)

    sum_e2, _, _ = np.histogram2d(data.x, data.y, weights=data.e2, **hist_kwargs)
    avg_e2 = np.nan_to_num(sum_e2/bin_occupancy)

    # are there galaxies in each bin?
    output[i][0] =  bin_occupancy > 0
    # average ellipticity in each direction
    output[i][1] = avg_e1
    output[i][2] = avg_e2

    output_90[i][0] = im.rotate(bin_occupancy, 90) > 0
    output_90[i][1] = -avg_e1
    output_90[i][2] = -avg_e2

    i += 1

# now do targets
# we just want a 1 in the bin where the center is, or a zero otherwise

targets = np.ndarray(shape=(n_skies, 6), dtype='int32')

t = DataFrame.from_csv("data/Training_halos.csv").reindex(sky_names)

targets = np.asarray(t[['halo_x1','halo_y1', 'halo_x2', 'halo_y2', 'halo_x3', 'halo_y3']])

np.savez(sys.argv[2], output=output, targets=targets, sky_name=sky_names)


