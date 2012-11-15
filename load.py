import os
import re

sky_re = re.compile("Sky[0-9]+")

from pandas import *
# d = DataFrame.from_csv("data/Train_Skies/Training_"+sky+".csv")

# t = DataFrame.from_csv("data/Training_halos.csv")
# xvals = (t.ix[sky]['halo_x1'], t.ix[sky]['halo_x2'], t.ix[sky]['halo_x3'])
# yvals = (t.ix[sky]['halo_y1'], t.ix[sky]['halo_y2'], t.ix[sky]['halo_y3'])

def load_data_folder(folder_path):
    files = os.listdir(folder_path)
    all_d = None
    first = True
    for f in files:
        sky_id = sky_re.search(f).group()
        d = DataFrame.from_csv(os.path.join(folder_path, f))
        galaxy = np.asarray(d.index)
        sky = np.asarray([sky_id for _ in xrange(len(galaxy))])
        ind = MultiIndex.from_arrays([sky, galaxy], names=['sky', 'galaxy'])
        d.index = ind
        if first:
            all_d = d
            first = False
        else :
            all_d = all_d.append(d)

    return all_d
