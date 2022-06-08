import numpy as np
from scipy import stats
from .const import MIN_COUNT


def convert_edges_to_centers(edges):
    return edges[:-1] + (edges[1] - edges[0]) / 2.0


def generic_pdf(x, bin_size, bounds = None):
    if bounds:
        if bounds[0] > bounds[1] or bounds[0] == bounds[1]:
            x_new = x
        else:
            x_new = np.delete(x, 
                              np.where((x < bounds[0]) | (x > bounds[1])))
    else:
        x_new = x
        bounds = [np.amin(x_new), np.amax(x_new)]
        
    mean = np.mean(x_new)
    median = np.median(x_new)
    std = np.std(x_new)
    
    # We have bin_size, must convert to num_bins
    num_bins = int((bounds[1] - bounds[0]) / bin_size)
    
    pdf, edges = np.histogram(x_new, num_bins, density = True)
    
    return {'pdf': pdf * bin_size, 
            'centers': convert_edges_to_centers(edges),
            'mean': mean,
            'median': median,
            'std': std}


def generic_binning(x, y, bin_size, min_count = MIN_COUNT):
    # We have bin_size, must convert to num_bins
    num_bins = (np.amax(x) - np.amin(x)) / bin_size
    
    # There has to be a faster way to do this!
    # (stat, bin_edges, bin_number)
    mean, be, bn = stats.binned_statistic(x, y, statistic = 'mean', 
                                    bins = num_bins)
    median, be, bn = stats.binned_statistic(x, y, statistic = 'median',
                                    bins = num_bins)
    std, be, bn = stats.binned_statistic(x, y, statistic = np.std, 
                                    bins = num_bins)
    count, be, bn = stats.binned_statistic(x, y, statistic = 'count',
                                           bins = num_bins)

    min_count_idx = np.where(count < min_count)
    
    mean[min_count_idx] = np.nan
    median[min_count_idx] = np.nan
    
    return {'mean': mean,
            'median': median,
            'std': std,
            'centers': convert_edges_to_centers(be),
            'count': count}