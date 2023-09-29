import numpy
import pickle
import os
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--pickle', required=True, type=str)
    args = parser.parse_args()
    
    with open(args.pickle, 'rb') as fi:
        data = pickle.load(fi)
    
    length_dist = []
    if isinstance(data, dict):
        for key in data:
            for sample_list in data[key]:
                for sample in sample_list:
                    length_dist.append(len(sample))
    elif isinstance(data, list):
        for sample_list in key:
            for sample in sample_list:
                length_dist.append(len(sample))
    
    percentiles = (50, 60, 70, 75, 80, 90, 95)
    statistics = numpy.percentile(length_dist, percentiles)
    print(percentiles)
    print(statistics)
