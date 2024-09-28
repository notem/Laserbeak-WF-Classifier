from argparse import ArgumentParser
import numpy as np
import pickle



parser = ArgumentParser()
parser.add_argument('--undef')
parser.add_argument('--defen')
parser.add_argument('--perc', type=int, default=50)
parser.add_argument('--lower', default=False, action="store_true")
args = parser.parse_args()



with open(args.undef, 'rb') as fi:
    undef_data = pickle.load(fi)
with open(args.defen, 'rb') as fi:
    defen_data = pickle.load(fi)



def get_load_times(data, perc=50):
    load_times = {}
    if isinstance(data, dict):
        for cls in data.keys():
            load_times[cls] = []
            samples = data[cls]
            for sample in samples:
                load_times[cls].append(abs(min([s[-1] for s in sample])))
    else:
        load_times[-1] = []
        for i in range(len(data)):
            sample = data[i][0]
            load_times[-1].append(abs(sample[-1]))
    return {cls: np.percentile(load_times[cls], perc) for cls in load_times.keys()}


def get_totals(data, med_load_times, lower=False):
    total_packets = 0
    if isinstance(data, dict):
        for cls in data.keys():
            samples = data[cls]
            for sample in samples:
                less_than = abs(sample[0][-1]) <= med_load_times[cls]
                if (lower and less_than) or \
                        (not lower and not less_than):
                    total_packets += len(sample[0])
    else:
        for i in range(len(data)):
            sample = data[i][0]
            less_than =  abs(sample[-1]) < med_load_times
            if (lower and less_than) or \
                    (not lower and not less_than):
                total_packets += len(sample)

    return total_packets


load_times = get_load_times(undef_data, args.perc)
undef_totals = get_totals(undef_data, load_times, lower=args.lower)
defen_totals = get_totals(defen_data, load_times, lower=args.lower)

print(f'bw: {defen_totals / undef_totals:.2f}')
