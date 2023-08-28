import torch



def rate_estimator(iats):
    """Simple/naive implementation of a running average traffic flow rate estimator
       It is entirely vectorized, so it is fast
    """
    times = torch.cumsum(iats, dim=0)
    indices = torch.arange(1, iats.size(0) + 1)
    flow_rate = torch.where(times != 0, indices / times, torch.ones_like(times))
    return flow_rate


def weighted_rate_estimator(iats, k=0.1):
    """Implementation of a traffic flow rate estimation function with an expoential decay
       follows guidance from: https://stackoverflow.com/questions/23615974/estimating-rate-of-occurrence-of-an-event-with-exponential-smoothing-and-irregul?noredirect=1&lq=1
    """
    times = torch.cumsum(iats, dim=0)
    exps1 = torch.exp(k * -iats)
    exps2 = torch.exp(k * -times)
    rates = [0]
    for i in range(len(iats)-1):
        rate = k + (exps1[i+1] * rates[i])
        rate = rate / (1 - exps2[i+1])
        rate = torch.clip(torch.nan_to_num(rate, nan=1e4), 1e4)
        rates.append(rate)
    return torch.tensor(rates)


class DataProcessor:
    """ Initialize with desired list of features
        Apply to map a raw traffic sample into it's feature representation
    """

    # list of valid processor options and any opts that depend on them
    DEPENS = {
                'times': [],        # times & dirs are always available so no need to list depens
                'dirs': [], 
                'time_dirs': [],    # tik-tok style representation

                'burst_edges': [],  # a sparse representation that identifies whenever traffic changes direction

                'cumul': ['cumul_norm'],  # per-packet cumulative sum (based on cumul)
                'cumul_norm': [],         # normalize cumul features into [-1,+1] range, centered around the mean
                'times_norm': [],         # normalize timestamp features into [-1,+1] range, centered around the mean

                'iats': ['iat_dirs'],     # difference between consequtive timestamps (e.g., inter-arrival-times
                'iat_dirs': [],           # iats with direction encoded into the representation

                'running_rates': ['running_rates_diff'],   # running average of flow rate (ignores direction)
                'running_rates_diff': [],                  # instantaneous change of flow rate
                'running_rates_decayed': ['up_rates_decayed', 'down_rates_decayed'],       # running average with exponential decay on old rates (expensive to compute)
                'up_rates_decayed': [],       # up-direction only (non-aligned)
                'down_rates_decayed': [],     # down-direction only (non-aligned)

                'up_iats': ['up_iats_sparse', 'up_rates', 'flow_iats'],         # iats computed on upload-only pkt sequence (is not packet aligned with time/dir seq)
                'down_iats': ['down_iats_sparse', 'down_rates', 'flow_iats'],   # iats computed on download-only pkt sequence (is not packet aligned with time/dir seq)
                'up_iats_sparse': [],                  # sparsified sequence (e.g., download pkts have value of zero)
                'down_iats_sparse': [],                # sparsified sequence (e.g., upload pkts have value of zero)
                'up_rates': ['up_rates_sparse'],       # simple rate estimator applied to up_iats
                'down_rates': ['down_rates_sparse'],   # simple rate estimator applied to down_iats
                'up_rates_sparse': [],        # sparsified, but pck-aligned sequence
                'down_rates_sparse': [],      # sparsified, but pck-aligned sequence

                'flow_iats': ['burst_filtered_times', 'inv_iat_logs'],  # up & down iats merged into one sequence (aligned with time/dir seqs)
                'burst_filtered_times': ['burst_filtered_time_dirs'],  # filtered sequence with large gaps removed (non-aligned)
                'burst_filtered_time_dirs': [],                        # with direction encoded (non-aligned)
                'inv_iat_logs': ['inv_iat_log_dirs'],    # log applied to the inverse of flow iats (adjust with +1 to avoid negative logs)
                'inv_iat_log_dirs': [],                  # with pkt direction encoded
             }


    def __init__(self, process_options = ('dirs',)):
        self.process_options = process_options if process_options else {}
        self.input_channels = len(self.process_options)

        assert len(self.process_options) > 0
        assert all(opt in self.DEPENS.keys() for opt in self.process_options)

    def _resolve_depens(self, opt):
        """get list of options that depend on opt"""
        depens = []
        for depen in self.DEPENS[opt]:
            depens.append(depen)
            depens.extend(self._resolve_depens(depen))
        return depens

    def _is_enabled(self, *opts):
        """if opt or any of its dependencies are in self.process_options, then func returns true"""
        required = list(opts)
        for opt in opts:
            required.extend(self._resolve_depens(opt))
        return any(opt in self.process_options for opt in required)

    def process(self, x):
        """Map raw metadata to processed pkt representations
        """
        size = len(x)

        def fix_size(z):
            if z.size(0) < size:
                z = F.pad(z, (0,size - z.size(0)))
            elif z.size(0) > size:
                z = z[:size]
            return z

        feature_dict = {}

        times = torch.abs(x)
        feature_dict['times'] = times
        dirs = torch.sign(x)
        feature_dict['dirs'] = dirs

        upload = dirs > 0
        download = ~upload

        if self._is_enabled("time_dirs"):
            feature_dict['time_dirs'] = times * dirs

        if self._is_enabled("times_norm"):
            # subtract mean and normalize by max
            times_norm = times.clone()
            times_norm -= torch.mean(times_norm)
            times_norm /= torch.amax(torch.abs(times_norm))
            feature_dict['times_norm'] = times_norm

        if self._is_enabled("iats"):
            # 1st-order diff of timestamps shows inter-packet arrival times
            iats = torch.diff(times, prepend=torch.tensor([0]))
            feature_dict['iats'] = iats

        if self._is_enabled("cumul"):
            # Direction-based representations
            cumul = torch.cumsum(dirs, dim=0)   # raw accumulation
            feature_dict['cumul'] = cumul

        if self._is_enabled("cumul_norm"):
            # subtract mean and normalize by max
            cumul_norm = cumul.clone()
            cumul_norm -= torch.mean(cumul_norm)
            cumul_norm /= torch.amax(torch.abs(cumul_norm))
            feature_dict['cumul_norm'] = cumul_norm

        if self._is_enabled("burst_edges"):
            # 1st-order diff of directions detects burst boundaries (with value +/-2)
            burst_edges = torch.diff(dirs, prepend=torch.tensor([0]))
            feature_dict['burst_edges'] = burst_edges

        if self._is_enabled("iat_dirs"):
            # adjusted iats by +1 to prevent zeros loosing directional representation
            iat_dirs = (1. + iats) * dirs
            feature_dict['iat_dirs'] = iat_dirs

        if self._is_enabled('running_rates'):
            running_rates = rate_estimator(iats)
            feature_dict['running_rates'] = running_rates

        if self._is_enabled('running_rates_diff'):
            running_rate_diff = torch.diff(running_rates, prepend = torch.tensor([0]))
            feature_dict['running_rates_diff'] = running_rate_diff

        if self._is_enabled('running_rates_decayed'):
            running_rates_decay = weighted_rate_estimator(iats)
            feature_dict['running_rates_decayed'] = running_rates_decay

        if self._is_enabled('up_iats'):
            upload_iats = torch.diff(times[upload], prepend=torch.tensor([0]))
            feature_dict['up_iats'] = upload_iats

        if self._is_enabled('down_iats'):
            download_iats = torch.diff(times[download], prepend=torch.tensor([0]))
            feature_dict['down_iats'] = download_iats

        if self._is_enabled('up_rates'):
            up_rates = rate_estimator(upload_iats)
            feature_dict['up_rates'] = up_rates

        if self._is_enabled('up_rates_sparse'):
            sparse_up_rate = torch.zeros_like(times)
            sparse_up_rate[upload] = up_rates
            feature_dict['up_rates_sparse'] = sparse_up_rates

        if self._is_enabled('up_rates_decayed'):
            up_rates_decay = weighted_rate_estimator(upload_iats)
            feature_dict['up_rates_decayed'] = up_rates_decay
            #sparse_up_rate_decay = torch.zeros_like(times)
            #sparse_up_rate_decay[upload] = up_rates_decay

        if self._is_enabled('down_rates'):
            down_rates = rate_estimator(download_iats)
            feature_dict['down_rates'] = down_rates

        if self._is_enabled('down_rates_sparse'):
            sparse_down_rate = torch.zeros_like(times)
            sparse_down_rate[download] = down_rates
            feature_dict['down_rates_sparse'] = sparse_down_rates

        if self._is_enabled('down_rates_decayed'):
            down_rates_decay = weighted_rate_estimator(download_iats)
            feature_dict['down_rates_decayed'] = down_rates_decay

        ## recombine calculated iats into chronological flow
        if self._is_enabled('flow_iats'):
            flow_iats = torch.zeros_like(times)
            flow_iats[upload] = upload_iats
            flow_iats[download] = download_iats
            feature_dict['flow_iats'] = flow_iats

        ## filter times by bursts (returns sparse vector)
        if self._is_enabled('burst_filtered_times'):
            delta_times = flow_iats < 0.01
            feature_dict['burst_filtered_times'] = times[delta_times]

        if self._is_enabled('burst_filtered_time_dirs'):
            feature_dict['burst_filtered_time_dirs'] = times[delta_times] * dirs[delta_times]

        if self._is_enabled('inv_iat_logs'):
            # inverse log of iats (adjusted from original to keep logs positive)
            inv_iat_logs = torch.log(torch.nan_to_num((1 / flow_iats)+1, nan=1e4, posinf=1e4))
            feature_dict['inv_iat_logs'] = inv_iat_logs

        if self._is_enabled('inv_iat_log_dirs'):
            feature_dict['inv_iat_log_dirs'] = inv_iat_logs * dirs


        # adjust feature vectors sizes to match traffic sequence length and stack
        feature_stack = list(fix_size(feature_dict[opt]) for opt in self.process_options)
        features = torch.nan_to_num(torch.stack(feature_stack, dim=-1))

        #assert not torch.any(features.isnan())
        #assert not torch.any(features.isinf())

        return features

    def __call__(self, x):
        return self.process(x)
