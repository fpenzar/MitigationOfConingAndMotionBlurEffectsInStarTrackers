import numpy as np
import copy


class ErrorCalculator:

    def __init__(self):
        pass

    def get_aligned_errors(self, *error_sequences):
        t0 = np.min(np.array([seq[0][1] for seq in error_sequences]))
        processed_error_sequences = []
        t_latest_start = np.max(np.array([seq[0][1] for seq in error_sequences]))
        for err_seq in error_sequences:
            if err_seq[0][1] == t_latest_start:
                processed_error_sequences.append(err_seq)
                continue
            new_seq = []
            offset_angle = 0
            for angle, t in err_seq:
                if t < t_latest_start:
                    new_seq.append((0, t))
                    offset_angle = angle
                else:
                    new_seq.append((angle - offset_angle, t))
            processed_error_sequences.append(copy.deepcopy(new_seq))
        
        avg_errors = []
        out_sequences = []
        for seq in processed_error_sequences:
            avg_error = seq[-1][0] / (seq[-1][1] - t_latest_start)
            avg_errors.append((avg_error, seq[-1][1] - t_latest_start))
            out_sequences.append(np.array([(float(e), float(t - t0)) for (e, t) in seq]))

        return out_sequences, avg_errors