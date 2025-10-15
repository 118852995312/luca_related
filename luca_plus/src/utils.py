
import numpy as np


def clean_seq(protein_id, seq, return_rm_index=False):
    seq = seq.upper()
    new_seq = ""
    has_invalid_char = False
    invalid_char_set = set()
    return_rm_index_set = set()
    for idx, ch in enumerate(seq):
        if 'A' <= ch <= 'Z' and ch not in ['J']:
            new_seq += ch
        else:
            invalid_char_set.add(ch)
            return_rm_index_set.add(idx)
            has_invalid_char = True
    if has_invalid_char:
        print("id: %s. Seq: %s" % (protein_id, seq))
        print("invalid char set:", invalid_char_set)
        print("return_rm_index:", return_rm_index_set)
    if return_rm_index:
        return new_seq, return_rm_index_set
    return new_seq





def process_outputs(truth, pred, output_truth, output_pred, ignore_index):


    cur_truth = truth.view(-1)
    cur_mask = cur_truth != ignore_index
    cur_pred = pred.view(-1)
    cur_truth = cur_truth[cur_mask]
    cur_pred = cur_pred[cur_mask]
    sum_v = cur_mask.sum().item()
    if sum_v > 0:
        cur_truth = cur_truth.detach().cpu().numpy()
        cur_pred = cur_pred.detach().cpu().numpy()
        if output_truth is None or output_pred is None:
            return cur_truth, cur_pred
        else:
            output_truth = np.append(output_truth, cur_truth,  axis=0)
            output_pred = np.append(output_pred, cur_pred,  axis=0)
            return output_truth, output_pred
    return truth, pred
