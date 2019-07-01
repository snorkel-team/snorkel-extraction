import argparse
import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class MetalDataset(Dataset):
    """A dataset that group each item in X with its label from Y

    Args:
        X: an n-dim iterable of items
        Y: a torch.Tensor of labels
            This may be predicted (int) labels [n] or probabilistic (float) labels [n, k]
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert len(X) == len(Y)

    def __getitem__(self, index):
        return tuple([self.X[index], self.Y[index]])

    def __len__(self):
        return len(self.X)


def label_matrix_to_one_hot(L, k=None):
    """Converts a 2D [n,m] label matrix into an [n,m,k] one hot 3D tensor

    Note that in the returned 3D matrix, abstain votes continue to be
    represented by 0s, not 1s.

    Args:
        L: a [n,m] label matrix with categorical labels (0 = abstain)
        k: the number of classes that could appear in L
            if None, k is inferred as the max element in L
    """
    n, m = L.shape
    if k is None:
        k = L.max()
    L_onehot = torch.zeros(n, m, k + 1)
    for i, row in enumerate(L):
        for j, k in enumerate(row):
            if k > 0:
                L_onehot[i, j, k - 1] = 1
    return L_onehot


def recursive_merge_dicts(x, y, misses="report", verbose=None):
    """
    Merge dictionary y into a copy of x, overwriting elements of x when there
    is a conflict, except if the element is a dictionary, in which case recurse.

    misses: what to do if a key in y is not in x
        'insert'    -> set x[key] = value
        'exception' -> raise an exception
        'report'    -> report the name of the missing key
        'ignore'    -> do nothing
    verbose: If verbose is None, look for a value for verbose in y first, then x

    TODO: give example here (pull from tests)
    """

    def recurse(x, y, misses="report", verbose=1):
        found = True
        for k, v in y.items():
            found = False
            if k in x:
                found = True
                if isinstance(x[k], dict):
                    if not isinstance(v, dict):
                        msg = f"Attempted to overwrite dict {k} with " f"non-dict: {v}"
                        raise ValueError(msg)
                    # If v is {}, set x[k] = {} instead of recursing on empty dict
                    # Otherwise, recurse on the items in v
                    if v:
                        recurse(x[k], v, misses, verbose)
                    else:
                        x[k] = v
                else:
                    if x[k] == v:
                        msg = f"Reaffirming {k}={x[k]}"
                    else:
                        msg = f"Overwriting {k}={x[k]} to {k}={v}"
                        x[k] = v
                    if verbose > 1 and k != "verbose":
                        print(msg)
            else:
                for kx, vx in x.items():
                    if isinstance(vx, dict):
                        found = recurse(vx, {k: v}, misses="ignore", verbose=verbose)
                    if found:
                        break
            if not found:
                msg = f'Could not find kwarg "{k}" in destination dict.'
                if misses == "insert":
                    x[k] = v
                    if verbose > 1:
                        print(f"Added {k}={v} from second dict to first")
                elif misses == "exception":
                    raise ValueError(msg)
                elif misses == "report":
                    print(msg)
                else:
                    pass
        return found

    # If verbose is not provided, look for an value in y first, then x
    # (Do this because 'verbose' kwarg is often inside one or both of x and y)
    if verbose is None:
        verbose = y.get("verbose", x.get("verbose", 1))

    z = copy.deepcopy(x)
    recurse(z, y, misses, verbose)
    return z


def recursive_transform(x, test_func, transform):
    """Applies a transformation recursively to each member of a dictionary

    Args:
        x: a (possibly nested) dictionary
        test_func: a function that returns whether this element should be transformed
        transform: a function that transforms a value
    """
    for k, v in x.items():
        if test_func(v):
            x[k] = transform(v)
        if isinstance(v, dict):
            recursive_transform(v, test_func, transform)
    return x


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    def str2bool(string):
        if string == "0" or string.lower() == "false":
            return False
        elif string == "1" or string.lower() == "true":
            return True
        else:
            raise Exception(f"Invalid value {string} for boolean flag")

    for param in config_dict:
        # Blacklist certain config parameters from being added as flags
        if param in ["verbose"]:
            continue
        default = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, bool):
                parser.add_argument(f"--{param}", type=str2bool, default=default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                    )
                else:
                    parser.add_argument(f"--{param}", action="append", default=default)
            else:
                parser.add_argument(f"--{param}", type=OrNone(default), default=default)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


def split_data(
    *inputs,
    splits=[0.5, 0.5],
    shuffle=True,
    stratify_by=None,
    index_only=False,
    seed=None,
):
    """Splits inputs into multiple splits of defined sizes

    Args:
        inputs: correlated tuples/lists/arrays/matrices/tensors to split
        splits: list containing split sizes (fractions or counts);
        shuffle: if True, shuffle the data before splitting
        stratify_by: (None or an input) if not None, use these labels to
            stratify the splits (separating the data into groups by these
            labels and sampling from those, rather than from the population at
            large); overrides shuffle
        index_only: if True, return only the indices of the new splits, not the
            split data itself
        seed: (int) random seed

    Example usage:
        Ls, Xs, Ys = split_data(L, X, Y, splits=[0.8, 0.1, 0.1])
        OR
        assignments = split_data(Y, splits=[0.8, 0.1, 0.1], index_only=True)

    Note: This is very similar to scikit-learn's train_test_split() method,
        but with support for more than two splits.
    """

    def fractions_to_counts(fracs, n):
        """Converts a list of fractions to a list of counts that sum to n"""
        counts = [int(np.round(n * frac)) for frac in fracs]
        # Ensure sum of split counts sums to n
        counts[-1] = n - sum(counts[:-1])
        return counts

    def slice_data(data, indices):
        if isinstance(data, list) or isinstance(data, tuple):
            return [d for i, d in enumerate(data) if i in set(indices)]
        else:
            try:
                # Works for np.ndarray, scipy.sparse, torch.Tensor
                return data[indices]
            except TypeError:
                raise Exception(
                    f"split_data() currently only accepts inputs "
                    f"of type tuple, list, np.ndarray, scipy.sparse, or "
                    f"torch.Tensor; not {type(data)}"
                )

    # Setting random seed
    if seed is not None:
        random.seed(seed)

    try:
        n = len(inputs[0])
    except TypeError:
        n = inputs[0].shape[0]
    num_splits = len(splits)

    # Check splits for validity and convert to fractions
    if all(isinstance(x, int) for x in splits):
        if not sum(splits) == n:
            raise ValueError(
                f"Provided split counts must sum to n ({n}), not {sum(splits)}."
            )
        fracs = [count / n for count in splits]

    elif all(isinstance(x, float) for x in splits):
        if not sum(splits) == 1.0:
            raise ValueError(f"Split fractions must sum to 1.0, not {sum(splits)}.")
        fracs = splits

    else:
        raise ValueError("Splits must contain all ints or all floats.")

    # Make sampling pools
    if stratify_by is None:
        pools = [np.arange(n)]
    else:
        pools = defaultdict(list)
        for i, val in enumerate(stratify_by):
            pools[val].append(i)
        pools = list(pools.values())

    # Make index assignments
    assignments = [[] for _ in range(num_splits)]
    for pool in pools:
        if shuffle or stratify_by is not None:
            random.shuffle(pool)

        counts = fractions_to_counts(fracs, len(pool))
        counts.insert(0, 0)
        cum_counts = np.cumsum(counts)
        for i in range(num_splits):
            assignments[i].extend(pool[cum_counts[i] : cum_counts[i + 1]])

    if index_only:
        return assignments
    else:
        outputs = []
        for data in inputs:
            data_splits = []
            for split in range(num_splits):
                data_splits.append(slice_data(data, assignments[split]))
            outputs.append(data_splits)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


def padded_tensor(items, pad_idx=0, left_padded=False, max_len=None):
    """Create a padded [n, ?] Tensor from a potentially uneven iterable of Tensors.
    Modified from github.com/facebookresearch/ParlAI

    Args:
        items: (list) the items to merge and pad
        pad_idx: (int) the value to use for padding
        left_padded: (bool) if True, pad on the left instead of the right
        max_len: (int) if not None, the maximum allowable item length

    Returns:
        padded_tensor: (Tensor) the merged and padded tensor of items
    """
    # number of items
    n = len(items)
    # length of each item
    lens = [len(item) for item in items]
    # max seq_len dimension
    max_seq_len = max(lens) if max_len is None else max_len

    output = items[0].new_full((n, max_seq_len), pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if left_padded:
            # place at end
            output[i, max_seq_len - length :] = item
        else:
            # place at beginning
            output[i, :length] = item

    return output


# DEPRECATION: This is replaced by move_to_device
def place_on_gpu(data):
    """Utility to place data on GPU, where data could be a torch.Tensor, a tuple
    or list of Tensors, or a tuple or list of tuple or lists of Tensors"""
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_gpu(data[i]) for i in range(len(data))]
        data = data_type(data)
        return data
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return ValueError(f"Data type {type(data)} not recognized.")


def move_to_device(obj, device=-1):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).

    device = -1 -> "cpu"
    device =  0 -> "cuda:0"

    """
    if device < 0 or not torch.cuda.is_available():
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Is this necessary?
        torch.backends.cudnn.enabled = True  # type: ignore
        torch.cuda.manual_seed(seed)  # type: ignore
