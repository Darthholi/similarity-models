#
# COPYRIGHT Martin Holecek 2019
#

import math
import numbers
from datetime import datetime

import numpy as np
import os
import tempfile
from pyqtree import Index
from triarray import TriMatrix


def model_timestamped_name(basename):
    dtime = datetime.now()
    return basename + dtime.strftime("%Y-%m-%d-%H-%M-%S") + ".h5"


def get_weights_save_file(weights_best_save, basename):
    """
    as bsename it is ok to use for example: Path(__file__).stem
    If given None or ".", make it automatic timestamp.
    If given directory (or ends with a "/"), appends automatic timestamp
    If given filename, do not touch it.
    """
    if weights_best_save in [".", None]:
        return model_timestamped_name(basename)
    weights_best_save = os.path.expanduser(weights_best_save)  # in case of provided ~
    if weights_best_save[-1] in ['\\', '/']:
        return weights_best_save + model_timestamped_name(basename)
    return weights_best_save


def care_weights_save_file(weights_best_save, basename):
    fname = get_weights_save_file(weights_best_save, basename)
    try:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    except:
        from pathlib import Path
        Path(fname).touch()
    print("Saving weights to: {}".format(fname))
    return fname


def hash_generator(checksum, gen):
    for item in gen:
        checksum.update(str(item).encode())
    return checksum


def hash_numpy(checksum, arr):
    for item in arr:
        checksum.update(item.view(np.uint8))
    return checksum


class tempmap(np.memmap):
    def __new__(subtype, dtype=np.uint8, mode='w+', offset=0,
                shape=None, order='C'):
        ntf = tempfile.NamedTemporaryFile()
        self = np.memmap.__new__(subtype, ntf, dtype, mode, offset, shape, order)
        self.temp_file_obj = ntf
        return self
    
    def __del__(self):
        if hasattr(self, 'temp_file_obj') and self.temp_file_obj is not None:
            self.temp_file_obj.close()
            del self.temp_file_obj


class dynamic_memmap:
    """
    Since forking something with memmap opened makes memory usage grow like crazy, we need this interface
     to be opened when needed. (Or maybe not reopen but keep open on per process basis?)
    """
    
    def __init__(self, name=None, dtype=np.uint8, mode='w+', offset=0,
                 shape=None, order='C'):
        if name is None:
            name = self.get_new_name()
        self.name = name
        self.dtype = dtype
        self.mode = mode
        self.offset = offset
        self.shape = shape
        self.order = order
    
    def __len__(self):
        return self.shape[0]
    
    def __del__(self):
        if isinstance(self.name, tempfile._TemporaryFileWrapper):
            self.name.close()
        del self.name
    
    def get(self, mode=None):
        return np.memmap(self.name, self.dtype, self.mode if mode is None else mode, self.offset, self.shape,
                         self.order)
    
    @classmethod
    def get_new_name(cls):
        return tempfile.NamedTemporaryFile()


class OpenMemmap:
    def __init__(self, dmap):
        if not isinstance(dmap, dynamic_memmap):
            raise ValueError()
        self.dmap = dmap
    
    def __enter__(self):
        self._opened_tempmap = self.dmap.get('r')
        return self._opened_tempmap
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._opened_tempmap
        self._opened_tempmap = None


class dynamic_trimap(dynamic_memmap):
    def __init__(self, name=None, dtype=np.uint8, mode='w+', offset=0,
                 side_shape=None, order='C'):
        assert side_shape is not None
        total_size = int(side_shape * (side_shape - 1) / 2)
        dynamic_memmap.__init__(self, name, dtype, mode, offset, total_size, order)
        self.side_shape = side_shape
    
    def _iof(self, index):
        if isinstance(index[0], np.ndarray) or isinstance(index[1], np.ndarray):
            i = np.maximum(index[0][:], index[1][:]) - 1
            j = np.minimum(index[0][:], index[1][:])
            return (i * (i + 1) // 2) + j
        else:
            x, y = index
            if x < 0:
                x = self.side_shape + x
            if y < 0:
                y = self.side_shape + y
            if x == y:
                return None
            
            i = max(x, y) - 1
            j = min(x, y)
            return (i * (i + 1) // 2) + j


class OpenTriMemmapSet:
    def __init__(self, dmap, mode='w+'):
        if not isinstance(dmap, dynamic_trimap):
            raise ValueError()
        self.dmap = dmap
        self.mode = mode
    
    def __enter__(self):
        self._opened_tempmap = self.dmap.get(mode=self.mode)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._opened_tempmap
        self._opened_tempmap = None
    
    def __getitem__(self, item):
        assert self._opened_tempmap is not None
        return self._opened_tempmap[self.dmap._iof(item)]
    
    def __setitem__(self, item, val):
        assert self._opened_tempmap is not None
        self._opened_tempmap[self.dmap._iof(item)] = val


class OpenTriMemmapGet:
    def __init__(self, dmap):
        if not isinstance(dmap, dynamic_trimap):
            raise ValueError()
        self.dmap = dmap
    
    def __enter__(self):
        self._opened_tempmap = self.dmap.get('r')
        return TriMatrix(self._opened_tempmap, upper=False, diag_val=0)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._opened_tempmap
        self._opened_tempmap = None


def np_as_tmp_map(nparray):
    tmpmap = tempmap(dtype=nparray.dtype, mode='w+', shape=nparray.shape)
    tmpmap[...] = nparray
    return tmpmap


def is_np_array(val):
    """
    Checks whether a variable is a numpy array.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    # out : bool
        True if the variable is a numpy array. Otherwise False.

    """
    # using np.generic here seems to also fire for scalar numpy values even
    # though those are not arrays
    # return isinstance(val, (np.ndarray, np.generic))
    return isinstance(val, np.ndarray)


def is_single_integer(val):
    """
    Checks whether a variable is an integer.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is an integer. Otherwise False.

    """
    return not isinstance(val, bool) and isinstance(val, numbers.Integral)


def is_single_float(val):
    """
    Checks whether a variable is a float.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a float. Otherwise False.

    """
    return isinstance(val, numbers.Real) and not is_single_integer(val)


def is_single_number(val):
    """
    Checks whether a variable is a number, i.e. an integer or float.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a number. Otherwise False.

    """
    return is_single_integer(val) or is_single_float(val)


def is_iterable(val):
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is an iterable. Otherwise False.

    """
    # can be made to be more abstract, not just restricted
    return isinstance(val, (tuple, list, np.ndarray))


def multiclass_temporal_class_weights(targets, class_weights):
    s_weights = np.ones((targets.shape[0],))
    if class_weights is not None:
        for i in range(len(s_weights)):
            weight = 0.0
            for itarget, target in enumerate(targets[i]):
                weight += class_weights[itarget][int(round(target))]
            s_weights[i] = weight
    return s_weights


def np_samesize(arrays, arrays_defaults=0, axis=None):
    """
    All arrays will be padded to have the same size.
    If Axis is specified, concatenates the arrays along the axis, else returns list of the padded arrays.
    """
    dt_representant = arrays[0]
    if all([item.shape == dt_representant.shape for item in arrays]):
        if axis is not None:
            return np.concatenate(arrays, axis=axis)
        else:
            return arrays
    
    if is_single_number(arrays_defaults):
        arrays_defaults = [arrays_defaults] * len(arrays)
    
    assert len(arrays) == len(arrays_defaults), \
        "Provide either one default number or list of the same length as arryas"
    assert all([item.ndim == dt_representant.ndim for item in arrays]), "arrays need to be at least the same ndim"
    
    shape = [max([item.shape[i] for item in arrays])
             for i in range(dt_representant.ndim)]
    
    padded_arrays = [np.full(tuple(shape), arrays_defaults[k], dtype=array.dtype) for k, array in enumerate(arrays)]
    # Here we put all the data to the beginning and leave the padded at te end.
    # Another options are to put it at random position, end, or overflow...
    for i, item in enumerate(arrays):
        padded_arrays[i][tuple([slice(0, n) for n in item.shape])] = item
    
    if axis is not None:
        return np.concatenate(padded_arrays, axis=axis)
    else:
        return padded_arrays


def np_pad_to_size(arrays, minsizes=None, default=0, dtype=None):
    """
    All arrays will be padded to have the same size.
    default will get copied by its shape.
    """
    dt_representant = arrays[0]
    assert all([item.ndim == dt_representant.ndim for item in arrays]), "arrays need to be at least the same ndim"
    
    shape = [max([item.shape[i] for item in arrays])
             for i in range(dt_representant.ndim)]
    shape = [len(arrays)] + shape
    
    if minsizes is not None:
        # defined from the back!
        for i in range(min(len(minsizes), len(shape))):
            if minsizes[-1 - i] is not None:
                shape[-1 - i] = max(shape[-1 - i], minsizes[-1 - i])
    
    padded_array = np.full(tuple(shape), default, dtype=dtype if dtype is not None else dt_representant.dtype)
    # Here we put all the data to the beginning and leave the padded at te end.
    # Another options are to put it at random position, end, or overflow...
    for i, item in enumerate(arrays):
        padded_array[i][tuple([slice(0, n) for n in item.shape])] = item
    
    return padded_array


"""
def test_dynamic_tempmap():
    x = dynamic_tempmap(shape=(5, 6))
    with x as arr:
        arr[0, 0] = 5
        print(arr)
    print("done")
"""


def log_class_weights_from_counts(class_counts, mu=0.15):
    """
    Gets 1d numpy array or list, where the numbers are the class counts (and the indices are class IDS).
    Returns class weights calculated with using logarithms.
    """
    total = np.sum(class_counts)
    class_weight = [0.0] * len(class_counts)
    
    for i in range(len(class_counts)):
        if (class_counts[i] <= 0):
            score = 1.0
        else:
            score = math.log(mu * total / float(class_counts[i]))
        class_weight[i] = score if score > 1.0 else 1.0
    
    return class_weight


def class_weights_from_counts(class_counts):
    """
    Gets 1d numpy array or list, where the numbers are the class counts (and the indices are class IDS).
    Returns class weights calculated with using 1/.
    """
    total = np.sum(class_counts)
    class_weight = [0.0] * len(class_counts)
    
    for i in range(len(class_counts)):
        score = total / float(class_counts[i])
        class_weight[i] = score if score > 1.0 else 1.0
    
    return class_weight


def log_class_weights_from_counts_binary(class_counts, mu=0.15):
    """
    Gets 1d numpy array or list, where the numbers are the class counts (and the indices are class IDS).
    Returns class weights calculated with using logarithms.
    """
    class_weight = [0.0] * len(class_counts)
    
    for i, col in enumerate(class_counts):
        assert len(col) == 2  # so far we do it for binary classification
        total = np.sum(col)
        if col[1] <= 0:
            score = 1.0
        else:
            score = math.log(mu * total / float(col[1]))
        class_weight[i] = score if score > 0.0 else 1.0
    return class_weight


def class_weights_from_counts_binary(class_counts, norm_weights_to_one=True):
    """
    Gets 1d numpy array or list, where the numbers are the class counts (and the indices are class IDS).
    Returns class weights calculated with using 1/.
    """
    class_weight = [[]] * len(class_counts)
    
    for i, col in enumerate(class_counts):
        # assert len(col) == 2  # so far we do it for binary classification
        total = np.sum(col)
        scores = [total / float(sc) for sc in col]
        # class_weight[i] = [score if score > 0.0 else 1.0 for score in scores]
        if norm_weights_to_one:
            xwsum = np.sum(scores)
            scores = [score / xwsum for score in scores]
        class_weight[i] = scores
    
    return class_weight


def bb_center(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def ss_to_center(center, bbox):
    d = center - bb_center(bbox)
    return np.sum(d * d)


def fov_dilate(bbox, fov_dilation):
    return [bb + fov for bb, fov in zip(bbox, fov_dilation)]


def produce_fov_ids(lookers, viewed, count, fov_dilation=[-0.1, -0.1, 0.1, 0.1]):
    """
    Produce ids of bboxes each of lookers can 'see' (intersect) in the viewed wordbox.
    bbox = l, t, r, b
    """
    xs = [item[0] for item in viewed] + [item[2] for item in viewed]
    ys = [item[1] for item in viewed] + [item[3] for item in viewed]
    l = min(xs)
    t = min(ys)
    r = max(xs)
    b = max(ys)
    spindex = Index(bbox=(l, t, r, b))
    # this example assumes you have a list of items with bbox attribute
    for i, bbox in enumerate(viewed):
        spindex.insert(i, bbox)
    
    ids_r = np.full((len(lookers), count), -1)
    
    for i, bbox in enumerate(lookers):
        matches = spindex.intersect(fov_dilate(bbox, fov_dilation))
        # now find the closest to the center of the original:
        center = bb_center(bbox)
        matches.sort(key=lambda j: ss_to_center(center, viewed[j]))  # ascending
        cnt = min(len(matches), count)
        ids_r[i, :cnt] = matches[:cnt]
    
    return ids_r


def equal_ifarray(a, b):
    if isinstance(a, np.ndarray):
        return all(a == b)
    else:
        return a == b


def project_fun_bbox(fun, a, b, dim):
    return fun((a[dim], a[dim + 2]), (b[dim], b[dim + 2]))


def analyze_bboxes_positions(a, b):
    """
    ...from a to b
    what do we want:
    - decide which edge sees the bbox (remember it for later)
      - lets define 2 closest edges to the bbox and take the edge, which contributes the MOST to the distance function.
    - to filter the bboxes according to seeing criteria:
        - nonobscuring first (lets not take obscuring into account and only order things based on their distance)
            - a) sees only in orthogonal directions (maybe even smaller/larger than the bbox is)
            - b) sees everything, pass
    - then order based on bbox (-eucleidian) distance all bboxes seen be each edge and take the top n closest.
    """
    x_dim_prop = project_fun_bbox(range_distance_side, a, b, 0)
    y_dim_prop = project_fun_bbox(range_distance_side, a, b, 1)
    
    # def - lets take the bigger distance from x and y projection and define, that
    # the edge to which it belongs is the orthogonal to that dimension (x/y) it will be THAT edge to which it belongs
    is_x_dist_bigger = x_dim_prop[0] > y_dim_prop[0]
    the_order = x_dim_prop[1] if is_x_dist_bigger else y_dim_prop[1]
    
    side_map = {(True, False): 0, (True, True): 2,
                # (in x dimension?, was the order of a,b retained (true) or switched?)
                (False, False): 1, (False, True): 3, }
    edge_id = side_map[(is_x_dist_bigger, the_order)]
    return x_dim_prop, y_dim_prop, edge_id


def range_distance_ordered(a_min, a_max, b_min, b_max):
    # returns minus when they do overlap.
    assert a_min <= a_max and b_min <= b_max
    assert a_min <= b_min
    return b_min - min(a_max, b_max)


def range_distance_side(a, b):
    # return range distance and the information on positions of the two ranges
    #  (if they are in the original order or not)
    if a[0] <= b[0]:
        order = [a, b]
        smaller_to_bigger = True
    else:
        order = [b, a]
        smaller_to_bigger = False
    return range_distance_ordered(order[0][0], order[0][1], order[1][0], order[1][0]), smaller_to_bigger
