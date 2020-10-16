#
# COPYRIGHT Martin Holecek 2019
#

import numpy as np
import random
import tensorflow as tf
from keras import backend as K
from tensorpack import BatchData, RNGDataFlow

from utils.manipulations_utils import np_pad_to_size


class BatchDataTflike(BatchData):
    """
    Stack datapoints (dicts) into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a numpy array of original components.

    Allows to use dictionaries and not only lists.
    Theoretically could be a child of tensorpack's BatchData dataflow, but that class is not easily subclassable.
    """
    
    def __init__(self, ds, batch_size,
                 output_types,
                 remainder=False, use_list=False,
                 allow_listify_specials=False,
                 ):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `ds.size()` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
            defaults_for_making_same_size (dict):
                if a key in this dict is provided, then if it happens, that the data to be batched (under this key) are
                of different sizes, then do not throw error, but pad with this default to match higher dimensions
            allow_listify_specials (bool):
                In case data cannot be batched and error is encountered, then instead of stopping, would just
                 batch the data into a list.
            save_orig_sizes_postfix (str):
                If using dictionaries to pass data, then if this string is provided, will save the original's lengths
                when padding was needed to be able to recover them later.
        """
        if not remainder:
            try:
                assert batch_size <= ds.size(), "batch size {} must be lesser than dataset size {}". \
                    format(batch_size, ds.size())
            except NotImplementedError:
                pass
        super(BatchDataTflike, self).__init__(ds, batch_size, remainder, use_list)
        
        self.allow_listify_specials = allow_listify_specials
        self.output_types = output_types
    
    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch_from_list(holder)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch_from_list(holder)
    
    """
    @classmethod
    def _expand_batch(cls, data, save_orig_sizes_postfix=None, pred_token="pred_"):
        
        When using the same dataflow for training and prediction, this method should help to unbatch the data, if
        save_orig_sizes_postfix is provided (and save_orig_sizes_postfix was prodvided on batching).
        When pred_token is provided, allows sharing size information for predictions together with goldstandard targets.
        
        batch_size = len(data[list(data.keys())[0]])  # if it is a numpy array, it should give us the leftmost dimension
        # or if it is just a list then the length of the list...
        for b in range(batch_size):
            expanded = {k: data[k][b] for k in data.keys()}
            if save_orig_sizes_postfix:
                for key in expanded.keys():
                    if key.startswith(pred_token):
                        saved_size_key = key[len(pred_token):] + save_orig_sizes_postfix
                    else:
                        saved_size_key = key + save_orig_sizes_postfix

                    if saved_size_key in data:
                        orig_shape = data[saved_size_key][b]
                        slicer = [slice(0, orig_dim_size) for orig_dim_size in orig_shape]
                        expanded[key] = expanded[key][slicer]
                        assert expanded[key].shape == orig_shape
                        # del expanded[saved_size_key]
                        # we have already used it so we can del it to not confuse anybody
                        #  - not true, because we use it for predictions too
            yield expanded
    """
    
    @classmethod
    def _get_keys(cls, item, prefix=None):
        if prefix is None:
            prefix = []
        allkeys = []
        if isinstance(item, list) or isinstance(item, tuple):
            for key, value in enumerate(item):
                allkeys.extend(cls._get_keys(value, prefix + [key]))
        elif isinstance(item, dict):
            for key in item:
                allkeys.extend(cls._get_keys(item[key], prefix + [key]))
        else:
            allkeys = [prefix]
        return allkeys
    
    @classmethod
    def _copy_struct(cls, item):
        # if prefix is None:
        #    prefix = []
        # allkeys = []
        if isinstance(item, list) or isinstance(item, tuple):
            struct = [None] * len(item)
            for key, value in enumerate(item):
                struct[key] = cls._copy_struct(value)
        elif isinstance(item, dict):
            struct = {}
            for key in item:
                struct[key] = cls._copy_struct(item[key])
        else:
            struct = None
        return struct
    
    @classmethod
    def _get_indexed(cls, x, indexed, ignore_error=False, ignore_error_return_val=None):
        ptr = x
        for i in indexed:
            try:
                ptr = ptr[i]
            except Exception as e:
                if ignore_error:
                    return ignore_error_return_val
                else:
                    raise e
        return ptr
    
    @classmethod
    def _set_indexed(cls, x, indexed, data):
        assert len(indexed) >= 1, "the type is not nested, cannot set property"
        ptr = x
        for i in indexed[:-1]:
            ptr = ptr[i]
        ptr[indexed[-1]] = data
    
    @classmethod
    def tftype_to_np(cls, t):
        if t == tf.int32:
            return np.int32
        elif t == tf.float32:
            return np.float32
        elif t == object:
            return np.object
        elif t == tf.string:
            return np.object
        elif t == tf.resource:
            return np.object
        elif t == list:
            return list
        raise ValueError("expand this small method ;)")
    
    def _aggregate_batch_from_list(self, data_holder):
        """
        Aggregates data into a batch, datapoints are made to be the same size by padding.
        If save_orig_sizes_postfix is provided, will save information about original sizes.

        Should replace tensorpack's dataflow.BatchData._aggregate_batch, but that one is staticmethod,
         so cannot be derived.
        """
        # data_representant = data_holder[0]  # first row of data from batch
        # assert isinstance(data_representant, dict), "BatchDataDict works only on dictionaries"
        batch = self._copy_struct(self.output_types)
        
        for indexed in self._get_keys(batch):  # for all dataparts
            batched_list = [self._get_indexed(x, indexed) for x in data_holder]  # ignore all other
            if self.use_list:
                self._set_indexed(batch, indexed, batched_list)
            else:
                self._set_indexed(batch, indexed,
                                  self._aggregate_batch_smart(batched_list, indexed,
                                                              self.tftype_to_np(
                                                                  self._get_indexed(self.output_types, indexed))))
        return batch
    
    def _aggregate_batch_smart(self, batched_list, indexed, dtype):
        dt_representant = batched_list[0]
        if dtype == list:
            return batched_list
        """
        if type(dt_representant) in list(six.integer_types) + [bool]:
            suggest_type = 'int32'
        elif type(dt_representant) == float:
            suggest_type = 'float32'
        else:
            try:
                suggest_type = dt_representant.dtype
            except AttributeError:
                if self.allow_listify_specials:
                    return batched_list
                else:
                    raise TypeError("Unsupported type to batch: {} from data part indexed by '{}',"
                                    " you can set 'allow_listify_specials'"
                                    "if this dataflow is here only for prediction and that field is here only "
                                    "to remember indices".format(type(dt_representant), k))
        """
        
        if isinstance(dt_representant, np.ndarray) and not all(
                [item.ndim == dt_representant.ndim for item in batched_list]):
            raise ValueError('Data part ({}) of inconsistent number of dimensions,'
                             ' cannot batch even with defaults'.format(indexed))
        
        return self._aggregate_batch_smart_inner(batched_list, indexed, dtype)
    
    def _aggregate_batch_smart_inner(self, batched_list, indexed, dtype):
        dt_representant = batched_list[0]
        try:
            return np.asarray(batched_list, dtype=dtype)
        except Exception as e:  # noqa
            if self.allow_listify_specials:
                return batched_list
            print("Cannot batch data. Perhaps they are of inconsistent shape?")
            # tensorpack_logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
            # if isinstance(dt_representant, np.ndarray):
            #    s = pprint.pformat(batched_list)
            #    tensorpack_logger.error("Shape of all arrays to be batched: " + s)


class BatchDataPadder(BatchDataTflike):
    """
    Makes data in each batch padded to maximal dimensions using provided default pad values.
    Also has the option/procedure to unbatch them later (for predict/eval) if is allowed to save the lengths.
    """
    
    def __init__(self, ds, batch_size,
                 output_types,
                 padded_shapes=None,
                 padding_values=0,
                 remainder=False, use_list=False,
                 allow_listify_specials=False,
                 # save_orig_sizes_postfix="_sizes_orig"
                 ):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `ds.size()` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
            defaults_for_making_same_size (dict):
                if a key in this dict is provided, then if it happens, that the data to be batched (under this key) are
                of different sizes, then do not throw error, but pad with this default to match higher dimensions
            allow_listify_specials (bool):
                In case data cannot be batched and error is encountered, then instead of stopping, would just
                 batch the data into a list.
            save_orig_sizes_postfix (str):
                If using dictionaries to pass data, then if this string is provided, will save the original's lengths
                when padding was needed to be able to recover them later.
        """
        super(BatchDataPadder, self).__init__(ds, batch_size, output_types, remainder, use_list,
                                              allow_listify_specials)
        
        # if defaults_for_making_same_size is not None:
        #    self.defaults_for_making_same_size = defaults_for_making_same_size
        # else:
        #    self.defaults_for_making_same_size = {}
        
        # self.save_orig_sizes_postfix = save_orig_sizes_postfix
        
        self.padded_shapes = padded_shapes
        self.padding_values = padding_values
    
    def _aggregate_batch_smart_inner(self, batched_list, indexed, dtype):
        
        for i in range(len(batched_list)):
            if not isinstance(batched_list[i], np.ndarray):
                batched_list[i] = np.array(batched_list[i])
        
        # dt_representant = batched_list[0]
        # if not k in self.defaults_for_making_same_size:
        #    raise ValueError('Data part ({}) of inconsistent shape, cannot batch'.format(k))
        
        default = self._get_indexed(self.padding_values, indexed, True, 0)
        return np_pad_to_size(batched_list, minsizes=self._get_indexed(self.padded_shapes, indexed, True, None),
                              default=default,
                              dtype=dtype)
        
        # if self.save_orig_sizes_postfix:
        #    batch[k + self.save_orig_sizes_postfix] = [item.shape for item in batched_list]
    
    """
    @classmethod
    def _expand_batch(cls, data, save_orig_sizes_postfix=None, pred_token="pred_"):
        
        When using the same dataflow for training and prediction, this method should help to unbatch the data, if
        save_orig_sizes_postfix is provided (and save_orig_sizes_postfix was prodvided on batching).
        When pred_token is provided, allows sharing size information for predictions together with goldstandard targets.
        
        batch_size = len(data[list(data.keys())[0]])  # if it is a numpy array, it should give us the leftmost dimension
        # or if it is just a list then the length of the list...
        for b in range(batch_size):
            expanded = {k: data[k][b] for k in data.keys()}
            if save_orig_sizes_postfix:
                for key in expanded.keys():
                    if key.startswith(pred_token):
                        saved_size_key = key[len(pred_token):] + save_orig_sizes_postfix
                    else:
                        saved_size_key = key + save_orig_sizes_postfix

                    if saved_size_key in data:
                        orig_shape = data[saved_size_key][b]
                        slicer = [slice(0, orig_dim_size) for orig_dim_size in orig_shape]
                        expanded[key] = expanded[key][slicer]
                        assert expanded[key].shape == orig_shape
            yield expanded
    """


def tf_dataset_as_iterator(ds, sess=None):
    if sess is None:
        sess = K.get_session()
    # most basic way to use tf dataset. Not optimal.
    iterator = ds.make_one_shot_iterator()
    
    iternext = iterator.get_next()
    # or    with tf.Session() as sess:
    while True:
        ret = sess.run(iternext)
        yield ret


class RandomOrderSequence(RNGDataFlow):
    def __init__(self, set_len, shuffle=True):
        """

        shuffle (bool): shuffle data.
        (settype = train or val)

        """
        super(RandomOrderSequence, self).__init__()
        self.shuffle = shuffle
        self.set_len = set_len
    
    def __len__(self):
        return self.set_len
    
    def __iter__(self):
        if not self.shuffle:
            for i in range(self.set_len):
                yield i
        else:
            shuffle_seed = random.random()
            idxs = list(range(self.set_len))
            random.Random(shuffle_seed).shuffle(idxs)
            for i in idxs:
                yield i


class RandomPhaseSequence(RNGDataFlow):
    def __init__(self, set_len, shuffle=True):
        """
        
        shuffle (bool): shuffle data.
        (settype = train or val)
        
        """
        super(RandomPhaseSequence, self).__init__()
        self.shuffle = shuffle
        self.set_len = set_len
    
    def __len__(self):
        return self.set_len
    
    def __iter__(self):
        if not self.shuffle:
            for i in range(self.set_len):
                yield None, i
        else:
            shuffle_seed = random.random()
            for i in range(self.set_len):
                yield shuffle_seed, i
