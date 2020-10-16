#
# COPYRIGHT Martin Holecek 2019
#

import datetime
from collections import defaultdict

import numpy as np
import six
import warnings
from keras import backend as K
from keras.callbacks import Callback
from sklearn import metrics as skmetrics
from tqdm import tqdm

from utils import produce_drawings
from utils.manipulations_utils import tempmap, np_as_tmp_map
from utils.sqlite_experiment_generator import DocsTextsSqliteNearest, FtypesTrgtDocsTextsSqliteNearest


class EvaluateFCallback(Callback):
    """
    Calls evaluate function each time a key metric improves (or after each epoch if key metric not provided).
    """
    
    def __init__(self, evaluation_f, monitor=None, mode=None, min_delta=0):
        super(EvaluateFCallback, self).__init__()
        self.evaluation_f = evaluation_f
        
        # copied from keras earlystopping:
        self.min_delta = min_delta
        self.monitor = monitor
        if mode not in ['auto', 'min', 'max', None]:
            warnings.warn('EvaluateFCallback mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'
        
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:  # auto:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = None
        run_e = False
        if self.monitor is not None and logs is not None:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn(
                    'EvaluateFCallback evaluating on on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
                )
                return
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                run_e = True
        else:
            run_e = True
        if run_e:  # either we do not use the 'eval on best metric' mechanism or we do:
            before = datetime.datetime.now()
            self.evaluation_f()
            after = datetime.datetime.now()
            print("EvaluateFCallback calculated evaluations in {} seconds".format((after - before).total_seconds()))


class EvaluateFCallbackToLogs(Callback):
    """
    Calls evaluate function each time a key metric improves (or after each epoch if key metric not provided).
    """
    
    def __init__(self, evaluation_f, monitor_save=None):
        super(EvaluateFCallbackToLogs, self).__init__()
        self.evaluation_f = evaluation_f
        self.monitor_save = monitor_save
    
    def on_epoch_end(self, epoch, logs=None):
        before = datetime.datetime.now()
        result = self.evaluation_f()
        after = datetime.datetime.now()
        print("EvaluateFCallback calculated evaluations in {} seconds".format((after - before).total_seconds()))
        
        logs[self.monitor_save] = result


"""
def weighted_masked_objective(fn): training utils py

how keras does it:
        if weights is not None:
            # reduce score_array to same ndim as weight array
            ndim = K.ndim(score_array)
            weight_ndim = K.ndim(weights)
            score_array = K.mean(score_array,
                                 axis=list(range(weight_ndim, ndim)))
            score_array *= weights
            score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return K.mean(score_array)
        
what we wanted:
                def multiclass_temporal_class_weights(targets, class_weights):
                    s_weights = np.ones((targets.shape[0],))
                    if class_weights is not None:
                        for i in range(len(s_weights)):
                            weight = 0.0
                            for itarget, target in enumerate(targets[i]):
                                weight += class_weights[itarget][int(round(target))]
                            s_weights[i] = weight
                    return s_weights
"""


def lastaxissum(x):
    return K.sum(x, axis=-1)


def matrix_weighted_target_loss(y_true, y_pred,
                                index=0,
                                weights_index=1,
                                fn=K.binary_crossentropy,
                                w_fn=lastaxissum  # sum over the cls_extract_types, not over the time axis
                                ):
    y_t = y_true[..., index]
    y_p = y_pred[..., index]
    score_array = fn(y_t, y_p)
    weights = w_fn(y_true[..., weights_index])
    
    # reduce score_array to same ndim as weight array:
    ndim = K.ndim(score_array)
    weight_ndim = K.ndim(weights)
    score_array = K.mean(score_array,
                         axis=list(range(weight_ndim, ndim)))
    score_array *= weights
    score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    return K.mean(score_array)


def matrix_weighted_target_bce_tempsum(y_true, y_pred):
    return matrix_weighted_target_loss(y_true, y_pred,
                                       index=0,
                                       weights_index=1,
                                       fn=K.binary_crossentropy,
                                       w_fn=lastaxissum  # sum over the cls_extract_types, not over the time axis
                                       )


def binary_crossentropy_with_weights_matrix(y_true, y_pred):
    return K.mean(
        K.binary_crossentropy(y_true[..., 0], y_pred[..., 0]) * y_true[..., 1],
        axis=-1)


def distances_and_classified_with_weights(y_true, y_pred):
    """
    y_pred: classified, distances
    y_true: true wordboxes classified, weights
    """
    class_crossentropy = K.sum(
        K.binary_crossentropy(y_true[..., 0], y_pred[..., 0]) * y_true[..., 1])
    
    # triplet loss inspirations:
    # https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    # https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973
    
    # classical triplet loss:
    # basic_loss = pos_dist - neg_dist + alpha
    # loss = K.maximum(basic_loss, 0.0)
    positive_distances = y_true[..., 0] * y_pred[..., 1]
    negative_distances = (1.0 - y_true[..., 0]) * y_pred[..., 1]
    positive_distances_weighted = positive_distances * y_true[..., 1]
    negative_distances_weighted = negative_distances * y_true[..., 1]
    
    tripletlike_base = 0.4 + K.sum(positive_distances_weighted) - K.sum(negative_distances_weighted)
    # tripletlike_loss = K.maximum(tripletlike_loss, 0) ... we have unbalanced numbers of positive and negative classes
    # so lets not use the maximum with zero.
    tripletlike_loss = K.switch(K.greater(tripletlike_base, 0.0), tripletlike_base, 0.0001 * tripletlike_base)
    
    return tripletlike_loss + class_crossentropy


def distances_and_classified_with_weights_v2(y_true, y_pred):
    """
    y_pred: classified, distances
    y_true: true wordboxes classified, weights
    """
    class_crossentropy = K.sum(
        K.binary_crossentropy(y_true[..., 0], y_pred[..., 0]) * y_true[..., 1])
    
    # triplet loss inspirations:
    # https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    # https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973
    
    # classical triplet loss:
    # basic_loss = pos_dist - neg_dist + alpha
    # loss = K.maximum(basic_loss, 0.0)
    positive_distances = y_true[..., 0] * y_pred[..., 1]
    negative_distances = (1.0 - y_true[..., 0]) * y_pred[..., 1]
    positive_distances_weighted = positive_distances * y_true[..., 1]
    negative_distances_weighted = negative_distances * y_true[..., 1]
    
    tripletlike_base = 0.4 + K.max(positive_distances) - K.min(negative_distances)
    tripletlike_loss = K.maximum(tripletlike_base, 0)  # ... we have unbalanced numbers of positive and negative classes
    # so  we took the max positive that must be lesser than the minimal negative
    # tripletlike_loss = K.switch(K.greater(tripletlike_base, 0.0), tripletlike_base, 0.0001 * tripletlike_base)
    
    return tripletlike_loss + class_crossentropy


def distances_and_classified_with_weights_and_mean(y_true, y_pred):
    """
    NEEDS SIGMOIDAL OUTPUT LAYER!
    
    y_pred: classified, distances
    y_true: true lwordboxes classified, weights
    """
    class_crossentropy = K.sum(
        K.binary_crossentropy(y_true[..., 0], y_pred[..., 0]) * y_true[..., 1])
    
    # triplet loss inspirations:
    # https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    # https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973
    
    real_samples = 1.0 - K.cast(K.equal(y_true[..., 1], 0.0), 'float32')
    
    # classical triplet loss:
    # basic_loss = pos_dist - neg_dist + alpha
    # loss = K.maximum(basic_loss, 0.0)
    positive_distances = y_true[..., 0] * y_pred[..., 1]
    negative_distances = (1.0 - y_true[..., 0]) * y_pred[..., 1]
    positive_distances_real = positive_distances * real_samples
    negative_distances_real_for_min = negative_distances * real_samples + (1.0 - real_samples) * 100000.0
    negative_distances_real = negative_distances * real_samples
    
    tripletlike_base = 0.4 + K.max(positive_distances_real) - K.min(negative_distances_real_for_min)
    tripletlike_loss = K.maximum(tripletlike_base, 0)  # ... we have unbalanced numbers of positive and negative classes
    # so  we took the max positive that must be lesser than the minimal negative
    # tripletlike_loss = K.switch(K.greater(tripletlike_base, 0.0), tripletlike_base, 0.0001 * tripletlike_base)
    
    real_count = K.maximum(K.sum(real_samples), 1)
    unbound_means = K.sum(positive_distances_real) / real_count - K.sum(negative_distances_real) / real_count
    
    return tripletlike_loss + class_crossentropy + unbound_means


def real_sampled_mean(item, weights):
    real_samples = 1.0 - K.cast(K.equal(weights, 0.0), 'float32')
    divisor = K.sum(real_samples)
    return K.switch(K.equal(divisor, 0.0), 0.0, K.sum(item * real_samples) / divisor)
    # careful, can return zeroes when divisor ==0...


def binary_accuracy_with_weights(y_true, y_pred):
    return real_sampled_mean(K.cast(K.equal(y_true[..., 0], K.round(y_pred[..., 0])), 'float32'),
                             y_true[..., 1])


def positive_samples_with_weights(y_true, y_pred):
    return real_sampled_mean(y_true[..., 0],
                             y_true[..., 1])


def positive_samples_with_weights_predicted(y_true, y_pred):
    return real_sampled_mean(K.round(y_pred[..., 0]),
                             y_true[..., 1])


def binary_accuracy_positive_with_weights(y_true, y_pred):
    # we do not use real_sampled_mean, because when y_true[..., 0] == 1.0, then y_true[..., 1] is never == 0
    pos = y_true[..., 0]
    return K.sum(
        K.cast(K.equal(y_true[..., 0], K.round(y_pred[..., 0])), K.floatx()) * pos
    ) / K.maximum(K.sum(pos), 0.00001)


def recall(good, bad, miss):
    try:
        return good / (good + bad + miss)
    except ZeroDivisionError:
        return 0.0


def precision(good, bad, extra):
    try:
        return good / (good + bad + extra)
    except ZeroDivisionError:
        return 0.0


def f1(precision, recall):
    try:
        return (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return 0.0


def T_prec(mat):
    ((tn, fn), (fp, tp)) = mat
    not_missing = tp + fp
    return tp / not_missing if not_missing > 0 else 0.0


def T_rec(mat):
    ((tn, fn), (fp, tp)) = mat
    not_extra = tp + fn
    return tp / not_extra if not_extra > 0 else 0.0


def T_f1(mat):
    denom = T_prec(mat) + T_rec(mat)
    if not denom:
        return 0.0
    else:
        return 2 * T_prec(mat) * T_rec(mat) / denom


def GWME_f1(dct):
    return f1(precision(dct['good'], dct['wrong'], dct['extra']),
              recall(dct['good'], dct['wrong'], dct['miss']))


def GWME_sum(dcts):
    result = defaultdict(lambda: 0)
    for dct_name, dct in dcts.items():
        for key in dct:
            result[key] += dct[key]
    return result


def evaluating_f_reuse(eval_ds, eval_size, model, verbose_progress=True, plots_prefix=None):
    ft_per_wb = defaultdict(lambda: np.zeros((2, 2), dtype=int))  # predicted, real cls_extract_type per wordbox
    ft_satisfication = defaultdict(lambda: [0, 0, 0])
    # success (needed & provided), miss (needed & not provided), extra (not needed & provided) ... we do not care about extras btw
    
    for istep in tqdm(range(eval_size), total=eval_size, disable=not verbose_progress):
        batch = six.next(eval_ds)
        # batch structire: [0] - x, [1]: concatenated y and weights
        annotations = {item: batch[0][item] for item in DocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        x = {item: batch[0][item] for item in batch[0] if item not in DocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        predicted_data = model.predict_on_batch(x)
        for b, (wb_poso, pred, truth, annotation, nearest_annotation, nearest_cls_extract_type_to_ordered_ids) in enumerate(
                zip(x['wb-poso'], predicted_data, batch[1], annotations['annotations'],
                    annotations['nearest-annotations'],
                    annotations['nearest-cls_extract_type-to-ordered-ids'])):
            # first remove batched-padded items:
            nearest_count = max(sum(nearest_cls_extract_type_to_ordered_ids.values(), [])) + 1
            this_count = truth.shape[0]
            for i in range(truth.shape[0]):
                if truth[i, 0, 1] == 0:
                    # as soon as we hit the padded 0-weight, we know that that is the real length of the array of wordboxes
                    this_count = i
            truth = truth[:this_count, :nearest_count, ...]
            # btw might be cut as a single piece from a much longer page, thats why some annotations might be missing!
            pred = pred[:this_count, :nearest_count, ...]
            wb_poso = wb_poso[:this_count, :]
            voted_cls_extract_type = are_wordboxes_in_cls_extract_type(pred, nearest_cls_extract_type_to_ordered_ids)
            truth_cls_extract_type = are_wordboxes_in_cls_extract_type(truth, nearest_cls_extract_type_to_ordered_ids)

            # the same in different format:
            truth = np.zeros((len(wb_poso), len(annotation['cls_extract_type'])))
            pred = np.zeros((len(wb_poso), len(annotation['cls_extract_type'])))
            for ft_bb_ids, (ft_i, ft) in zip(annotation['ids'], enumerate(annotation['cls_extract_type'])):
                truth[ft_bb_ids, ft_i] = 1.0
                if ft in voted_cls_extract_type:
                    pred[voted_cls_extract_type[ft], ft_i] = 1.0
                
            produce_drawings(istep, b, annotation['cls_extract_type'], truth, pred, x['wb-bbox'][b],
                             x['nearest-annotated'][b], x['nearest-wb-bbox'][b], plots_prefix)
            # per wordbox stats:
            for ft in truth_cls_extract_type.keys():
                tp = np.sum(voted_cls_extract_type[ft] & truth_cls_extract_type[ft])
                fp = np.sum(voted_cls_extract_type[ft] & ~truth_cls_extract_type[ft])
                fn = np.sum(~voted_cls_extract_type[ft] & truth_cls_extract_type[ft])
                tn = np.sum(~voted_cls_extract_type[ft] & ~truth_cls_extract_type[ft])
                ft_per_wb[ft][1, 1] += tp
                ft_per_wb[ft][0, 0] += tn
                ft_per_wb[ft][1, 0] += fp
                ft_per_wb[ft][0, 1] += fn
            
            # embedding - capacity stats:
            needed_fts = {anot for anot, ids in zip(annotation['cls_extract_type'], annotation['ids']) if len(ids) > 0}
            provided_fts = {anot for anot, ids in zip(nearest_annotation['cls_extract_type'], nearest_annotation['ids']) if
                            len(ids) > 0}
            
            for ft in needed_fts.union(provided_fts):
                isneed = ft in needed_fts
                isprov = ft in provided_fts
                if isneed and isprov:
                    ft_satisfication[ft][0] += 1
                elif isneed and not isprov:
                    ft_satisfication[ft][1] += 1
                elif not isneed and isprov:
                    ft_satisfication[ft][2] += 1
    
    print("micro nonbg f1:")
    for ft in ft_per_wb:
        print("{}: {}".format(ft, T_f1(ft_per_wb[ft])))
    print("total micro conf matrix:")
    print(sum([ft_per_wb[ft] for ft in ft_per_wb]))
    print("total micro nonbg f1:")
    totalmicrof1 = T_f1(sum([ft_per_wb[ft] for ft in ft_per_wb]))
    print("... {}".format(totalmicrof1))
    return totalmicrof1


def are_wordboxes_in_cls_extract_type(pred, nearest_cls_extract_type_to_ordered_ids, method=np.mean,  # or np.max
                               ):
    # now lets vectorize - the votes for each wordbox to be what cls_extract_type.
    # there can be more metrics included, lets try to count the maximal prediction for now.
    votes_per_cls_extract_type = {ft: method(pred[:, nearest_cls_extract_type_to_ordered_ids[ft], 0], axis=-1)
                            if len(nearest_cls_extract_type_to_ordered_ids[ft]) > 0 else 0
                           for ft in nearest_cls_extract_type_to_ordered_ids}
    # now lets find what the network predicts - anywhere the prediction is greater than 0.5
    # (that threshold is after a trained layer), it is that class to predict. Remember it can be multiple classes, no argmax
    voted_cls_extract_type = {ft: votes_per_cls_extract_type[ft] >= 0.5 for ft in votes_per_cls_extract_type}
    return voted_cls_extract_type


def binary_crossentropy_with_weights_mt(y_true, y_pred, bin_size):
    # assert lengths are the same in bin_size:-1 == 0: bin_size
    return K.mean(K.mean(
        K.binary_crossentropy(y_true[..., 0:bin_size], y_pred[..., 0:bin_size]) * y_true[..., bin_size:],
        axis=-1), axis=-1)


def binary_accuracy_with_weights_mt(y_true, y_pred, bin_size):
    return real_sampled_mean(K.cast(K.equal(y_true[..., 0:bin_size],
                                            K.round(y_pred[..., 0:bin_size])), 'float32'),
                             y_true[..., bin_size:])


def positive_samples_with_weights_mt(y_true, y_pred, bin_size):
    return real_sampled_mean(y_true[..., 0:bin_size],
                             y_true[..., bin_size:])


def positive_samples_with_weights_predicted_mt(y_true, y_pred, bin_size):
    return real_sampled_mean(K.round(y_pred[..., 0:bin_size]),
                             y_true[..., bin_size:])


def binary_accuracy_positive_with_weights_mt(y_true, y_pred, bin_size):
    # we do not use real_sampled_mean, because when y_true[..., 0] == 1.0, then y_true[..., 1] is never == 0
    pos = y_true[..., 0:bin_size]
    return K.sum(
        K.cast(K.equal(y_true[..., 0:bin_size], K.round(y_pred[..., 0:bin_size])), K.floatx()) * pos
    ) / K.maximum(K.sum(pos), 0.00001)


def recall_multitarget(bin_size):
    # also called recall
    def recall(t, p):
        return binary_accuracy_positive_with_weights_mt(t, p, bin_size)
    
    return recall


def accuracy_binary_multitarget(bin_size):
    def accuracy(t, p):
        return binary_accuracy_with_weights_mt(t, p, bin_size)
    
    return accuracy


def num_predicted_ones_binary_multitarget(bin_size):
    def cnt_pred_1s(t, p):
        return positive_samples_with_weights_predicted_mt(t, p, bin_size)
    
    return cnt_pred_1s


def num_truth_ones_binary_multitarget(bin_size):
    def cnt_true_1s(t, p):
        return positive_samples_with_weights_mt(t, p, bin_size)
    
    return cnt_true_1s


def evaluating_ftypes_targets(eval_ds, eval_size, model, cls_extract_types, verbose_progress=True):
    ft_per_wb = defaultdict(lambda: np.zeros((2, 2), dtype=int))  # predicted, real cls_extract_type per wordbox
    ft_satisfication = defaultdict(lambda: [0, 0, 0])
    # success (needed & provided), miss (needed & not provided), extra (not needed & provided) ... we do not care
    # about extras btw
    
    for i in tqdm(range(eval_size), total=eval_size, disable=not verbose_progress):
        batch = six.next(eval_ds)
        # batch structire: [0] - x, [1]: concatenated y and weights
        annotations = {item: batch[0][item] for item in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS
                       if item in batch[0]}
        x = {item: batch[0][item] for item in batch[0] if
             item not in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        predicted_data = model.predict_on_batch(x)
        for b, (wb_poso, pred, truth, annotation) in enumerate(
                zip(x['wb-poso'], predicted_data, batch[1], annotations['annotations'])):
            # first remove batched-padded items:
            this_count = truth.shape[0]
            for i in range(truth.shape[0]):
                if truth[i, -1] == 0:  # the last channel is surely a weight
                    # as soon as we hit the padded 0-weight, we know that that is the real length of
                    # the array of wordboxes
                    this_count = i
            truth = truth[:this_count, ...]
            # btw might be cut as a single piece from a much longer page, thats why some annotations
            # might be missing!
            pred = pred[:this_count, ...]
            wb_poso = wb_poso[:this_count, :]
            
            # per wordbox stats:
            for ft, ft_name in enumerate(cls_extract_types):
                voted_cls_extract_type = pred[:, ft] >= 0.5
                truth_cls_extract_type = truth[:, ft] >= 0.5
                tp = np.sum(voted_cls_extract_type & truth_cls_extract_type)
                fp = np.sum(voted_cls_extract_type & ~truth_cls_extract_type)
                fn = np.sum(~voted_cls_extract_type & truth_cls_extract_type)
                tn = np.sum(~voted_cls_extract_type & ~truth_cls_extract_type)
                ft_per_wb[ft_name][1, 1] += tp
                ft_per_wb[ft_name][0, 0] += tn
                ft_per_wb[ft_name][1, 0] += fp
                ft_per_wb[ft_name][0, 1] += fn
    
    print("micro nongb f1:")
    for ft in cls_extract_types:
        print("{}: {}".format(ft, T_f1(ft_per_wb[ft])))
        print("{}".format(ft_per_wb[ft]))
        print(" ")
    print("total micro conf matrix:")
    print(sum([ft_per_wb[ft] for ft in ft_per_wb]))
    print("total micro nonbg f1:")
    totalmicrof1 = T_f1(sum([ft_per_wb[ft] for ft in ft_per_wb]))
    print("... {}".format(totalmicrof1))
    return totalmicrof1


def repair_annotations(trgt_annots):
    out = []
    while len(trgt_annots) > 0:
        first, *rest = trgt_annots
        first = set(first)
        
        lf = -1
        while len(first) > lf:
            lf = len(first)
            
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2
        
        out.append(first)
        trgt_annots = rest
    return out


def eval_match_annotations(trgt_annots, pred_annots):
    # assert in target annotations nothing is a part of 2 and more annotations (we are dealing with per-cls_extract_type now, so that should hold)
    all_nms_to_trgt = defaultdict(lambda: -1)
    for i, nms in enumerate(trgt_annots):
        for nm in nms:
            assert nm not in all_nms_to_trgt
            all_nms_to_trgt[nm] = i
    
    trgt_matched = [False] * len(trgt_annots)
    
    evl = {'good': 0, 'wrong': 0, 'miss': 0, 'extra': 0}
    
    for pred_annot in pred_annots:
        assert len(pred_annot) > 0, "pass only nonempty annotations please"
        corresponds = set(all_nms_to_trgt[nm] for nm in pred_annot)
        
        if len(corresponds) > 1:
            # some wordboxes were found in target
            # (-1 in corresponds) but also some wordbox was not found in the target , so that is 'bad' (with some extra wordboxes)
            # or (2 nonegative numbers in corresponds) - was found to be from 2 target annotations - also bad
            evl['wrong'] += len([c for c in corresponds if c >= 0 and trgt_matched[c] == False])
            
            for c in corresponds:
                if c >= 0:
                    trgt_matched[c] = True
        else:
            if -1 in corresponds:
                # we found all the wordboxes to not belong to any target
                evl['extra'] += 1
            else:
                # we found all the wordboxes to belong to one target, yay!
                belong = list(corresponds)[0]
                if trgt_matched[belong] == False:
                    evl['good'] += 1
                    trgt_matched[belong] = True
    
    # do not forget to run over trgt matched and produce miss?
    evl['miss'] += sum([1 for mt in trgt_matched if mt == False])
    
    return evl


def evaluating_ftypes_targets_separated(eval_ds, eval_size, model, cls_extract_types, verbose_progress=True,
                                        plots_prefix=None):
    ft_per_wb = defaultdict(lambda: np.zeros((2, 2), dtype=int))  # predicted, real cls_extract_type per wordbox
    ft_satisfication = defaultdict(lambda: [0, 0, 0])
    # success (needed & provided), miss (needed & not provided), extra (not needed & provided) ... we do not care
    # about extras btw
    
    ft_per_annotation = defaultdict(lambda: dict({'good': 0, 'wrong': 0, 'miss': 0, 'extra': 0}))
    
    for istep in tqdm(range(eval_size), total=eval_size, disable=not verbose_progress):
        batch = six.next(eval_ds)
        # batch structire: [0] - x, [1]: concatenated y and weights
        annotations = {item: batch[0][item] for item in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS
                       if item in batch[0]}
        x = {item: batch[0][item] for item in batch[0] if
             item not in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        predicted_data = model.predict_on_batch(x)
        for b, (wb_poso, pred, truth, truth_weights, annotation) in enumerate(
                zip(x['wb-poso'], predicted_data, batch[1], batch[2], annotations['annotations'])):
            # first remove batched-padded items:
            this_count = truth.shape[0]
            for i in range(truth.shape[0]):
                if truth_weights[i] == 0:
                    # as soon as we hit the padded 0-weight, we know that that is the real length of
                    # the array of wordboxes
                    this_count = i
            truth = truth[:this_count, ...]
            # btw might be cut as a single piece from a much longer page, thats why some annotations
            # might be missing!
            pred = pred[:this_count, ...]
            wb_poso = wb_poso[:this_count, :]
            
            trgt_annots = [(ft, annot_ids) for ft, annot_ids in zip(annotation['cls_extract_type'], annotation['ids'])
                           if len(annot_ids) > 0]

            if plots_prefix is not None:
                produce_drawings(istep, b, cls_extract_types, truth, pred, x['wb-bbox'][b],
                                 x['nearest-annotated'][b] if 'nearest-annotated' in x else None,
                                 x['nearest-wb-bbox'][b] if 'nearest-wb-bbox' in x else None,
                                 plots_prefix)
            
            # per wordbox stats:
            for ft, ft_name in enumerate(cls_extract_types):
                voted_cls_extract_type = pred[:, ft] >= 0.5
                truth_cls_extract_type = truth[:, ft] >= 0.5
                tp = np.sum(voted_cls_extract_type & truth_cls_extract_type)
                fp = np.sum(voted_cls_extract_type & ~truth_cls_extract_type)
                fn = np.sum(~voted_cls_extract_type & truth_cls_extract_type)
                tn = np.sum(~voted_cls_extract_type & ~truth_cls_extract_type)
                ft_per_wb[ft_name][1, 1] += tp
                ft_per_wb[ft_name][0, 0] += tn
                ft_per_wb[ft_name][1, 0] += fp
                ft_per_wb[ft_name][0, 1] += fn
                
                # Groupping mechanism for evaluating per annotation:
                # now the wordboxes are ordered in a reading order. Lets say that we have a mechanism,
                # that would concatenate all in the same line to be the same annotation!
                # [x['wb-poso'][0, i, 0:2] for i in range(all wordboxes!)]
                # ... maybe this does not need to be the best algorithm! the rows are selected using a constant!
                
                produced_fls = []
                votes = list(voted_cls_extract_type) + [False]
                rowsbegs = list(wb_poso[:, 1]) + [0]
                annot_beg = None
                for wordbox_i in range(this_count + 1):
                    voted = votes[wordbox_i]
                    if ((annot_beg is not None) and  # we are already in an annotation
                            (not voted or rowsbegs[
                                wordbox_i] == 0)):  # and now we encounter something not-for extraction or on 'new line'
                        # produce annotation_beg -> wordbox_i - 1 INCLUDING
                        produced_fls.append((annot_beg, wordbox_i - 1))
                        annot_beg = None
                    elif annot_beg is None and voted:  # encountered begin:
                        annot_beg = wordbox_i
                
                pred_annots = [list(range(prod[0], prod[1] + 1)) for prod in produced_fls]
                trgt_fl_annots = [item[1] for item in trgt_annots if item[0] == ft_name]
                
                lables_eval = eval_match_annotations(repair_annotations(trgt_fl_annots), pred_annots)
                for restype in ft_per_annotation[ft_name].keys():
                    ft_per_annotation[ft_name][restype] += lables_eval[restype]
    
    print("micro nongb f1:")
    for ft in cls_extract_types:
        print("{}: {} (lbl {} -> {})".format(ft, T_f1(ft_per_wb[ft]), str(dict(ft_per_annotation[ft])),
                                             GWME_f1(ft_per_annotation[ft])))
        print("{}".format(ft_per_wb[ft]))
        print(" ")
    totalmicrof1 = T_f1(sum([ft_per_wb[ft] for ft in ft_per_wb]))
    print("total micro conf matrix: (total micro nonbg f1: {})".format(totalmicrof1))
    print(sum([ft_per_wb[ft] for ft in ft_per_wb]))
    lbl_tot = GWME_sum(ft_per_annotation)
    print("total micro lbl: (f1: {})".format(GWME_f1(lbl_tot)))
    print(dict(lbl_tot))
    return totalmicrof1


def evaluating_ftypes_targets_reuse(eval_ds, eval_size, model, cls_extract_types, verbose_progress=True, plots_prefix=None):
    ft_per_wb = defaultdict(lambda: np.zeros((2, 2), dtype=int))  # predicted, real cls_extract_type per wordbox
    ft_satisfication = defaultdict(lambda: [0, 0, 0])
    # success (needed & provided), miss (needed & not provided), extra (not needed & provided) ... we do not care
    # about extras btw
    
    for istep in tqdm(range(eval_size), total=eval_size, disable=not verbose_progress):
        batch = six.next(eval_ds)
        # batch structire: [0] - x, [1]: concatenated y and weights
        annotations = {item: batch[0][item] for item in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        x = {item: batch[0][item] for item in batch[0] if
             item not in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        predicted_data = model.predict_on_batch(x)
        for b, (wb_poso, pred, truth, annotation, nearest_annotation, nearest_cls_extract_type_to_ordered_ids) in enumerate(
                zip(x['wb-poso'], predicted_data, batch[1], annotations['annotations'],
                    annotations['nearest-annotations'],
                    annotations['nearest-cls_extract_type-to-ordered-ids'])):
            # first remove batched-padded items:
            this_count = truth.shape[0]
            for i in range(truth.shape[0]):
                if truth[i, -1] == 0:  # the last channel is surely a weight
                    # as soon as we hit the padded 0-weight, we know that that is the real length of
                    # the array of wordboxes
                    this_count = i
            truth = truth[:this_count, ...]
            # btw might be cut as a single piece from a much longer page, thats why some annotations
            # might be missing!
            pred = pred[:this_count, ...]
            wb_poso = wb_poso[:this_count, :]

            if plots_prefix is not None:
                produce_drawings(istep, b, cls_extract_types, truth, pred, x['wb-bbox'][b],
                                 x['nearest-annotated'][b], x['nearest-wb-bbox'][b],
                                 plots_prefix)
            # per wordbox stats:
            for ft, ft_name in enumerate(cls_extract_types):
                voted_cls_extract_type = pred[:, ft] >= 0.5
                truth_cls_extract_type = truth[:, ft] >= 0.5
                tp = np.sum(voted_cls_extract_type & truth_cls_extract_type)
                fp = np.sum(voted_cls_extract_type & ~truth_cls_extract_type)
                fn = np.sum(~voted_cls_extract_type & truth_cls_extract_type)
                tn = np.sum(~voted_cls_extract_type & ~truth_cls_extract_type)
                ft_per_wb[ft_name][1, 1] += tp
                ft_per_wb[ft_name][0, 0] += tn
                ft_per_wb[ft_name][1, 0] += fp
                ft_per_wb[ft_name][0, 1] += fn
            
            # embedding - capacity stats:
            needed_fts = {anot for anot, ids in zip(annotation['cls_extract_type'], annotation['ids']) if len(ids) > 0}
            provided_fts = {anot for anot, ids in zip(nearest_annotation['cls_extract_type'], nearest_annotation['ids']) if
                            len(ids) > 0}
            
            for ft in needed_fts.union(provided_fts):
                isneed = ft in needed_fts
                isprov = ft in provided_fts
                if isneed and isprov:
                    ft_satisfication[ft][0] += 1
                elif isneed and not isprov:
                    ft_satisfication[ft][1] += 1
                elif not isneed and isprov:
                    ft_satisfication[ft][2] += 1
    
    print("micro nongb f1:")
    for ft in cls_extract_types:
        print("{}: {}".format(ft, T_f1(ft_per_wb[ft])))
        print("{}".format(ft_per_wb[ft]))
        print(" ")
    print("total micro conf matrix:")
    print(sum([ft_per_wb[ft] for ft in ft_per_wb]))
    print("total micro nonbg f1:")
    totf = T_f1(sum([ft_per_wb[ft] for ft in ft_per_wb]))
    print("... {}".format(totf))
    return totf


def evaluate_ftypes_legacy(validation_data, validation_steps, model, cls_extract_types, verbose_progress):
    """
    Testing legacy evaluation function, as was used in the previous article, to see if the scores match.
    """
    
    y_pred_percent, y_real_classes = collect_eval_data(validation_data, validation_steps, model,
                                                       verbose_progress=verbose_progress)
    all_m = array_all_classification_metrics(y_pred_percent, y_real_classes,
                                             classnames=cls_extract_types,
                                             as_binary_problems=True)
    
    body_i = None
    table_body = "N/A"
    head_i = None
    table_head = "N/A"
    
    all_nontable = [item for i, item in enumerate(all_m) if i not in [body_i, head_i]]
    nontable_confusion = sum([np.asarray(score['confusion']) for score in all_nontable])
    all_confusion = sum([np.asarray(score['confusion']) for score in all_m])
    
    def acc_rep(good, extra, miss, total):
        not_missing = good + extra
        precision = good / not_missing if not_missing > 0 else 0.0
        not_extra = good + miss
        recall = good / not_extra if not_extra > 0 else 0.0
        accuracy = good / total if total > 0.0 else 0.0
        p_plus_r = precision + recall
        f1 = 2.0 * precision * recall / p_plus_r if p_plus_r > 0.0 else 0.0
        return {"prec": precision, "recall": recall,
                "accuracy": accuracy,
                "f1": f1}
    
    def print_confs(conf_m):
        
        if not isinstance(conf_m, np.ndarray):
            return None
        
        if conf_m.shape != (2, 2):
            # print("N/A (too small dataset to contain both gs and preds)")
            return None
        
        good = conf_m[0][0] + conf_m[1][1]
        extra = conf_m[1][0]
        miss = conf_m[0][1]
        total = np.sum(conf_m)
        
        return {"all-micro": acc_rep(good, extra, miss, total),
                "nonbg-micro": acc_rep(conf_m[1][1], extra, miss, conf_m[1][1] + extra + miss),
                "conf": conf_m,
                }
    
    result = {"table_body": table_body, "table_head": table_head,
              "all": print_confs(all_confusion),
              "nontables": print_confs(nontable_confusion),
              }
    
    print("tldr:")
    print(result)
    
    return result


def collect_eval_data(validation_data, validation_steps, model, use_np_tmp=True, verbose_progress=True):
    if not use_np_tmp:
        array_represent = lambda x: x
    else:
        array_represent = np_as_tmp_map
    
    classes = None
    reals = []
    preds = []
    
    for i in tqdm(range(validation_steps), total=validation_steps, disable=not verbose_progress):
        batch = six.next(validation_data)
        # batch structire: [0] - x, [1]: concatenated y and weights
        annotations = {item: batch[0][item] for item in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        x = {item: batch[0][item] for item in batch[0] if
             item not in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        predicted_data = model.predict_on_batch(x)
        for b, (wb_poso, pred, truth, annotation, nearest_annotation, nearest_cls_extract_type_to_ordered_ids) in enumerate(
                zip(x['wb-poso'], predicted_data, batch[1], annotations['annotations'],
                    annotations['nearest-annotations'],
                    annotations['nearest-cls_extract_type-to-ordered-ids'])):
            # first remove batched-padded items:
            this_count = truth.shape[0]
            for i in range(truth.shape[0]):
                if truth[i, -1] == 0:  # the last channel is surely a weight
                    # as soon as we hit the padded 0-weight, we know that that is the real length of
                    # the array of wordboxes
                    this_count = i
            truth = truth[:this_count, ...]
            # btw might be cut as a single piece from a much longer page, thats why some annotations
            # might be missing!
            pred = pred[:this_count, ...]
            wb_poso = wb_poso[:this_count, :]
            
            if truth.ndim <= 1:
                this_classes = 1
            else:
                this_classes = truth.shape[-1]
            
            if classes is None:
                classes = this_classes
            
            assert classes == this_classes
            
            # y_pred_percent = tempmap(shape=(totlen, classes), dtype=np.float)
            ritem = array_represent(np.reshape(truth, (-1, classes)))
            pitem = array_represent(np.reshape(pred, (-1, classes)))
            assert ritem.shape == pitem.shape
            reals.append(ritem)
            preds.append(pitem)
    
    totlen = sum([real.shape[0] for real in reals])
    y_pred_percent = tempmap(shape=(totlen, classes), dtype=np.float32)
    y_pred_percent[:, ...] = 0
    y_real_classes = tempmap(shape=(totlen, classes), dtype=np.float32)
    y_real_classes[:, ...] = 0
    # btw if classes == 1 that means the model predicts class ids directly and not probabilities, so
    # we will not be able to compute auc roc.
    
    curr = 0
    for real, pred in zip(reals, preds):
        y_pred_percent[curr:(curr + pred.shape[0]), :] = pred
        y_real_classes[curr:(curr + real.shape[0]), :] = real
        curr += real.shape[0]
    
    return y_pred_percent, y_real_classes


def array_all_classification_metrics(y_pred_percent, y_real_classes, classnames=None, as_binary_problems=False):
    if (as_binary_problems):
        xret = []
        classes = y_pred_percent.shape[-1]
        for c in range(classes):
            print("Evaluation scores for " + str(c) + "-th class--------------------")
            if (classnames is not None):
                print(classnames[c])
            print("auc scores:")
            auc_scores = None
            try:
                auc_scores = skmetrics.roc_auc_score(y_real_classes[:, c], y_pred_percent[:, c], average=None,
                                                     sample_weight=None)
                print(auc_scores)
            except:
                print("cannot compute auc scores this time.")
            
            print("confusion matrices: (from arrays of size {})".format(y_pred_percent.shape))
            y_round = np.round(y_pred_percent[:, c])
            # df_confusion = pd.crosstab(y_real_classes[:, c], y_round, rownames=['Actual'], colnames=['Predicted'],
            #                            margins=True)
            df_confusion = skmetrics.confusion_matrix(y_real_classes[:, c], y_round)
            if df_confusion.shape == (1, 1):
                real_confusion = np.zeros((2, 2), dtype=int)
                r_pos = int(y_real_classes[:, c][0])
                real_confusion[r_pos, r_pos] = df_confusion[0, 0]
                df_confusion = real_confusion
            
            print(df_confusion)
            print("Accuracy: {}".format(float(sum([df_confusion[i, i] for i in range(len(df_confusion))], 0.0))
                                        / float(sum(df_confusion.flatten(), 0.0))))
            clsreport = skmetrics.classification_report(y_real_classes[:, c], y_round)  # target_names!=classnames
            print(clsreport)
            
            metricsreport = skmetrics.precision_recall_fscore_support(y_real_classes[:, c], y_round)
            
            xret.append({"confusion": df_confusion, "classfication_report": clsreport, "metricsreport": metricsreport})
        return xret
    else:
        if (len(y_pred_percent.shape) <= 1):
            xarrlen = y_pred_percent.shape[0]
            y_pred = np.zeros(xarrlen)
            for i in range(xarrlen):
                if (y_pred_percent[i] >= 0.5):
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
        else:
            y_pred = np.argmax(y_pred_percent, axis=1)
        
        if (len(y_real_classes.shape) <= 1):
            y_real = y_real_classes
        else:
            y_real = np.argmax(y_real_classes, axis=1)
        
        print("auc scores:")
        auc_scores = None
        try:
            assert y_real_classes.ndim > 1 and y_real_classes.shape[-1] > 1
            auc_scores = skmetrics.roc_auc_score(y_real_classes, y_pred_percent, average=None, sample_weight=None)
            print(auc_scores)
        except:
            print("cannot compute auc scores this time.")
        
        print("confusion matrices:")
        # df_confusion = pd.crosstab(y_real, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
        df_confusion = skmetrics.confusion_matrix(y_real, y_pred)
        print(df_confusion)
        print("Accuracy: {}".format(float(sum([df_confusion[i, i] for i in range(len(df_confusion))], 0.0))
                                    / float(sum(df_confusion.flatten(), 0.0))))
        clsreport = skmetrics.classification_report(y_real, y_pred, target_names=classnames)
        print(clsreport)
        
        metricsreport = skmetrics.precision_recall_fscore_support(y_real, y_pred)
        
        return {"confusion": df_confusion, "classfication_report": clsreport, "metricsreport": metricsreport}


def report_ftypes(legacy_eval_f, eval_ds, eval_size, model, dfobj, verbose, plots_prefix=None):
    if legacy_eval_f:
        result = evaluate_ftypes_legacy(eval_ds, eval_size, model,
                                        cls_extract_types=dfobj.pass_cls_extract_types, verbose_progress=verbose == 1)
    else:
        result = evaluating_ftypes_targets_reuse(eval_ds, eval_size, model,
                                                 cls_extract_types=dfobj.pass_cls_extract_types,
                                                 verbose_progress=verbose == 1, plots_prefix=plots_prefix)
    
    return result
