#
# COPYRIGHT Martin Holecek 2019
#

import ast

import click
import copy
import datetime
import pprint
import tempfile
import warnings

import keras.backend as K
import numpy as np
import six
import sklearn.metrics as skmetrics
import tensorflow as tf
from keras.callbacks import Callback
from keras.layers.pooling import _GlobalPooling2D


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


def np_as_tmp_map(nparray):
    tmpmap = tempmap(dtype=nparray.dtype, mode='w+', shape=nparray.shape)
    tmpmap[...] = nparray
    return tmpmap


def equal_ifarray(a, b):
    if isinstance(a, np.ndarray):
        return all(a == b)
    else:
        return a == b


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


def make_product_matrix(vect_inp):
    '''
    vect_inp: [batches, Nboxes, Cfeatures]
    should return [batches, Nboxes, Nboxes, 2*Cfeatures]
    [b,i,j,...] should be (vect_inp[b,i,:], vect_inp[b,j,:])

    example:
    data = [[0 1],[2 3],[4 5]]  # shape (1, 3, 2),
    [[[0 1],[2 3],[4 5]]]  # (1,1,3,2)
    [[[0 1]],[[2 3]],[[4 5]]]  # (1,3,1,2)
    after repets:
    [[[0 1],[2 3],[4 5]], [[0 1],[2 3],[4 5]], [[0 1],[2 3],[4 5]] ]  # (1,3,3,2)
    [[[0 1], [0 1], [0 1]],[[2 3], [2 3], [2 3]],[[4 5], [4 5], [4 5]]]  # (1,3,3,2)
    and then it could be just concatenated
    '''
    Nboxes = tf.shape(vect_inp)[-2]
    v = tf.expand_dims(vect_inp, -3)
    v_t = tf.expand_dims(vect_inp, -2)
    v = tf.tile(v, [1, Nboxes, 1, 1])
    v_t = tf.tile(v_t, [1, 1, Nboxes, 1])
    ret = tf.concat([v, v_t], -1)
    
    assert_op = tf.Assert(tf.equal(tf.shape(ret)[-2], tf.shape(ret)[-3]), [ret])
    with tf.control_dependencies([assert_op]):
        return ret


class GlobalMaxPooling1DFrom4D(_GlobalPooling2D):
    """Global max pooling operation for spatial data.

    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        3D tensor with shape:
        `(batch_size, channels)`
    """
    
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.max(inputs, axis=-2)
        else:
            return K.max(inputs, axis=-2)
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1], input_shape[3])
        else:
            return (input_shape[0], input_shape[1], input_shape[3])


def tf_dataset_as_iterator(ds, sess=K.get_session()):
    # most basic way to use tf dataset. Not optimal.
    iterator = ds.make_one_shot_iterator()
    
    iternext = iterator.get_next()
    # or    with tf.Session() as sess:
    while True:
        ret = sess.run(iternext)
        yield ret


def evaluate(validation_data, validation_steps, model, config):
    # as_binary_problems=False, predict_gold_f=self.predictions_golds_from_batch
    y_pred_percent, y_real_classes = collect_eval_data(validation_data, validation_steps, model, config)
    all_m = array_all_classification_metrics(y_pred_percent, y_real_classes,
                                             classnames=None,  # config['data_config']['bools_names'],
                                             as_binary_problems=True)
    
    try:
        body_i = config['data_config']['bools_names'].index('table_body')
        metrics_r = all_m[body_i]['metricsreport']
        table_body = (metrics_r[2][0] * metrics_r[3][0] + metrics_r[2][1] * metrics_r[3][1]) / (
                metrics_r[3][0] + metrics_r[3][1])
    except:
        body_i = None
        table_body = "N/A"
    try:
        head_i = config['data_config']['bools_names'].index('table_header')
        metrics_r = all_m[head_i]['metricsreport']
        table_head = (metrics_r[2][0] * metrics_r[3][0] + metrics_r[2][1] * metrics_r[3][1]) / (
                metrics_r[3][0] + metrics_r[3][1])
    except:
        head_i = None
        table_head = "N/A"
    
    # if None not in [table_head, table_body]:
    #     tables_confusion = np.sum([all_m[i]['confusion'] for i in [table_head, table_body]])
    # else:
    #     tables_confusion = None
    
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
    # print(result)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)
    
    return result


def collect_eval_data(validation_data, validation_steps, model, config, use_np_tmp=False):
    if not use_np_tmp:
        array_represent = lambda x: x
    else:
        array_represent = np_as_tmp_map
    
    classes = None
    reals = []
    preds = []
    for i in range(0, validation_steps):
        batch = six.next(validation_data)
        for item, pred in iterate_predictions_batch(batch, model, config):
            # item is (x, y, sample weights )
            # throw away all samples that have zero weigts!
            x, y, sample_weights = item
            
            use_samples = np.nonzero(sample_weights)
            x = {key: x[key][use_samples] for key in x}
            y = y[use_samples]
            pred = pred[use_samples]
            
            if y.ndim <= 1:
                this_classes = 1
            else:
                this_classes = y.shape[-1]
            
            if classes is None:
                classes = this_classes
            
            assert classes == this_classes
            
            # y_pred_percent = tempmap(shape=(totlen, classes), dtype=np.float)
            ritem = array_represent(np.reshape(y, (-1, classes)))
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


def expand_batch(data):
    """
    The input is some struct full of lists, dictss and ultimately numpy arrays.
    We will deconstruct the struct and yield individual items from the batch. In the same format as the input.
    """
    str_o = StructIndexer.from_example(data)
    indexed = str_o.unpack_from(data)
    batch_size = len(indexed[list(indexed.keys())[0]])  # if it is a numpy array, it should give us the leftmost dimension
    # or if it is just a list then the length of the list...
    for b in range(batch_size):
        expanded = {k: indexed[k][b] for k in indexed.keys()}
        yield str_o.pack_from(expanded)


def iterate_predictions_batch(batch, model, config):
    predictions = model.predict_on_batch(batch[0])
    for item, pred in zip(expand_batch(batch), expand_batch(predictions)):
        yield item, pred


class StructIndexer(object):
    def __init__(self, indices, classes_struct):
        self.indices = indices
        self.clases_struct = classes_struct
    
    def unpack_from(self, data):
        if self.indices is None:
            return {None: data}
        indexed = {}
        for indice in self.indices:
            indexed[tuple(indice)] = self._get_indexed(data, indice)
        return indexed
    
    def _get_indexed(self, data, indice):
        x = data
        for index in indice:
            x = x[index]
        return x
    
    def _set_indexed(self, data, indice, value):
        x = data
        for index in indice[:-1]:
            x = x[index]
        x[indice[-1]] = value
    
    def pack_from(self, indexed):
        if self.indices is None:
            return indexed[None]
        data = copy.deepcopy(self.clases_struct)
        for indice in indexed.keys():
            self._set_indexed(data, indice, indexed[indice])
        return data
    
    @classmethod
    def from_example(cls, struct):
        if isinstance(struct, dict) or isinstance(struct, list) or isinstance(struct, tuple):
            indices, clstruct = cls.analyze([], struct)
        else:
            indices = None
            clstruct = None
        return cls(indices, clstruct)
    
    @classmethod
    def analyze(cls, indexes, template):
        struct_indices = []
        struct_classes = None
        if isinstance(template, dict):
            struct_classes = {}
            for key in template.keys():
                new_indices, new_classes = cls.analyze(tuple(list(indexes) + [key]), template[key])
                struct_indices.extend(new_indices)
                struct_classes[key] = new_classes
        elif isinstance(template, list) or isinstance(template, tuple):
            struct_classes = [None] * len(template)
            for key in range(len(template)):
                new_indices, new_classes = cls.analyze(tuple(list(indexes) + [key]), template[key])
                struct_indices.extend(new_indices)
                struct_classes[key] = new_classes
        else:
            return tuple([indexes]), None
        return tuple(struct_indices), struct_classes


class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except:
                raise click.BadParameter(value)
        else:
            return value

class PythonLiteralOptionOrString(click.Option):
    def type_cast_value(self, ctx, value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except:
                return value
        else:
            return value
    