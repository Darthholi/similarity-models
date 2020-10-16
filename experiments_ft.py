#
# COPYRIGHT Martin Holecek 2019
#

from collections import OrderedDict
from pathlib import Path

import click
import keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Concatenate, Dropout, Reshape
from keras.layers import Lambda, Conv2D, Conv3D, Conv1D, MaxPooling2D, Flatten, TimeDistributed
from keras.models import Model

from utils.attention import AttentionTransformer, SinCosPositionalEmbedding
from utils.evals import EvaluateFCallback, EvaluateFCallbackToLogs, binary_crossentropy_with_weights_mt, \
    recall_multitarget, accuracy_binary_multitarget, num_predicted_ones_binary_multitarget, \
    num_truth_ones_binary_multitarget, evaluating_ftypes_targets, evaluating_ftypes_targets_separated
from utils.k_utils import GatherFromIndices
from utils.keras_utils_crop_render import keras_crop_and_resize_batches_of_arrays
from utils.manipulations_utils import care_weights_save_file
from utils.sqlite_experiment_generator import FtypesTrgtDocsTextsSqlite, FtypesTrgtDocsTextsSqliteSeparated
from utils.generic_utils import PythonLiteralOption

pic_inpgrid_width = 620
pic_inpgrid_height = 877


def doc_inputs_all_for_linear(prefix="", neighbours=0,
                         shapes_inp=None):
    inp_text_f = Input(shape=shapes_inp[0]['wb-text-features'], name=prefix + 'wb-text-features')
    inp_text_chars = Input(shape=shapes_inp[0]['wb-text-onehot'], name=prefix + 'wb-text-onehot')
    inp_bbox = Input(shape=shapes_inp[0]['wb-bbox'], name=prefix + 'wb-bbox')
    inp_poso = Input(shape=shapes_inp[0]['wb-poso'], name=prefix + 'wb-poso')
    if neighbours > 0:
        inp_neighs = Input(shape=shapes_inp[0]['neighbours-ids'], name=prefix + 'neighbours-ids',
                           dtype=tf.int32)
    else:
        inp_neighs = None
    
    chars_reshaped = Reshape(target_shape=(-1, shapes_inp[0]['wb-text-onehot'][-1] * shapes_inp[0]['wb-text-onehot'][-2]))(inp_text_chars)
    merged_result = Concatenate(axis=-1)([inp_text_f, chars_reshaped, inp_bbox, inp_poso])
    
    # gather neighbours so that we will see them and can operate on them also:
    boxes_input_with_neighbours = GatherFromIndices(mask_value=0,
                                                     include_self=True, flatten_indices_features=True) \
        ([merged_result, inp_neighs]) if neighbours > 0 else merged_result
    
    
    inputs_all = [input for input in [inp_text_f, inp_text_chars, inp_bbox, inp_poso, inp_neighs]
                  if input is not None]
    return inputs_all, boxes_input_with_neighbours


def doc_inputs_siam_part(prefix="", n_siz=1, neighbours=0,
                         shapes_inp=None, use_pic=False, siam_layer=None, emb_size=640,
                         n_att=1):
    if siam_layer is None:
        siam_layer = lambda layer, name: layer
    
    inp_text_f = Input(shape=shapes_inp[0]['wb-text-features'], name=prefix + 'wb-text-features')
    inp_text_chars = Input(shape=shapes_inp[0]['wb-text-onehot'], name=prefix + 'wb-text-onehot')
    inp_bbox = Input(shape=shapes_inp[0]['wb-bbox'], name=prefix + 'wb-bbox')
    inp_poso = Input(shape=shapes_inp[0]['wb-poso'], name=prefix + 'wb-poso')
    if neighbours > 0:
        inp_neighs = Input(shape=shapes_inp[0]['neighbours-ids'], name=prefix + 'neighbours-ids',
                           dtype=tf.int32)
    else:
        inp_neighs = None
    positions_embedded = siam_layer(SinCosPositionalEmbedding(4 * n_siz,
                                                              embeddings=['sin', 'cos', 'lin'],
                                                              from_inputs_features=[0, 1, 2, 3],
                                                              # embedd all 4 integers
                                                              pos_divisor=10000,
                                                              keep_ndim=True), 'posemb-pos')(inp_poso)
    bboxes_embedded = siam_layer(SinCosPositionalEmbedding(4 * n_siz,
                                                           embeddings=['sin', 'cos', 'lin'],
                                                           from_inputs_features=[0, 1, 2, 3],
                                                           # embedd all 4 integers
                                                           pos_divisor=10000,
                                                           keep_ndim=True), 'posemb-bb')(inp_bbox)
    # "channels_last"
    chars_emb = siam_layer(Conv2D(filters=50 * n_siz, kernel_size=(1, 3),
                                  strides=1, padding='same', activation='relu'), 'chars_emb1') \
        (inp_text_chars)
    chars_emb = siam_layer(Conv2D(filters=50 * n_siz, kernel_size=(1, 3),
                                  strides=1, padding='same', activation='relu'), 'chars_emb2') \
        (chars_emb)
    # custom maxpooling of feaures:
    chars_feats_emb = siam_layer(Lambda(lambda x: [K.max(x, axis=-2, keepdims=False),
                                                   K.mean(x, axis=-2, keepdims=False)]), 'meanmaxpool')(
        chars_emb)
    
    all_features = siam_layer(Concatenate(axis=-1), 'concatall') \
        ([positions_embedded, bboxes_embedded, inp_text_f] + chars_feats_emb)
    
    if use_pic:
        pic_input = Input(shape=(pic_inpgrid_width, pic_inpgrid_height, 1), name=prefix + 'pic')
        '''
        The text W/H ratios have mean 3.58996720367 and median 2.625
        The text Ws have mean 71.3550074689 and median 54.0
        The text Hs have mean 20.1268264316 and median 21.0

        so 20 is the 'average' height text box in coordinates of RENDER_WIDTH = 1240x(RENDER_WIDTH * 1.414)
        - h - 0,011406671 -> w = 0,02994251
        '''
        pic_conv_kernel_size = 5
        lookup_layer = 5
        global_pic = True
        
        pic_1 = siam_layer(Conv2D(64, pic_conv_kernel_size, padding='same', strides=1, activation='relu'),
                           'picconv')(pic_input)
        pic_2 = siam_layer(MaxPooling2D(pool_size=4), 'picpool')(pic_1)  # 155x219
        pic_3 = siam_layer(Conv2D(32, pic_conv_kernel_size, padding='same', strides=1, activation='relu'),
                           'picconv2')(pic_2)
        '''
        if we take the crops from here, that would mean, that we crop average thing:
        height: 219×0.011406671 ~ 2.5 (w corresponding by the median is 6.56)
        The morphological dilation takes the mean height and adds it to all sides of the bbox, so we end up with:
        w: 11.56 h: 7.5, lets take 12x8 in 'keras_crop_and_resize_batches_of_arrays'
        (note that it originally sees 48x32 pixels)
        '''
        pic_4 = siam_layer(MaxPooling2D(pool_size=4), 'picpool2')(pic_3)  # 38x54
        pic_5 = siam_layer(Conv2D(32, pic_conv_kernel_size, padding='same', dilation_rate=3, activation='relu'),
                           'picconv3')(pic_4)
        '''
        if we take it from here, the 12x8 changes to 4x downsample -> 3x2
        '''
        pic_6 = siam_layer(MaxPooling2D(pool_size=4), 'picpool3')(pic_5)
        pic_7 = siam_layer(Conv2D(32, pic_conv_kernel_size, padding='same', dilation_rate=3, activation='relu'),
                           'picconv4')(pic_6)
        pic_8 = siam_layer(MaxPooling2D(pool_size=4), 'picpool4')(pic_7)
        pic_9 = Flatten()(pic_8)
        pic_final = siam_layer(Dense(32, activation='relu'), 'picfinal')(pic_9)
        
        # bboxes ltrb are at first positions:
        def extract_bbox_apply_fov_f(boxes_input):
            # lets use as 'dilation' (the distance to see everything around) the mean height of the bbox
            # ... mean - across the batch.
            bboxes = boxes_input[..., 0:4]
            dilate = K.mean(K.abs(bboxes[:, 3] - bboxes[:, 1]))
            return bboxes + [-dilate, -dilate, dilate, dilate]
        
        boxes_input_bigger_fov = TimeDistributed(Lambda(extract_bbox_apply_fov_f))(inp_bbox)
        
        if lookup_layer == 3:
            boxes_cuts = keras_crop_and_resize_batches_of_arrays(pic_3, boxes_input_bigger_fov,
                                                                  crop_size=(12, 8))
            boxes_cuts = siam_layer(Conv3D(10 * n_siz, [1, pic_conv_kernel_size, pic_conv_kernel_size],
                                            padding='valid', activation='relu'), 'piclookconv')(boxes_cuts)
            boxes_cuts = TimeDistributed(Flatten())(boxes_cuts)
        elif lookup_layer == 5:
            boxes_cuts = keras_crop_and_resize_batches_of_arrays(pic_5, boxes_input_bigger_fov,
                                                                  crop_size=(3, 2))
            # dim 3x2*32
            boxes_cuts = TimeDistributed(Flatten())(boxes_cuts)
            boxes_cuts = siam_layer(Dense(64 * n_siz, activation='relu'), 'piclookdense')(boxes_cuts)
        
        else:
            raise ValueError("lookup layer can be only 3 or 5")
        
        boxes_cuts = siam_layer(Conv1D(100 * n_siz, kernel_size=1, padding='same', activation='relu'),
                                 'picconvcuts')(boxes_cuts)
        # should be the same as size of boxes ...
        
        if global_pic is not None:
            def timedistributed_concat(packed):
                x, pic = packed
                return K.concatenate([x, K.repeat(pic, K.shape(x)[-2])], axis=-1)
            
            merged_result_one = Lambda(timedistributed_concat)([all_features, pic_final])
        else:
            merged_result_one = all_features
        
        # merged_result_one = Concatenate()([boxes, pic_final])
        merged_result = Concatenate(axis=-1)([merged_result_one, boxes_cuts])
        # dimensionality should be now bigger
    else:
        merged_result = all_features
        pic_input = None
    
    # gather neighbours so that we will see them and can operate on them also:
    boxes_input_with_neighbours = GatherFromIndices(mask_value=0,
                                                     include_self=True, flatten_indices_features=True) \
        ([merged_result, inp_neighs]) if neighbours > 0 else merged_result
    
    feat_ext = siam_layer(Dense(256 * n_siz, activation='relu'), 'denseall')(boxes_input_with_neighbours)
    feat_regulariz = siam_layer(Dropout(0.15), 'dropoutall')(feat_ext)
    
    seq_conv = siam_layer(Conv1D(128 * n_siz, kernel_size=5, padding='same',
                                 activation='relu'), 'convall')(feat_regulariz)
    densflat = siam_layer(Dense(64 * n_siz, activation='relu'), 'denseflatbefatt')(seq_conv)
    
    att = siam_layer(AttentionTransformer(usesoftmax=True, usequerymasks=False, num_heads=8 * n_siz,
                                          num_units=64 * n_siz,
                                          causality=False), 'atrafo1')([densflat, densflat, densflat])
    
    if n_att == 2:
        att = siam_layer(AttentionTransformer(usesoftmax=True, usequerymasks=False, num_heads=8 * n_siz,
                                              num_units=64 * n_siz,
                                              causality=False), 'atrafo1')([att, att, att])
    
    emb_perbox = siam_layer(Dense(emb_size * n_siz, activation='sigmoid'), 'sigmemb')(att)
    
    inputs_all = [input for input in [inp_text_f, inp_text_chars, inp_bbox, inp_poso, inp_neighs, pic_input]
                  if input is not None]
    return inputs_all, emb_perbox


@click.command()
@click.option('--sqlite_source', default=None, )
@click.option('--checkpoint_resume', default=None, )
@click.option('--n_epochs', default=100, )
@click.option('--verbose', default=2,  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
              )
@click.option('--stop_early', default=True, )
@click.option('--key_metric', default='val_loss', )
@click.option('--weights_best_save', default='~/', )
@click.option('--patience', default=15, )
@click.option('--key_metric_mode', default='min', )
@click.option('--batch_size', default=4)
@click.option('--df_proc_num', default=4)
@click.option('--binary_class_weights', cls=PythonLiteralOption, default="(1.0, 1.0)")
@click.option('--cls_extract_types', cls=PythonLiteralOption, default="None")
@click.option('--limit', default=None)
@click.option('--n_siz', default=1)
@click.option('--neighbours', default=1)
@click.option('--debug', default=False)
@click.option('--weights_separate', is_flag=True)
@click.option('--use_pic', is_flag=True)
@click.option('--texts_method', default=None)
@click.option('--n_att', default=1)
@click.option('--emb_size', default=640)  # try 64 too
@click.option('--plots_prefix', default="plotsft-")
def run_keras_rendered_experiment_cls_extract_types_only(sqlite_source,
                                                  checkpoint_resume,
                                                  n_epochs,
                                                  verbose,
                                                  stop_early,
                                                  key_metric,
                                                  weights_best_save,
                                                  patience,
                                                  key_metric_mode,
                                                  batch_size,
                                                  df_proc_num,
                                                  binary_class_weights,
                                                  cls_extract_types,
                                                  limit,
                                                  n_siz,
                                                  neighbours,
                                                  debug,
                                                  weights_separate,
                                                  use_pic,
                                                  texts_method,
                                                  n_att,
                                                  emb_size,
                                                  plots_prefix
                                                  ):
    weights_best_fname = care_weights_save_file(weights_best_save, Path(__file__).stem)
    
    if checkpoint_resume == 'None':
        checkpoint_resume = None
    
    if not isinstance(cls_extract_types, list):
        cls_extract_types = 'all'
    
    if weights_separate:
        dfclass = FtypesTrgtDocsTextsSqliteSeparated
    else:
        dfclass = FtypesTrgtDocsTextsSqlite
    
    dfobj = dfclass(sqlite_source, 'pdf',
                    df_proc_num=df_proc_num,
                    batch_size=batch_size,
                    df_batches_to_prefetch=3,
                    binary_class_weights=binary_class_weights,
                    limit=limit,
                    verbose_progress=verbose == 1,
                    use_neighbours=neighbours,
                    pass_cls_extract_types=cls_extract_types,
                    ft_weights='auto',
                    use_pic=use_pic,
                    texts_method=texts_method
                    )
    
    dfobj.get_index()
    
    def build_model(n_siz):
        shapes_inp = dfobj.get_batchpadded_shapes()
        ft_count = len(dfobj.pass_cls_extract_types)
        
        inputs_doc, feats_doc = doc_inputs_siam_part(prefix="", n_siz=n_siz, neighbours=neighbours,
                                                     shapes_inp=shapes_inp, use_pic=use_pic, siam_layer=None,
                                                     emb_size=emb_size, n_att=n_att)
        
        outputs = Dense(ft_count,
                        activation='sigmoid',
                        name="targets")(feats_doc)
        
        if weights_separate:
            out_dupl_dim = outputs
            loss_option = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            out_dupl_dim = Concatenate(axis=-1)([outputs, outputs])  # because we need custom loss function
            
            def loss_option(t, p):
                return binary_crossentropy_with_weights_mt(t, p, ft_count)
            
            metrics = [accuracy_binary_multitarget(ft_count), recall_multitarget(ft_count),
                       num_truth_ones_binary_multitarget(ft_count),
                       num_predicted_ones_binary_multitarget(ft_count)]
        
        allinputs = inputs_doc
        model = Model(inputs=allinputs, outputs=out_dupl_dim)
        model.compile(optimizer='adam', loss=loss_option,
                      metrics=metrics,
                      sample_weight_mode='temporal' if weights_separate else None)
        return model
    
    # Load checkpoint:
    if checkpoint_resume is not None:
        # Load model:
        
        # model = load_model(checkpoint_resume)
        model = build_model(n_siz=n_siz)
        model.load_weights(checkpoint_resume)
        # Finding the epoch index from which we are resuming
        # initial_epoch = get_init_epoch(checkpoint_path)
    else:
        model = build_model(n_siz=n_siz)
        # initial_epoch = 0
    
    model.summary()
    
    eval_ds, eval_size = dfobj.get_final_dataflow_dataset('val', datacolumns_mode='evaluate')
    
    def get_report_f(eval_ds, eval_size, model, cls_extract_types, verbose, plots_prefix=None):
        def report():
            if weights_separate:
                return evaluating_ftypes_targets_separated(eval_ds, eval_size, model, cls_extract_types, verbose == 1, plots_prefix)
            else:
                return evaluating_ftypes_targets(eval_ds, eval_size, model, cls_extract_types, verbose == 1)
        
        return report
    
    callbacks = OrderedDict()
    if key_metric == 'custom':
        callbacks['val_reporter'] = EvaluateFCallbackToLogs(
            get_report_f(eval_ds, eval_size, model, cls_extract_types, verbose),
            monitor_save=key_metric, )
    else:
        callbacks['val_reporter'] = EvaluateFCallback(get_report_f(eval_ds, eval_size, model, cls_extract_types, verbose),
                                                      monitor=key_metric,
                                                      mode=key_metric_mode
                                                      )
    callbacks['checkpointer'] = ModelCheckpoint(weights_best_fname,
                                                monitor=key_metric, save_best_only=True, mode=key_metric_mode,
                                                save_weights_only=True,  # otherwise some error in saving...
                                                verbose=verbose)
    
    if checkpoint_resume is not None or debug or n_epochs <= 0:
        # if loaded, lets evaluate is to see...
        # callbacks['val_reporter'].on_epoch_end(epoch=0)
        print("evaluating before first epoch")
        get_report_f(eval_ds, eval_size, model, cls_extract_types, verbose)()
    
    if stop_early:
        callbacks['early_stopping'] = EarlyStopping(monitor=key_metric, patience=patience,
                                                    mode=key_metric_mode, verbose=verbose)
    with K.get_session() as sess:
        if n_epochs > 0:
            # train_ds, train_size = dfobj.get_final_tf_data_dataset('train')
            # val_ds, val_size = dfobj.get_final_tf_data_dataset('val')
            train_ds, train_size = dfobj.get_final_dataflow_dataset('train', datacolumns_mode='training')
            val_ds, val_size = dfobj.get_final_dataflow_dataset('val', datacolumns_mode='training')
            
            hist = model.fit_generator(
                train_ds,  # tf_dataset_as_iterator(train_ds, sess=sess),
                train_size, n_epochs,
                verbose=verbose,
                # class_weight=class_weight,
                # keras cannot use class weight and sample weights at the same time
                validation_data=val_ds,  # tf_dataset_as_iterator(val_ds, sess=sess),
                # we validate on the same set btw
                validation_steps=val_size,
                callbacks=[callbacks[key] for key in callbacks],
                workers=0,  # because we use dataflow
                use_multiprocessing=False
            )
    
        test_ds, test_size = dfobj.get_final_dataflow_dataset('test', datacolumns_mode='evaluate')
        if test_size > 0:
            print("...")
            print("TRAINING COMPLETE, LOADING BEST SCORE ON VAL AND TESTING")
            print("...")
            print("...")
            if n_epochs > 0:
                model.load_weights(weights_best_fname)
            get_report_f(test_ds, test_size, model, cls_extract_types, verbose, plots_prefix)()
    
    return None


if __name__ == "__main__":
    run_keras_rendered_experiment_cls_extract_types_only()
