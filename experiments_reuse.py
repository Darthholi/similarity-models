#
# COPYRIGHT Martin Holecek 2019
#

from pathlib import Path

import click
import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Concatenate
from keras.layers import Lambda, Conv2D, Add
from keras.models import Model
from collections import OrderedDict

from utils.attention import AttentionTransformer
from utils.evals import EvaluateFCallback, EvaluateFCallbackToLogs, binary_crossentropy_with_weights_matrix, \
    distances_and_classified_with_weights_v2, distances_and_classified_with_weights_and_mean, \
    binary_accuracy_with_weights, positive_samples_with_weights, positive_samples_with_weights_predicted, \
    binary_accuracy_positive_with_weights, evaluating_f_reuse, matrix_weighted_target_bce_tempsum
from utils.k_utils import tile_for_product_matrix, tile_for_product_matrix_shape, tile_to_match, GatherFromIndices
from utils.manipulations_utils import care_weights_save_file
from utils.sqlite_experiment_generator import dfobj_cache_logic, DocsTextsSqliteNearest, DocsTextsSqliteWeightedFtypes, \
    DocsTextsSqliteWeightedFtypesDebug
from experiments_ft import doc_inputs_siam_part
from utils.generic_utils import PythonLiteralOption, PythonLiteralOptionOrString


def xacc_1(y_true, y_pred):
    # also called recall
    return binary_accuracy_positive_with_weights(y_true, y_pred)


def xacc(y_true, y_pred):
    return binary_accuracy_with_weights(y_true, y_pred)


def xpred_1(y_true, y_pred):
    return positive_samples_with_weights_predicted(y_true, y_pred)


def xcount_1(y_true, y_pred):
    return positive_samples_with_weights(y_true, y_pred)


@click.command()
@click.option('--sqlite_source', default=None, )
@click.option('--checkpoint_resume', default=None, )
@click.option('--n_epochs', default=100, )
@click.option('--verbose', default=1,  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
              )
@click.option('--stop_early', default=True, )
@click.option('--key_metric', default='val_loss', )
@click.option('--weights_best_save', default='~/', )
@click.option('--patience', default=15, )
@click.option('--key_metric_mode', default='min', )
@click.option('--batch_size', default=4)
@click.option('--df_proc_num', default=4)
@click.option('--binary_class_weights', cls=PythonLiteralOption, default="(0.02, 1.0)")
@click.option('--limit', default=None)
@click.option('--n_siz', default=1)
@click.option('--neighbours', default=0)
@click.option('--mode', default='classic')  # ['triplet', 'triplet-mean', 'classic', 'classic-weightssum']
@click.option('--pass_cls_extract_types', cls=PythonLiteralOption, default="False")
@click.option('--only_previous', default=1000, cls=PythonLiteralOptionOrString)
@click.option('--embeddings_dist_cache', default=None)
@click.option('--embeddings_dist_cache_cmd', default='norewrite')  # rewrite, 'create_only'
@click.option('--use_pic', is_flag=True)
@click.option('--ft_weights', default='auto')
@click.option('--allowed_train_and_val', is_flag=True)
@click.option('--use_fov_annotated', default=0)
@click.option('--n_att', default=2)
@click.option('--emb_size', default=640)
@click.option('--no_refine', is_flag=True)
@click.option('--debug_predict_same_lengths', is_flag=True)
@click.option('--plots_prefix', default="reuse")
def run_keras_rendered_experiment_binary(sqlite_source,
                                         checkpoint_resume,
                                         mode,
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
                                         limit,
                                         n_siz,
                                         neighbours,
                                         pass_cls_extract_types,
                                         only_previous,
                                         embeddings_dist_cache,
                                         embeddings_dist_cache_cmd,
                                         use_pic,
                                         use_fov_annotated,
                                         ft_weights,
                                         allowed_train_and_val,
                                         n_att,
                                         emb_size,
                                         no_refine,
                                        debug_predict_same_lengths,
                                        plots_prefix,
                                         ):
    """
    About class weights:
    in total there are 272777372 predicions, 7338849 class 1 and 265438523 class
    ^ 2% of positive classes!

    -> there is 50 times less positive classes, so lets give them 50 times the weight:
    So lets use bin_class_weights = (1.0 , 50.0)

    """
    assert not use_fov_annotated, "So far this option is unused"
    
    weights_best_fname = care_weights_save_file(weights_best_save, Path(__file__).stem)
    
    if checkpoint_resume == 'None':
        checkpoint_resume = None
    if ft_weights == 'None':
        ft_weights = None
    
    if not isinstance(pass_cls_extract_types, list):
        if pass_cls_extract_types == True:
            pass_cls_extract_types = 'all'
        else:
            pass_cls_extract_types = None
    
    if ft_weights is not None:
        if debug_predict_same_lengths:
            dfclass = DocsTextsSqliteWeightedFtypesDebug
        else:
            dfclass = DocsTextsSqliteWeightedFtypes
    else:
        dfclass = DocsTextsSqliteNearest
    
    dfobj = dfclass(sqlite_source, 'pdf',
                    df_proc_num=df_proc_num,
                    batch_size=batch_size,
                    df_batches_to_prefetch=3,
                    binary_class_weights=binary_class_weights,
                    limit=limit,
                    verbose_progress=verbose == 1,
                    use_neighbours=neighbours,
                    pass_cls_extract_types=pass_cls_extract_types,
                    use_pic=use_pic,
                    ft_weights=ft_weights,
                    use_fov_annotated=use_fov_annotated)
    
    dfobj.get_index()
    if not dfobj_cache_logic(dfobj, embeddings_dist_cache, embeddings_dist_cache_cmd, verbose):
        return
    
    datakwargs = {'only_previous': only_previous, 'allowed_train_and_val': allowed_train_and_val}
    
    def build_model():
        # inputs: ['wb-text-features', 'wb-text-onehot', 'wb-bbox', 'wb-poso',
        #  'nearest-wb-text-features', 'nearest-wb-text-onehot', 'nearest-wb-bbox', 'nearest-wb-poso',
        #  'nearest-reuse-ids']
        shapes_inp = dfobj.get_batchpadded_shapes()
        nearest_boxes_ids = Input(shape=shapes_inp[0]['nearest-reuse-ids'],
                                  name='nearest-reuse-ids', dtype=np.int32)
        # as if use_only_annotated_wordboxes turned on always
        
        siam_reusing_objects_dict = {}
        
        def siam_layer(construction, name):
            if name not in siam_reusing_objects_dict:
                siam_reusing_objects_dict[name] = construction
            return siam_reusing_objects_dict[name]

        inputs_doc, feats_doc = doc_inputs_siam_part("", n_siz=n_siz, neighbours=neighbours,
                                                     shapes_inp=shapes_inp, use_pic=use_pic,
                                                     siam_layer=siam_layer, emb_size=emb_size, n_att=n_att)
        inputs_nearest, feats_nearest = doc_inputs_siam_part("nearest-", n_siz=n_siz, neighbours=neighbours,
                                                             shapes_inp=shapes_inp, use_pic=use_pic,
                                                             siam_layer=siam_layer, emb_size=emb_size, n_att=n_att)
        
        if pass_cls_extract_types:
            # there are, of course, different methods how to pass this
            nearest_ftypes = Input(shape=shapes_inp[0]['nearest-annotated'], name='nearest-annotated')
            inputs_nearest.append(nearest_ftypes)
            
            feats_nearest = Concatenate(axis=-1)([feats_nearest, nearest_ftypes])
            feats_nearest = Dense(emb_size, activation='sigmoid')(feats_nearest)
        
        feats_annotated_nearest = GatherFromIndices(mask_value=0,
                                                    include_self=False, flatten_indices_features=False)(
            [feats_nearest, nearest_boxes_ids])
        # note that now we use the knowledge, that padding for 'nearest-reuse-ids' is -1,
        # so all padded items will have features set to be mask_value=0
        
        tiled_nearest, tiled_doc = Lambda(lambda u: tile_for_product_matrix(u[0], u[1]),
                                          output_shape=lambda u: tile_for_product_matrix_shape(u[0], u[1]))([
            feats_annotated_nearest, feats_doc])
        # the (both) shapes are then (batch, feats_doc[sequence len], nearest-seq-len, features dim)
        
        if mode in ['triplet', 'triplet-mean', 'triplet-l2cos-mean']:
            if mode in ['triplet-l2cos-mean']:
                # the idea is to give the network also the distances in cosine similarity
                distances_eucl = Lambda(lambda u:
                                        K.sum(K.square(u[0][..., :emb_size / 2] - u[1][..., :emb_size / 2]),
                                              axis=-1, keepdims=True),
                                        )([tiled_doc, tiled_nearest])
                
                distances_cos = Lambda(lambda u:
                                       K.sum(u[0][..., emb_size / 2:] * u[1, emb_size / 2:], axis=-1,
                                             keepdims=True),
                                       )([tiled_doc, tiled_nearest])
                
                distances_decide = Concatenate(axis=-1)([distances_eucl, distances_cos])
                distances_summed = Add(axis=-1)([distances_eucl, distances_cos])
            else:  # classical only eucleid space
                distances_decide = Lambda(lambda u:
                                          K.sum(K.square(u[0] - u[1]), axis=-1, keepdims=True),
                                          )([tiled_doc, tiled_nearest])
                distances_summed = distances_decide
            
            distances_classified = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(distances_decide)
            
            out_dupl_dim = Concatenate(axis=-1)(
                [distances_classified, distances_summed])  # because we need custom loss function
            if mode in ['triplet-mean', 'triplet-l2cos-mean']:  # converges faster
                loss_option = distances_and_classified_with_weights_and_mean
            else:  # triplet
                loss_option = distances_and_classified_with_weights_v2
        else:  # 'classic', 'classic-weightssum'
            both = Concatenate(axis=-1)([tiled_doc, tiled_nearest])
            
            proc_both = Conv2D(64 * n_siz, kernel_size=(1, 1), activation='relu')(both)
            refine = not no_refine
            if refine:
                maxed = Lambda(lambda u: K.max(u, axis=-2))(proc_both)
                tuned = AttentionTransformer(usesoftmax=True, usequerymasks=False, num_heads=8 * n_siz,
                                             num_units=64 * n_siz,
                                             causality=False)([maxed, maxed, maxed])
                
                classified_refined = Lambda(lambda u: tile_to_match(u[0], u[1]))(
                    [tuned, proc_both])  # repeat tuned to dist classified
                cls_concern = Concatenate(axis=-1)([proc_both, classified_refined])
            else:
                cls_concern = proc_both
            cls_final = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(cls_concern)
            out_dupl_dim = Concatenate(axis=-1)([cls_final, cls_final])  # because we need custom loss function
            if mode == 'classic':
                loss_option = binary_crossentropy_with_weights_matrix
            elif mode == 'classic-weightssum':
                loss_option = matrix_weighted_target_bce_tempsum
            else:
                raise ValueError('Wrong mode')
        
        allinputs = inputs_doc + inputs_nearest + [nearest_boxes_ids]
        
        model = Model(inputs=allinputs, outputs=out_dupl_dim)
        model.compile(optimizer='adam', loss=loss_option,
                      metrics=[xacc, xacc_1, xcount_1, xpred_1],
                      )  # sample_weight_mode='temporal')  # we use custom looss to encode the weights...
        return model
    
    # Load checkpoint:
    if checkpoint_resume is not None:
        # Load model:
        
        # model = load_model(checkpoint_resume)
        model = build_model()
        model.load_weights(checkpoint_resume)
        # Finding the epoch index from which we are resuming
        # initial_epoch = get_init_epoch(checkpoint_path)
    else:
        model = build_model()
        # initial_epoch = 0
    
    model.summary()
    
    # eval_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val'))
    eval_ds, eval_size = dfobj.get_final_dataflow_dataset('val', datacolumns_mode='evaluate', **datakwargs)

    def get_report_f(eval_ds, eval_size, model, verbose, plots_prefix=None):
        def report():
            result = evaluating_f_reuse(eval_ds, eval_size, model, verbose_progress=verbose == 1, plots_prefix=plots_prefix)
            return result
        return report

    callbacks = OrderedDict()
    if key_metric == 'custom':
        callbacks['val_reporter'] = EvaluateFCallbackToLogs( get_report_f(eval_ds, eval_size, model, verbose),
            monitor_save=key_metric, )
    else:
        callbacks['val_reporter'] = EvaluateFCallback(get_report_f(eval_ds, eval_size, model, verbose),
                                                      monitor=key_metric,
                                                      mode=key_metric_mode
                                                      )
    callbacks['checkpointer'] = ModelCheckpoint(weights_best_fname,
                                                monitor=key_metric, save_best_only=True, mode=key_metric_mode,
                                                save_weights_only=True,  # otherwise some error in saving...
                                                verbose=verbose)
    
    if checkpoint_resume is not None:
        # if loaded, lets evaluate is to see...
        # callbacks['val_reporter'].on_epoch_end(epoch=0)
        print("evaluationg before first epoch")
        get_report_f(eval_ds, eval_size, model, verbose)()
    
    if stop_early:
        callbacks['early_stopping'] = EarlyStopping(monitor=key_metric, patience=patience,
                                                    mode=key_metric_mode, verbose=verbose)
    with K.get_session() as sess:
        if n_epochs > 0:
            # train_ds, train_size = dfobj.get_final_tf_data_dataset('train')
            # val_ds, val_size = dfobj.get_final_tf_data_dataset('val')
            train_ds, train_size = dfobj.get_final_dataflow_dataset('train', datacolumns_mode='training', **datakwargs)
            val_ds, val_size = dfobj.get_final_dataflow_dataset('val', datacolumns_mode='training', **datakwargs)
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
            get_report_f(test_ds, test_size, model, verbose, plots_prefix)()


if __name__ == "__main__":
    run_keras_rendered_experiment_binary()
