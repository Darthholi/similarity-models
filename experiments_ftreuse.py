#
# COPYRIGHT Martin Holecek 2019
#

from collections import OrderedDict
from pathlib import Path

import click
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Concatenate
from keras.models import Model

from utils.attention import AttentionTransformer
from utils.evals import EvaluateFCallback, EvaluateFCallbackToLogs, binary_crossentropy_with_weights_mt, \
    recall_multitarget, accuracy_binary_multitarget, num_predicted_ones_binary_multitarget, \
    num_truth_ones_binary_multitarget, report_ftypes, evaluating_ftypes_targets_separated
from experiments_ft import doc_inputs_siam_part
from utils.k_utils import GatherFromIndices
from utils.manipulations_utils import care_weights_save_file
from utils.sqlite_experiment_generator import FtypesTrgtDocsTextsSqliteNearest, dfobj_cache_logic, \
    FtypesTrgtDocsTextsSqliteNearestSeparated
from utils.generic_utils import PythonLiteralOption, PythonLiteralOptionOrString

pic_inpgrid_width = 620
pic_inpgrid_height = 877

"""
todo:
1) jina eval funkce? Ne
2) jina mnozina dat
selekce cls_extract_types (ne, je stejna)
3) jinej dataset? (fci na dataset co ma v sobe embeddingy) zkousi se
4) jiny weights a jejich implementace?
 ------co kdyz to je v def multiclass_temoral_class_weights!!!!
5) jinej model?
- mene attention
- i picture?

6) zkouset zmenit model ve words reuse pridavanim zrychleni..?
7) zkusit reprodukovat model bez nearest, jenom cls_extract_types - experiments_ft
 - - weights jsou stejny cca 3-4 rady (s https://jenkins.rossum.ai/job/PythonRun/1602/consoleText tj non reuse articlemodel)
"""


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
@click.option('--use_fov_annotated', default=0)
@click.option('--use_only_annotated_wordboxes', default=True)
@click.option('--debug', default=False)
@click.option('--only_previous', default=1000, cls=PythonLiteralOptionOrString)
@click.option('--embeddings_dist_cache', default=None)
@click.option('--embeddings_dist_cache_cmd', default=None)  # rewrite, 'create_only'
@click.option('--legacy_eval_f', is_flag=True)
@click.option('--weights_separate', is_flag=True)
@click.option('--use_pic', is_flag=True)
@click.option('--texts_method', default=None)
@click.option('--att_ask_all', is_flag=True)
@click.option('--att_add_skip', is_flag=True)
@click.option('--att_add_dense', is_flag=True)
@click.option('--allowed_train_and_val', is_flag=True)
@click.option('--ft_weights', default='auto')
@click.option('--n_att', default=2)
@click.option('--emb_size', default=640)  # try 64 too
@click.option('--plots_prefix', default="plots-ftreuse-")  # try 64 too
def run_keras_rendered_experiment_cls_extract_types(sqlite_source,
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
                                             use_fov_annotated,
                                             use_only_annotated_wordboxes,
                                             debug,
                                             only_previous,
                                             embeddings_dist_cache,
                                             embeddings_dist_cache_cmd,
                                             legacy_eval_f,
                                             weights_separate,
                                             use_pic,
                                             texts_method,
                                             att_ask_all,
                                             att_add_skip,
                                             att_add_dense,
                                             allowed_train_and_val,
                                             ft_weights,
                                             n_att,
                                             emb_size,
                                             plots_prefix
                                             ):
    weights_best_fname = care_weights_save_file(weights_best_save, Path(__file__).stem)
    
    if checkpoint_resume == 'None':
        checkpoint_resume = None
    if ft_weights == 'None':
        ft_weights = None
    if use_only_annotated_wordboxes in ['None', 'False']:
        use_only_annotated_wordboxes = False
    
    if not isinstance(cls_extract_types, list):
        cls_extract_types = 'all'
    
    if weights_separate:
        obj_gen = FtypesTrgtDocsTextsSqliteNearestSeparated
    else:
        obj_gen = FtypesTrgtDocsTextsSqliteNearest
    
    dfobj = obj_gen(sqlite_source, 'pdf',
                    df_proc_num=df_proc_num,
                    batch_size=batch_size,
                    df_batches_to_prefetch=3,
                    binary_class_weights=binary_class_weights,
                    limit=limit,
                    verbose_progress=verbose == 1,
                    use_neighbours=neighbours,
                    use_fov_annotated=use_fov_annotated,
                    pass_cls_extract_types=cls_extract_types,
                    ft_weights=ft_weights,
                    use_pic=use_pic,
                    texts_method=texts_method
                    )
    
    dfobj.get_index()
    
    if not dfobj_cache_logic(dfobj, embeddings_dist_cache, embeddings_dist_cache_cmd, verbose=verbose):
        return
    
    datakwargs = {'only_previous': only_previous, 'allowed_train_and_val': allowed_train_and_val}
    
    siam_reusing_objects_dict = {}
    
    def siam_layer(construction, name):
        if name not in siam_reusing_objects_dict:
            siam_reusing_objects_dict[name] = construction
        return siam_reusing_objects_dict[name]
    
    def build_model(n_siz, use_only_annotated_wordboxes):
        shapes_inp = dfobj.get_batchpadded_shapes()
        ft_count = len(dfobj.pass_cls_extract_types)
        
        inputs_doc, feats_doc = doc_inputs_siam_part("", n_siz=n_siz, neighbours=neighbours,
                                                     shapes_inp=shapes_inp, use_pic=use_pic,
                                                     siam_layer=siam_layer, emb_size=emb_size, n_att=n_att)
        inputs_nearest, feats_nearest = doc_inputs_siam_part("nearest-", n_siz=n_siz, neighbours=neighbours,
                                                             shapes_inp=shapes_inp, use_pic=use_pic,
                                                             siam_layer=siam_layer, emb_size=emb_size, n_att=n_att)
        
        nearest_ftypes = Input(shape=shapes_inp[0]['nearest-annotated'], name='nearest-annotated')
        inputs_nearest.append(nearest_ftypes)
        
        annotation_all_orig = Concatenate(axis=-1)([feats_nearest, nearest_ftypes])  # along features
        annotation_all = Dense(emb_size * n_siz, activation='sigmoid')(annotation_all_orig)
        
        if use_only_annotated_wordboxes:
            # Just be careful, when we also use the 'ask for the annotation from the annotated document',
            # it might not work, because there is no way to target the zero-classes
            nearest_boxes_ids = Input(shape=shapes_inp[0]['nearest-reuse-ids'],
                                      name='nearest-reuse-ids', dtype=np.int32)
            inputs_nearest.append(nearest_boxes_ids)
            annotation_all = GatherFromIndices(mask_value=0,
                                               include_self=False, flatten_indices_features=False)(
                [annotation_all, nearest_boxes_ids])
            
            # note that now we use the knowledge, that padding for 'nearest-reuse-ids' is -1,
            # so all padded items will have features set to be mask_value=0
        
        # 3 ways to proceed:
        # 1) Ask for the annotations from the annotated document
        # 2) Ask for the annotations from the annotated document and concatenate with the originally calculated
        #   - (we want to use now)
        # 3) Queries = document_inner, values and keys = concatenated document_inner and document_inner_annotated
        # Attention accepts the inputs in the order: queries keys values
        
        if att_ask_all:
            annotation_all_to_att = Concatenate(axis=-2)([annotation_all, feats_doc])
        else:
            annotation_all_to_att = annotation_all
        
        fork_node = AttentionTransformer(usesoftmax=True, usequerymasks=False, num_heads=8 * n_siz,
                                         num_units=emb_size * n_siz,
                                         causality=False)([feats_doc, annotation_all_to_att, annotation_all_to_att])
        # attention transformer has already some skip connections to queries inside, so lets hope it will use them,
        # or we can add some now
        if att_add_skip:
            fork_node = Concatenate(axis=-1)([fork_node, feats_doc])
        
        if att_add_dense:
            fork_node = Dense(128 * n_siz, activation='relu')(fork_node)
        
        if use_fov_annotated > 0:
            # add field of view wordboxes (by indices) from the annotated document
            fov_ids = Input(shape=shapes_inp[0]['fov_ids'], name='fov_ids', dtype=tf.int32)
            inputs_doc.append(fov_ids)
            
            gather_fovs = GatherFromIndices(mask_value=0,
                                            include_self=False, flatten_indices_features=True) \
                ([annotation_all_orig, fov_ids])
            
            fork_node = Concatenate(axis=-1)([fork_node, gather_fovs])
            fork_node = Dense(128 * n_siz, activation='relu')(fork_node)
        else:
            fov_ids = None
        
        outputs = Dense(ft_count,
                        activation='sigmoid',
                        name="targets")(fork_node)
        
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
        
        allinputs = inputs_doc + inputs_nearest
        model = Model(inputs=allinputs, outputs=out_dupl_dim)
        model.compile(optimizer='adam', loss=loss_option,
                      metrics=metrics,
                      sample_weight_mode='temporal' if weights_separate else None)  # we use custom loss to encode the weights...
        return model
    
    # Load checkpoint:
    if checkpoint_resume is not None:
        # Load model:
        
        # model = load_model(checkpoint_resume)
        model = build_model(n_siz=n_siz, use_only_annotated_wordboxes=use_only_annotated_wordboxes)
        model.load_weights(checkpoint_resume)
        # Finding the epoch index from which we are resuming
        # initial_epoch = get_init_epoch(checkpoint_path)
    else:
        model = build_model(n_siz=n_siz, use_only_annotated_wordboxes=use_only_annotated_wordboxes)
        # initial_epoch = 0
    
    model.summary()
    
    # eval_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val'))
    eval_ds, eval_size = dfobj.get_final_dataflow_dataset('val', datacolumns_mode='evaluate', **datakwargs)
    
    def get_report_f(eval_ds, eval_size, model, cls_extract_types, verbose, plots_prefix=None):
        def report():
            if weights_separate:
                return evaluating_ftypes_targets_separated(eval_ds, eval_size, model, cls_extract_types, verbose == 1, plots_prefix)
            else:
                return report_ftypes(legacy_eval_f, eval_ds, eval_size, model, dfobj, verbose, plots_prefix)
        
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
        print("evaluationg before first epoch")
        get_report_f(eval_ds, eval_size, model, cls_extract_types, verbose)()
    
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
            get_report_f(test_ds, test_size, model, cls_extract_types, verbose, plots_prefix=plots_prefix)()
        
    return None


if __name__ == "__main__":
    run_keras_rendered_experiment_cls_extract_types()
