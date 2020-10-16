#
# COPYRIGHT Martin Holecek 2019
#

from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import six
from tqdm import tqdm

from utils import produce_drawings
from utils.evals import eval_match_annotations, \
    repair_annotations, T_f1, GWME_f1, GWME_sum
from utils.generic_utils import PythonLiteralOption, PythonLiteralOptionOrString
from utils.manipulations_utils import care_weights_save_file
from utils.sqlite_experiment_generator import FtypesTrgtDocsTextsSqliteNearest
from utils.sqlite_experiment_generator import dfobj_cache_logic, \
    FtypesTrgtDocsTextsSqliteNearestSeparated

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


class CopyModel(object):
    def __init__(self, pass_cls_extract_types):
        self.pass_cls_extract_types = pass_cls_extract_types
        self.weight_by_dists = True
        self.just_take_best = True
    
    def predict_on_batch(self, x):
        ret_batch = []
        for b, (fov_ids, nannot, nrids, wbbbox, nbbox) in enumerate(
                zip(x['fov_ids'], x['nearest-annotated'] ,x['nearest-reuse-ids'], x['wb-bbox'], x['nearest-wb-bbox'])):
            current_wbs = len(fov_ids)
            all_types = nannot.shape[-1]
            
            this_box_centers = np.stack([(wbbbox[:, 0] + wbbbox[:, 2]) * 0.5, (wbbbox[:, 1] + wbbbox[:, 3]) * 0.5], axis=1)
            near_box_centers = np.stack(
                [(nbbox[:, 0] + nbbox[:, 2]) * 0.5, (nbbox[:, 1] + nbbox[:, 3]) * 0.5], axis=1)
            
            answer = np.zeros((current_wbs, all_types))  # leads to f score = 0
            
            # look at all in fov and let them simply vote, summed and weighted by distance
            for wb in range(current_wbs):
                real_in_fov = fov_ids[wb][fov_ids[wb] >= 0]
                if len(real_in_fov) > 0:
                    if self.just_take_best:
                        answer[wb, :] = nannot[real_in_fov[0]]
                    else:
                        if self.weight_by_dists:
                            this_box_center = this_box_centers[wb]
                            filter_nearest_box_centers = near_box_centers[real_in_fov]
                            dists = np.sqrt(((filter_nearest_box_centers[:, 0] - this_box_center[0]) * (
                                        filter_nearest_box_centers[:, 0] - this_box_center[0]) +
                                     (filter_nearest_box_centers[:, 1] - this_box_center[1]) * (
                                                 filter_nearest_box_centers[:, 1] - this_box_center[1])))
                            
                            summed_all_fov = np.sum(nannot[real_in_fov] * np.expand_dims(dists, -1), axis=0)
                        else:
                            summed_all_fov = np.sum(nannot[real_in_fov], axis=0)
                        max_sall = np.max(summed_all_fov)
        
                        answer[wb, summed_all_fov >= max_sall] = 1.0
            ret_batch.append(answer)
        return ret_batch


def evaluating_copy(eval_ds, eval_size, model, cls_extract_types, verbose_progress=True, plots_prefix="copy-"):
    ft_per_wb = defaultdict(lambda: np.zeros((2, 2), dtype=int))  # predicted, real cls_extract_type per wordbox
    ft_satisfication = defaultdict(lambda: [0, 0, 0])
    # success (needed & provided), miss (needed & not provided), extra (not needed & provided) ... we do not care
    # about extras btw
    
    ft_per_annotation = defaultdict(lambda: dict({'good': 0, 'wrong': 0, 'miss': 0, 'extra': 0}))
    
    for ieval in tqdm(range(eval_size), total=eval_size, disable=not verbose_progress):
        batch = six.next(eval_ds)
        # batch structire: [0] - x, [1]: concatenated y and weights
        annotations = {item: batch[0][item] for item in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS
                       if item in batch[0]}
        x = {item: batch[0][item] for item in batch[0] if
             item not in FtypesTrgtDocsTextsSqliteNearest.ANNOTATION_COLUMNS}
        predicted_data = model.predict_on_batch(x)
        for b, (wb_poso, pred, truth, truth_weights, annotation) in enumerate(
                zip(x['wb-poso'], predicted_data, batch[1], batch[2], annotations['annotations'],
                    )):
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
            
            produce_drawings(ieval, b, cls_extract_types, truth, pred, x['wb-bbox'][b],
                             x['nearest-annotated'][b], x['nearest-wb-bbox'][b], plots_prefix=plots_prefix)
            
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
@click.option('--plots_prefix', default="copy")
def run_experiment_copy(sqlite_source,
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
                    use_fov_annotated=10,
                    pass_cls_extract_types=cls_extract_types,
                    ft_weights=ft_weights,
                    use_pic=use_pic,
                    texts_method=texts_method
                    )
    
    dfobj.get_index()
    
    if not dfobj_cache_logic(dfobj, embeddings_dist_cache, embeddings_dist_cache_cmd, verbose=verbose):
        return
    
    datakwargs = {'only_previous': only_previous, 'allowed_train_and_val': allowed_train_and_val}
    
    #    eval_ds, eval_size = dfobj.get_final_dataflow_dataset('val', datacolumns_mode='evaluate', **datakwargs)
    model = CopyModel(dfobj.pass_cls_extract_types)
    
    def get_report_f(eval_ds, eval_size, model, cls_extract_types, verbose):
        def report():
            return evaluating_copy(eval_ds, eval_size, model, cls_extract_types, verbose == 1,
                                   plots_prefix)
        
        return report
    
    test_ds, test_size = dfobj.get_final_dataflow_dataset('test', datacolumns_mode='evaluate', **datakwargs)
    assert test_size > 0
    get_report_f(test_ds, test_size, model, cls_extract_types, verbose)()
    
    return None


if __name__ == "__main__":
    run_experiment_copy()
