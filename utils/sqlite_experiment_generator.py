#
# COPYRIGHT Martin Holecek 2019
#

import json
import math
import sqlite3
import time
from collections import defaultdict
from contextlib import closing

import hashlib
import io
import numpy as np
import os
import random
import tensorflow as tf
from pympler import asizeof
from tensorpack import DataFromList, MapData, MultiProcessMapData
from tensorpack.dataflow import RepeatedData
from tqdm import tqdm

from utils.boxgeometry import sanitize_bbox
from utils.dataflows import BatchDataPadder, tf_dataset_as_iterator, RandomOrderSequence, RandomPhaseSequence
from utils.manipulations_utils import hash_generator, hash_numpy, dynamic_memmap, OpenMemmap, dynamic_trimap, \
    OpenTriMemmapSet, OpenTriMemmapGet, is_single_integer, multiclass_temporal_class_weights, \
    class_weights_from_counts_binary, produce_fov_ids
from utils.textutils import default_char_list, default_char_vocab, text_onehot_chars, features_from_text, \
    features_from_text_len

PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


class DFSourceDataset:
    """
    This design is just a fast prototype, that paralelizes  dataflows (dataflow_packer
     and expand_data_call) and then batches and pads using tf data dataset.
    """
    
    def __init__(self, df_proc_num, batch_size, df_batches_to_prefetch):
        # assert isinstance(pagegen_obj, ConceptsPageGen)
        self.df_proc_num = df_proc_num
        self.batch_size = batch_size
        self.df_batches_to_prefetch = df_batches_to_prefetch
        self.shuffle_in_tfdataset = False
    
    def get_indices(self, phase='train', **kwargs):
        return DataFromList(lst=list(range(kwargs['pages_per_epoch'])))  # just data indices
    
    def expand_data_call(self, *args, **kwargs):
        return None
    
    def get_output_types(self, **kwargs):
        return None
    
    def get_batchpadded_shapes(self, **kwargs):
        return None
    
    def get_batchpadded_values(self, **kwargs):
        return None
    
    def get_final_tf_data_dataset(self, phase='train', **kwargs):
        df, size = self.dataflow_packer(phase, **kwargs)
        return self.tf_data_dataset_batcher_from_dataflow(df, phase, **kwargs), size / self.batch_size
    
    def dataflow_packer(self,
                        phase='train',  # or val or predict
                        **kwargs
                        # random_state=42,
                        ):
        phase_fit = phase in ['train', 'val']  # validation happens in fitting also
        
        df = self.get_indices(phase, **kwargs)
        if df is None:
            return None
        
        buffer_size = self.batch_size * self.df_batches_to_prefetch
        orig_size = df.size()
        
        # at first, datapoint components are flat (for BatchData)
        if self.df_proc_num <= 1:
            df = MapData(df, lambda arg: self.expand_data_call(arg, phase=phase, **kwargs))
        else:
            # btw in python 3 it can hang when n_proc = 1 AND buffer_size > 1 batch
            df = MultiProcessMapData(df, self.df_proc_num,
                                     lambda arg: self.expand_data_call(arg, phase=phase, **kwargs),
                                     buffer_size=min(buffer_size, orig_size),
                                     strict=True)  # https://github.com/tensorpack/tensorpack/pull/794
        return df, orig_size
    
    def tf_data_dataset_batcher_from_dataflow(self, ds, phase='train', **kwargs):
        ds.reset_state()
        return self.tf_data_dataset_batcher_from_generator(ds, phase=phase, **kwargs).get_data()
    
    def tf_data_dataset_batcher_from_generator(self, gen, phase='train', **kwargs):
        phase_fit = phase in ['train', 'val']  # validation happens in fitting also
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            self.get_output_types(**kwargs),
            output_shapes=None)
        
        if self.shuffle_in_tfdataset and phase_fit:  # not needed now
            dataset = dataset.shuffle(1000)
        
        # Transform and batch data at the same time
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #    preprocess_fn, batch_size,
        #    num_parallel_batches=4,  # cpu cores
        #    drop_remainder=True if is_training else False))
        
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=self.get_batchpadded_shapes(**kwargs),
            padding_values=self.get_batchpadded_values(**kwargs))
        
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)  # tf.data.experimental.AUTOTUNE
        return dataset
        # then in keras model just call make oneshot iterator according to
        # https://gist.github.com/datlife/abfe263803691a8864b7a2d4f87c4ab8
    
    ############### second interface, in case of memoryleaking tf data dataset
    def get_final_dataflow_dataset(self, phase='train', **kwargs):
        df, size = self.dataflow_packer(phase, **kwargs)
        df_batched = self.dataflow_batcher_from_generator(df, phase=phase, **kwargs)
        df_batched.reset_state()
        return df_batched.get_data(), int(math.ceil(size / self.batch_size))
    
    def dataflow_batcher_from_generator(self, gen, phase='train', **kwargs):
        # phase_fit = phase in ['train', 'val']  # validation happens in fitting also
        
        padder = BatchDataPadder(gen,
                                 batch_size=self.batch_size,
                                 output_types=self.get_output_types(**kwargs),
                                 padded_shapes=self.get_batchpadded_shapes(**kwargs),
                                 padding_values=self.get_batchpadded_values(**kwargs),
                                 remainder=True,
                                 use_list=False,
                                 allow_listify_specials=False)
        dataset = RepeatedData(padder, -1)
        
        # if self.shuffle_in_tfdataset and phase_fit:  # not needed now
        #    dataset = dataset.shuffle(1000)
        
        return dataset


class IndexObject:
    def __init__(self, ds_doc_page):
        """
        ds_doc_page: (dataset id, doc id, settype, page n) ordered by dataset ids and doc ids and page n
        (settype = train or val)
        """
        self.ds_doc_page = ds_doc_page
        
        self.train_set_ids = [i for i, p in enumerate(self.ds_doc_page) if p[2] == 'train']
        self.val_set_ids = [i for i, p in enumerate(self.ds_doc_page) if p[2] == 'val']
        self.test_set_ids = [i for i, p in enumerate(self.ds_doc_page) if p[2] == 'test']
        self.all_set_ids = [i for i in range(len(self.ds_doc_page))]
    
    def hash(self):
        checksum = hashlib.sha256()
        checksum = hash_generator(checksum, (item for item in self.ds_doc_page))
        return checksum.hexdigest()
    
    def get_set_ids(self, phase):
        if phase == 'train':
            return self.train_set_ids
        elif phase == 'val':
            return self.val_set_ids
        elif phase == 'test':
            return self.val_set_ids
        else:
            return self.all_set_ids


class IndexObjectNearest(IndexObject):
    def __init__(self, ds_doc_page, embeddings):
        """
        ds_doc_page: (dataset id, doc id, settype, page n) ordered by dataset ids and doc ids and page n
        (settype = train or val)
        """
        IndexObject.__init__(self, ds_doc_page)
        self.embeddings = embeddings
        
        self._cache_embsdists = None
    
    def hash(self):
        checksum = hashlib.sha256()
        checksum = hash_generator(checksum, (item for item in self.ds_doc_page))
        checksum = hash_numpy(checksum, self.embeddings.get('r'))
        return checksum.hexdigest()
    
    def get_nearest_i(self,
                      current_ordering,
                      current_i,
    
                      only_previous=1000,
                      only_same_dataset_id=True,
                      allowed_train_and_val=False,  # False = only train allowed
    
                      ):
        """
        Gets the index of the nearest page.
        The code is not optimized and runs fast only when the distances are precomputed.
        """
        assert len(current_ordering) <= self.embeddings.shape[0]
        assert len(current_ordering) <= len(self.ds_doc_page)
        
        current = self.ds_doc_page[current_ordering[current_i]]
        phase = current[2]
        
        def filter_allows(s, current):
            filtered = self.ds_doc_page[s]
            filtered_phase = filtered[2]
            if only_same_dataset_id and current[0] != filtered[0]:
                return False
            if current[1] == filtered[1]:  # doc id
                return False
            if phase == 'train':
                if filtered_phase != 'train':
                    return False
            elif phase in ['val', 'test']:
                if not allowed_train_and_val and filtered_phase == 'train':
                    return False
            return True
        
        if only_previous == True or is_single_integer(only_previous):
            search = current_ordering[: current_i]
        else:
            search = current_ordering
        if allowed_train_and_val and phase in ['val', 'test']:
            search = self.train_set_ids + search
        
        if is_single_integer(only_previous):
            search = search[-only_previous:]
        
        search_filtered = [s for s in search if filter_allows(s, current)]
        
        if only_previous and len(search_filtered) <= 0:
            # if by any chance in the current ordering, there are no suitable documents before this one, consider the whole set.
            # and run the last lines of code again...
            search = current_ordering
            if allowed_train_and_val and phase in ['val', 'test']:
                search = self.train_set_ids + search
            if is_single_integer(only_previous):
                search = search[-only_previous:]
            search_filtered = [s for s in search if filter_allows(s, current)]
        
        assert len(search_filtered) > 0
        
        if only_previous == 'random':
            nearest_i = random.choice(search_filtered)
            return nearest_i
        
        if self._cache_embsdists is None:
            embeddings = self.embeddings.get('r')
            nearest_i = self.l2_find_min(search_filtered, embeddings, embeddings[current_ordering[current_i]])
        else:
            nearest_i = self.l2_find_min_cached(search_filtered, self._cache_embsdists, current_ordering[current_i])
        
        """ in case we would need to verify the integrity while debugging
        with OpenTriMemmapGet(self._cache_embsdists) as dmat:
            all_dists1 = [dmat[i, current_ordering[current_i]] for i in search_filtered]

        embeddings = self.embeddings.get('r')
        all_dists2 = [np.sum(np.square(embeddings[considered_i] - embeddings[current_ordering[current_i]])) for considered_i in search_filtered]

        embs = np.array([embeddings[i] for i in search_filtered])
        embs = embs - embeddings[current_ordering[current_i]]
        dists = np.sum(np.square(embs), axis=-1)
        """
        
        return nearest_i
    
    def l2_find_min_cached(self, search_filtered, dists, to_i):
        with OpenTriMemmapGet(dists) as dmat:
            return min(search_filtered, key=lambda considered_i: dmat[considered_i, to_i])
    
    def l2_find_min(self, search_filtered, embeddings, point):
        if len(search_filtered) <= 5000:
            embs = np.array([embeddings[i] for i in search_filtered])
            embs = embs - point
            dists = np.sum(np.square(embs), axis=-1)
            
            argmin = np.argmin(dists)
            return search_filtered[argmin]
        else:
            return min(search_filtered, key=lambda considered_i:
            np.sum(np.square(embeddings[considered_i] - point)))
    
    def l2_batched_min(self, search_filtered, embeddings, point):
        """
        nearest_i = min(search_filtered, key=lambda considered_i:
        np.sum(np.square(self.embeddings[considered_i] - self.embeddings[current_ordering[current_i]])))

        """
        # nearest_i = min(search_filtered, key=lambda considered_i:
        # np.sum(np.square(embeddings[considered_i] - point)))
        
        batch_size = 1000
        
        def chunk_min(b):
            starts = b * batch_size
            chunklen = max(len(search_filtered) - starts, batch_size)
            embs = np.array([embeddings[i] for i in search_filtered[starts: starts + chunklen]])
            embs = embs - point
            dists = np.sum(np.square(embs), axis=-1)
            
            argmin = np.argmin(dists)
            mindist = dists[argmin]
            
            return argmin, mindist
        
        chunks = [chunk_min(b) for b in range(math.ceil(len(search_filtered) / batch_size))]
        argmin, mindist = min(chunks, key=lambda item: item[1])
        
        return search_filtered[argmin]
    
    def get_nearest(self,
                    current_ordering,
                    current_i,
    
                    only_previous=1000,
                    only_same_dataset_id=True,
                    allowed_train_and_val=False,  # False = only train allowed
    
                    ):
        return self.ds_doc_page[self.get_nearest_i(current_ordering,
                                                   current_i, only_previous, only_same_dataset_id,
                                                   allowed_train_and_val)]
    
    def exists_embcache(self, filepath):
        infof = filepath + ".json"
        distsfile = filepath + ".bin"
        return os.path.exists(infof) and os.path.exists(distsfile)
    
    def is_embcache_compatible(self, filepath):
        try:
            info = filepath + ".json"
            with open(info) as json_file:
                data = json.load(json_file)
                return data['hash'] == self.hash()
        except:
            return False
    
    def _verify_dists(self, dists, embs_all, i, j):
        n_embs = embs_all.shape[0]
        dst = (embs_all[i] - embs_all[j])
        dst = np.sum(dst * dst)
        return dists[i, j] - dst
        # np.array([dmat[i, 0] for i in range(100)]) - \
        # np.array([np.sum(np.square(embeddings[i] - embeddings[0])) for i in range(100)])
    
    def load_embcache(self, filepath):
        if self.exists_embcache(filepath):
            if self.is_embcache_compatible(filepath):
                n_embs = len(self.embeddings)
                assert n_embs == len(self.ds_doc_page)
                cache_embsdists = dynamic_trimap(side_shape=n_embs, dtype=np.float, name=filepath + ".bin", mode='r')
                with OpenTriMemmapGet(cache_embsdists) as dists:
                    with OpenMemmap(self.embeddings) as embs_all:
                        if (self._verify_dists(dists, embs_all, n_embs - 1, n_embs - 2) != 0 or
                                self._verify_dists(dists, embs_all, int(n_embs / 2) - 1, int(n_embs / 2)) != 0 or
                                self._verify_dists(dists, embs_all, int(n_embs / 2), 0)):
                            raise ValueError("Data corrupted, some datapoints are not equal")
                self._cache_embsdists = cache_embsdists
            else:
                raise ValueError("Cache not compatible with data")
        else:
            raise ValueError("Cache does not exist")
    
    def precompute_embcache(self, filepath, n_jobs=2, verbose=True):
        data = {'hash': self.hash()}
        with open(filepath + '.json', 'w') as outfile:
            json.dump(data, outfile)
        
        with OpenMemmap(self.embeddings) as embs_all:
            n_embs = embs_all.shape[0]
            
            print("creating array cache of size {} bytes".format(int(n_embs * (n_embs - 1) / 2) * 8))
            
            dt = dynamic_trimap(side_shape=n_embs, dtype=np.float, name=filepath + ".bin")
            all_pairs_cnt = dt.shape
            
            with OpenTriMemmapSet(dt, 'w+') as dists_all2all:
                from joblib import Parallel, delayed
                bef = time.time()
                print("starting")
                """
                result = Parallel(n_jobs=n_jobs)(delayed(put_item)(i, j, embs_all, dists_all2all)
                                            for (i, j) in tqdm(all_pairs(n_embs), total=all_pairs_cnt,
                                                               disable=not verbose,
                                                               #miniters=int(all_pairs_cnt/100)
                                                               ))
                """
                result = Parallel(n_jobs=n_jobs)(delayed(put_item_batched)(b, embs_all, dists_all2all)
                                                 for b in
                                                 tqdm(all_pairs_batched(n_embs, 5000), total=all_pairs_cnt / 5000,
                                                      disable=not verbose,
                                                      # miniters=int(all_pairs_cnt/100)
                                                      ))
                after = time.time()
                print(after - bef)


def dfobj_cache_logic(dfobj, embeddings_dist_cache, embeddings_dist_cache_cmd, verbose):
    if embeddings_dist_cache:
        if not dfobj.get_index().exists_embcache(embeddings_dist_cache) or embeddings_dist_cache_cmd == 'rewrite':
            print("creating cache of distances for embeddings (takes a lot of time ususally)")
            if embeddings_dist_cache_cmd == 'norewrite':
                raise ValueError("Cache does not exist and you wanted us to raise in this case!")
            dfobj.get_index().precompute_embcache(embeddings_dist_cache, n_jobs=4, verbose=verbose)
        print("loading and validating existing cache of distances for embeddings")
        dfobj.get_index().load_embcache(embeddings_dist_cache)
        print("cache loaded and verified")
        if embeddings_dist_cache_cmd == 'create_only':
            return False
    return True


class DocsTextsSqliteBase(DFSourceDataset):
    """
    """
    TEXT_VALUES_SCALE = [1.0, 1000.0, 10000000.0]
    TEXT_SCALE = 20.0
    CHARACTERS_LEN = 40
    ANNOTATION_FILTER = "and texts_list_ids!='[]' and cls_extract_type not in ('table_body', 'table_header', 'table_footer')"
    MAX_WORDBOXES_GUARD = 300
    CLOSING_DB = True
    ANNOTATION_COLUMNS = ('annotations', 'nearest-annotations', 'nearest-cls_extract_type-to-ordered-ids')
    
    def __init__(self, sqlite_source,
                 filter_doctype,
                 df_proc_num, batch_size, df_batches_to_prefetch,
                 shuffle=True,
                 binary_class_weights=(1.0, 1.0),  # for positive ang negative class
                 limit=None,
                 verbose_progress=True,
                 use_neighbours=0,
    
                 pass_cls_extract_types=None,
                 use_pic=False,
                 texts_method=None,
                 ):
        DFSourceDataset.__init__(self, df_proc_num, batch_size, df_batches_to_prefetch)
        self.sqlite_source = sqlite_source
        self.filterdoctype = filter_doctype
        self.shuffle = shuffle
        self._index = None  # index of all documents
        self.drop_empty_pages = True
        self.limit_index = limit
        self.binary_class_weights = binary_class_weights
        self.verbose_progress = verbose_progress
        self.use_neighbours = use_neighbours
        self.use_pic = use_pic
        self.texts_method = texts_method
        # sqlite connections should be opened each for each process only once and not shared!
        self._db_by_pid = {}
        
        self.pass_cls_extract_types = pass_cls_extract_types
        self._all_wordboxes = None
        self._all_ft_counts = None
    
    def get_my_db_object(self):
        if self.CLOSING_DB:
            return self._get_new_db_object()  # wil be closed anyway then
        else:
            pid = os.getpid()
            if pid not in self._db_by_pid:
                self._db_by_pid[pid] = self._get_new_db_object()
            return self._db_by_pid[pid]
    
    def _get_new_db_object(self):
        """
        Should be special for each thread/process!
        """
        sqlite_obj = sqlite3.connect(self.sqlite_source, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        return sqlite_obj
    
    def get_annotation_filter(self):
        if self.pass_cls_extract_types not in (None, 'all'):
            fts_set = ','.join("'{}'".format(k) for k in self.pass_cls_extract_types)
            filter = "and texts_list_ids!='[]' and cls_extract_type in ({})".format(fts_set)
        else:
            filter = self.ANNOTATION_FILTER
        return filter
    
    def get_nitems_on_doc(self, docid, cursor):
        n_texts_per_page = cursor.execute(
            "Select COUNT(*), page from texts where docid=='{}' group by page order by page asc".format(
                docid)).fetchall()
        n_texts_on_page = defaultdict(lambda: 0)
        n_texts_on_page.update({page: n for n, page in n_texts_per_page})
        
        annotations_per_page = cursor.execute(
            str(
                "Select count(*), page from annotations where docid=='{}' " + self.get_annotation_filter() + " group by page order by page asc").format(
                docid)).fetchall()
        annotations_on_page = defaultdict(lambda: 0)
        annotations_on_page.update({page: n for n, page in annotations_per_page})
        return n_texts_on_page, annotations_on_page
    
    def _indexquery(self):
        qr = ""
        if self.filterdoctype:
            assert self.filterdoctype in ['pdf', 'ocr']
            qr += "doctype=='{}'".format(self.filterdoctype)
        if len(qr) > 0:
            qr = " where " + qr
        indexquery = "Select * from docs" + qr + " order by dataset, docid asc"
        if self.limit_index:
            indexquery += " limit {}".format(self.limit_index)
        return indexquery
    
    def _compute_cls_extract_type_counts(self):
        if self._all_ft_counts is not None and self._all_wordboxes is not None:
            return  # already computed
        
        with closing(self.get_my_db_object()) as sqlite_obj:
            with closing(sqlite_obj.cursor()) as cursor:
                try:
                    print("Querying index for class counts (and cls_extract_type weights)...")
                    
                    a_counts_per_ft = defaultdict(list)
                    count_wordboxes_all = 0
                    
                    for (ds_id, docid, settype, pgi) in tqdm(self._index.ds_doc_page, disable=not self.verbose_progress,
                                                             total=len(self._index.ds_doc_page)):
                        
                        count_wordboxes = cursor.execute(
                            "Select count(texts.itemorder) from texts "
                            "where docid == '{}' and page == {}".format(docid, pgi)).fetchone()[0]
                        count_wordboxes_all += count_wordboxes
                        
                        query_annots = ("Select docid, cls_extract_type, texts_list_ids "
                                        "from annotations "
                                        "where docid == '{}' and page == {} {} "
                                        "order by docid, page, itemorder asc".format(
                            docid, pgi, self.get_annotation_filter()))
                        
                        annotations_on_page = cursor.execute(query_annots)
                        
                        for annotation in annotations_on_page:
                            cls_extract_type = annotation[1]
                            ids_wordboxes = json.loads(annotation[2])
                            a_counts_per_ft[cls_extract_type].append(len(ids_wordboxes))
                    
                    assert len(list(a_counts_per_ft.keys())) == len(self.pass_cls_extract_types), \
                        "all cls_extract_types should be used"
                    
                    total_ft_wordboxes = {ft: np.sum(a_counts_per_ft[ft]) for ft in a_counts_per_ft}
                    self._all_wordboxes = count_wordboxes_all
                    self._all_ft_counts = total_ft_wordboxes
                
                except sqlite3.Error as error:
                    print("Failed to read data from sqlite table", error)
                    raise error
    
    def get_index(self):
        if self._index is None:
            self._compute_index()
            self._check_pass_cls_extract_types()
            print("computed index with hash {}".format(self._index.hash()))
        return self._index
    
    def produce_ft_onehot(self, doc_struct, pass_cls_extract_types):
        # for targets or for sources...
        ft_onehot = np.zeros((len(doc_struct['wordboxes']['bbox']), len(pass_cls_extract_types)))
        ft_id = {ft: i for i, ft in enumerate(pass_cls_extract_types)}
        for ids, ft in zip(doc_struct['annotation']['ids'], doc_struct['annotation']['cls_extract_type']):
            if ft in ft_id:
                ft_onehot[ids, ft_id[ft]] = 1
        return ft_onehot
    
    def _compute_index(self):
        with closing(self.get_my_db_object()) as sqlite_obj:
            with closing(sqlite_obj.cursor()) as cursor:
                try:
                    print("Querying index...")
                    cursor.execute(self._indexquery())
                    # bef = time.time()
                    records = cursor.fetchall()
                    docs_pages = []
                    
                    max_total_pages = sum(record[3] for record in records)
                    real_total_pages = 0
                    
                    for record in tqdm(records,
                                       disable=not self.verbose_progress):  # add all pages in range of pages...
                        ds_id = record[0]
                        docid = record[1]
                        n_pages = record[3]
                        settype = record[4]  # train or val
                        
                        # filter empty pages:
                        if self.drop_empty_pages:
                            textboxes, annotations = self.get_nitems_on_doc(docid, cursor)
                        for i in range(n_pages):
                            if not self.drop_empty_pages or (textboxes[i] > 0 and annotations[i] > 0):
                                docs_pages.append((ds_id, docid, settype, i))
                                real_total_pages += 1
                    
                    assert real_total_pages == len(docs_pages)
                    print("Index of {} items of byte size: {}".format(len(docs_pages), asizeof.asizeof(docs_pages)))
                    print("({} pages dropped due to no content)".format(max_total_pages - real_total_pages))
                    
                    self._index = IndexObject(docs_pages)
                    print("Index computed")
                
                except sqlite3.Error as error:
                    print("Failed to read data from sqlite table", error)
                    raise error
    
    def _check_pass_cls_extract_types(self):
        if self.pass_cls_extract_types == 'all':
            self._compute_cls_extract_type_counts()
            
            self.pass_cls_extract_types = sorted(list(self._all_ft_counts.keys()))
            if self.verbose_progress:
                print("passing automatically computed cls_extract_types({}): {}".
                      format(len(self.pass_cls_extract_types), self.pass_cls_extract_types))
    
    def proc_bbox(self, ibbox, textfields_aabb, learning_phase):
        bbox = sanitize_bbox(ibbox, None)  # some bboxes do overlap the page, but let leave it be now.
        l = (bbox[0] - textfields_aabb[0]) / (textfields_aabb[2] - textfields_aabb[0])
        t = (bbox[1] - textfields_aabb[1]) / (textfields_aabb[3] - textfields_aabb[1])
        r = (bbox[2] - textfields_aabb[0]) / (textfields_aabb[2] - textfields_aabb[0])
        b = (bbox[3] - textfields_aabb[1]) / (textfields_aabb[3] - textfields_aabb[1])
        # assert l >= 0 and t >= 0 and r >= 0 and b >= 0 lets allow negatives
        if learning_phase:
            # here we could add more augments
            wiggle_range = 0.01
            l = l + random.uniform(-wiggle_range, +wiggle_range)
            t = t + random.uniform(-wiggle_range, +wiggle_range)
            r = r + random.uniform(-wiggle_range, +wiggle_range)
            b = b + random.uniform(-wiggle_range, +wiggle_range)
        return [l, t, r, b]
    
    def mangle_text(self, text):
        if self.texts_method == 'a':
            text = 'a' * len(text)
        elif self.texts_method == 'blank':
            text = ''
        return text
    
    def get_text_features(self, text):
        text = self.mangle_text(text)
        return features_from_text(text, values_scales=self.TEXT_VALUES_SCALE, scale=self.TEXT_SCALE,
                                  char_vocab=default_char_vocab)
    
    def get_text_onehot(self, text):
        text = self.mangle_text(text)
        return text_onehot_chars(text, self.CHARACTERS_LEN)
    
    def get_doc_data(self, dataset_id, doc_id, settype, page_n, cursor, textfields_aabb, learning_phase, get_pic):
        texts_on_page = cursor.execute(
            "Select docid, page, itemorder, "
            "bbox_l, bbox_t, bbox_r, bbox_b, "
            "content, row_readings_pos_1, row_readings_pos_2, col_readings_pos_1, col_readings_pos_2, "
            "neighbours_ids from texts where docid=='{}' and page=={} order by itemorder asc".format(
                doc_id, page_n)).fetchall()
        
        annotations_on_page = cursor.execute(
            str("Select docid, page, itemorder, cls_extract_type, content, texts_list_ids "
                "from annotations where "
                "docid=='{}' and page=={} " + self.get_annotation_filter() + " order by itemorder asc").format(
                doc_id, page_n)).fetchall()
        
        ret = {'annotation': {
            'ids': [json.loads(annotation[-1]) for annotation in annotations_on_page],
            'cls_extract_type': [annotation[-3] for annotation in annotations_on_page],
        }, 'wordboxes': {
            'bbox': [self.proc_bbox(textbox[3:7], textfields_aabb, learning_phase) for textbox in texts_on_page],
            'text-features': [self.get_text_features(textbox[7]) for textbox in texts_on_page],
            'text-onehot': [self.get_text_onehot(textbox[7]) for textbox in texts_on_page],
            'posorder': [textbox[8:12] for textbox in texts_on_page],
            'neighbours-ids': [textbox[12] for textbox in texts_on_page],
        }}
        
        if get_pic:
            page_pics = cursor.execute(
                str("Select docid, ipage, pic_array "
                    "from pics where docid=='{}' and ipage=={}").format(doc_id, page_n)).fetchall()
            ret['pic'] = page_pics[0][-1] / 255.0
        
        # tldr:
        if len(ret['wordboxes']['bbox']) >= self.MAX_WORDBOXES_GUARD:
            all_anns = sum(ret['annotation']['ids'], [])
            assert len(all_anns) > 0, "no annotations detected"
            """
            chose min and max from the range and then random inside. Fail - sometimes selects emtpy interval!!
            from copy import deepcopy
            ret_copied = deepcopy(ret)
            possible_range = max(0, min(all_anns) - 20),\
                             min(max(all_anns) + 20, len(ret['wordboxes']['bbox']))
            possible_beg_beg = min(possible_range[0], len(ret['wordboxes']['bbox']) - self.MAX_WORDBOXES_GUARD)
            possible_beg_end = max(possible_range[-1]-self.MAX_WORDBOXES_GUARD, possible_beg_beg)

            beg_cut = random.randint(possible_beg_beg, possible_beg_end)
            """
            # -Second method - select random annotation id and the interval (somehow randomly) around it!
            random_included = all_anns[random.randrange(0, len(all_anns))]
            beg_cut_r = random_included - int(self.MAX_WORDBOXES_GUARD / 2) \
                        + random.randrange(-int(self.MAX_WORDBOXES_GUARD / 4),
                                           int(self.MAX_WORDBOXES_GUARD / 4))
            beg_cut_min = max(0, beg_cut_r)
            beg_cut = min(beg_cut_min, len(ret['wordboxes']['bbox']) - self.MAX_WORDBOXES_GUARD)
            
            ret['wordboxes']['bbox'] = ret['wordboxes']['bbox'][beg_cut:beg_cut + self.MAX_WORDBOXES_GUARD]
            ret['wordboxes']['text-features'] = ret['wordboxes']['text-features'][
                                                beg_cut:beg_cut + self.MAX_WORDBOXES_GUARD]
            ret['wordboxes']['text-onehot'] = ret['wordboxes']['text-onehot'][
                                              beg_cut:beg_cut + self.MAX_WORDBOXES_GUARD]
            ret['wordboxes']['posorder'] = ret['wordboxes']['posorder'][
                                           beg_cut:beg_cut + self.MAX_WORDBOXES_GUARD]
            ret['wordboxes']['neighbours-ids'] = ret['wordboxes']['neighbours-ids'][
                                                 beg_cut:beg_cut + self.MAX_WORDBOXES_GUARD]
            
            for i, annotation in enumerate(ret['annotation']['ids']):
                ret['annotation']['ids'][i] = [z - beg_cut for z in annotation
                                               if z >= beg_cut and z < beg_cut + self.MAX_WORDBOXES_GUARD]
            
            for i, neighbour in enumerate(ret['wordboxes']['neighbours-ids']):
                ret['wordboxes']['neighbours-ids'][i] = [z - beg_cut if
                                                         z >= beg_cut and z < beg_cut + self.MAX_WORDBOXES_GUARD
                                                         else -1
                                                         for z in neighbour
                                                         ]
            
            assert len(sum(ret['annotation']['ids'], [])) > 0, "cut too much"
        assert len(sum(ret['annotation']['ids'], [])) > 0, "no annotations??"
        return ret
    
    @classmethod
    def _filter_datapoint(cls, datapoint, **kwargs):
        filter_mode = kwargs.get('datacolumns_mode', None)
        if filter_mode is not None and filter_mode != 'evaluate':
            for item in cls.ANNOTATION_COLUMNS:
                if item in datapoint[0]:
                    del datapoint[0][item]
            # datapoint[0] = {item: datapoint[0][item] for item in datapoint[0]
            #                if item not in cls.ANNOTATION_COLUMNS}
        else:
            filter = kwargs.get('datacolumns', None)
            if filter:
                for item in datapoint[0]:
                    if item not in filter:
                        del datapoint[0][item]
                # datapoint[0] = {datapoint[0][item] for item in filter}
        return datapoint
    
    def use_neighours_count(self, input_arr, use_n):
        # remember we have a neighbour for each side of the rectangle!
        xdim = input_arr.shape[-1]
        assert xdim % 4 == 0, "each side of the box should have neighbours"
        assert xdim >= 4 * self.use_neighbours, \
            "the data should have been exported with more neighbours"
        xneigh = int(xdim / 4)
        if use_n == xneigh:
            return input_arr
        copy_first_neighbours = [edge_n * xneigh + nid for edge_n in range(4) for nid in
                                 range(use_n)]
        return input_arr[:, copy_first_neighbours]
    
    def get_indices(self, phase='train', **kwargs):
        return RandomOrderSequence(len(self.get_index().get_set_ids(phase)), self.shuffle)
    
    def expand_data_call(self, *args, **kwargs):
        with closing(self.get_my_db_object()) as db:
            with closing(db.cursor()) as cursor:
                textfields_aabb = (0, 0, PAGE_WIDTH, PAGE_HEIGHT)
                
                phase = kwargs.get('phase')
                learning_phase = phase == 'train'
                
                current_i = args[0]
                
                (dataset_id, doc_id, settype, page_n) = self.get_index().ds_doc_page[current_i]
                doc = self.get_doc_data(dataset_id, doc_id, settype, page_n, cursor, textfields_aabb, learning_phase,
                                        self.use_pic)
        
        targets = self.produce_targets(doc)
        
        datapoint = [{
            # sources
            'wb-text-features': doc['wordboxes']['text-features'],
            'wb-text-onehot': doc['wordboxes']['text-onehot'],
            'wb-bbox': doc['wordboxes']['bbox'],
            'wb-poso': doc['wordboxes']['posorder'],
            # just for metrics computations outside NN
            'annotations': doc['annotation'],
        }]
        if 'pic' in doc:  # === self.use_pic
            datapoint[0]['pic'] = doc['pic']
        self.add_targets(datapoint, targets)
        
        if self.use_neighbours > 0:
            docnids = np.stack(doc['wordboxes']['neighbours-ids'], axis=0)
            datapoint[0]['neighbours-ids'] = self.use_neighours_count(docnids, self.use_neighbours)
        
        return tuple(self._filter_datapoint(datapoint, **kwargs))
    
    def produce_targets(self, doc, **kwargs):
        raise ValueError("implemented in subclasses")
    
    def add_targets(self, datapoint, targets):
        """
        The function produce targets has the option to produce only targets or also weights.
        Here we take care of it.
        """
        if isinstance(targets, list) or isinstance(targets, tuple):
            datapoint.extend(targets)
        else:
            datapoint.append(targets)
    
    def get_output_types(self, **kwargs):
        otypes = ({
                      'wb-text-features': tf.float32,
                      'wb-text-onehot': tf.float32,
                      'wb-bbox': tf.float32,
                      'wb-poso': tf.float32,
                      # just for metrics computations outside NN:
                      'annotations': list,
                  },
                  tf.float32,
        )
        if self.use_neighbours > 0:
            otypes[0]['neighbours-ids'] = tf.int32
        if self.use_pic:
            otypes[0]['pic'] = tf.float32
        return self._filter_datapoint(otypes, **kwargs)
    
    def get_batchpadded_shapes(self, **kwargs):
        txtfeats = features_from_text_len(values_scales=self.TEXT_VALUES_SCALE, scale=self.TEXT_SCALE)
        shapes = ({
                      'wb-text-features': [None, txtfeats],
                      'wb-text-onehot': [None, self.CHARACTERS_LEN, len(default_char_list)],
                      'wb-bbox': [None, 4],
                      'wb-poso': [None, 4],
                      # just for metrics computations outside NN:
                      'annotations': [None],
                  },
                  [None, None, 2],
        )
        if self.use_neighbours > 0:
            shapes[0]['neighbours-ids'] = [None, 4 * self.use_neighbours]
        if self.use_pic:
            shapes[0]['pic'] = [None, None, 1]
        return self._filter_datapoint(shapes, **kwargs)
    
    def get_batchpadded_values(self, **kwargs):
        bvals = ({
                     'wb-text-features': 0.0,
                     'wb-text-onehot': 0.0,
                     'wb-bbox': 0.0,
                     'wb-poso': 0.0,
                     # just for metrics computations outside NN:
                     'annotations': None,
                 },
                 0.0,
            # 0.0
        )
        if self.use_neighbours > 0:
            bvals[0]['neighbours-ids'] = -1  # will make tensorflow put there zeroes in our actual gather layer.
        if self.use_pic:
            bvals[0]['pic'] = 0.0
        return self._filter_datapoint(bvals, **kwargs)


class DocsTextsSqliteNearest(DocsTextsSqliteBase):
    """
    """
    
    def __init__(self, sqlite_source,
                 filter_doctype,
                 df_proc_num, batch_size, df_batches_to_prefetch,
                 shuffle=True,
                 binary_class_weights=(1.0, 1.0),  # for positive ang negative class
                 limit=None,
                 verbose_progress=True,
                 use_neighbours=0,
                 use_fov_annotated=0,
    
                 pass_cls_extract_types=None,
                 use_pic=False,
                 texts_method=None,
                 **kwargs
                 ):
        DocsTextsSqliteBase.__init__(self, sqlite_source,
                                     filter_doctype,
                                     df_proc_num, batch_size, df_batches_to_prefetch,
                                     shuffle,
                                     binary_class_weights,
                                     limit,
                                     verbose_progress,
                                     use_neighbours,
                                     pass_cls_extract_types,
                                     use_pic,
                                     texts_method,
                                     )
        self.use_fov_annotated = use_fov_annotated

        self.keep_test_ordering = True
    
    def _compute_index(self):
        with closing(self.get_my_db_object()) as sqlite_obj:
            with closing(sqlite_obj.cursor()) as cursor:
                try:
                    print("Querying index...")
                    cursor.execute(self._indexquery())
                    # bef = time.time()
                    records = cursor.fetchall()
                    docs_pages = []
                    
                    max_total_pages = sum(record[3] for record in records)
                    dp_embeddings = None
                    real_total_pages = 0
                    
                    for record in tqdm(records,
                                       disable=not self.verbose_progress):  # add all pages in range of pages...
                        ds_id = record[0]
                        docid = record[1]
                        n_pages = record[3]
                        settype = record[4]  # train or val
                        doc_embeddings = cursor.execute(
                            "Select embedding from pages where docid=='{}' order by ipage asc".format(docid)) \
                            .fetchall()
                        doc_embeddings = [item[0] for item in doc_embeddings]
                        assert len(doc_embeddings) == n_pages  # number of embeddings is the number of pages
                        assert np.all([not np.all(emb == 0) for emb in
                                       doc_embeddings]), "some returned embeddings are full zeroes!"
                        if dp_embeddings is None:  # first access = allocate
                            # dp_embeddings = tempmap(shape=(max_total_pages, doc_embeddings[0].shape[0]),
                            #                        dtype=doc_embeddings[0].dtype)
                            dp_embeddings = dynamic_memmap(shape=(max_total_pages, doc_embeddings[0].shape[0]),
                                                           dtype=doc_embeddings[0].dtype)
                            dp_embeddings_array = dp_embeddings.get('w+')
                        
                        # filter empty pages:
                        if self.drop_empty_pages:
                            textboxes, annotations = self.get_nitems_on_doc(docid, cursor)
                        for i in range(n_pages):
                            if not self.drop_empty_pages or (textboxes[i] > 0 and annotations[i] > 0):
                                docs_pages.append((ds_id, docid, settype, i))
                                dp_embeddings_array[real_total_pages:(real_total_pages + 1)] = doc_embeddings[i]
                                real_total_pages += 1
                    
                    assert real_total_pages == len(docs_pages)
                    print("Index of {} items of byte size: {}".format(len(docs_pages), asizeof.asizeof(docs_pages)))
                    print("({} pages dropped due to no content)".format(max_total_pages - real_total_pages))
                    
                    # texts_on_page = self.cursor.execute("SELECT COUNT(*), docid, page from texts group by docid,
                    # page order by docid, page")
                    x = dp_embeddings_array[len(docs_pages) - 1]
                    del dp_embeddings_array
                    
                    dp_embeddings.shape = (len(docs_pages), doc_embeddings[0].shape[0])
                    # we need to truncate the shape, because we threw out some pages!
                    # We can do it because of the ordering
                    assert np.all(dp_embeddings.get('r')[-1] == x), "after the resize the array needs to be the same"
                    
                    self._index = IndexObjectNearest(docs_pages, dp_embeddings)
                    print("Index computed")
                
                except sqlite3.Error as error:
                    print("Failed to read data from sqlite table", error)
                    raise error
    
    def get_indices(self, phase='train', **kwargs):
        return RandomPhaseSequence(len(self.get_index().get_set_ids(phase)),
                                   self.shuffle and (phase != 'test' or not self.keep_test_ordering))
    
    def get_doc_pair(self, shuffle_seed, current_i, **kwargs):
        phase = kwargs.get('phase')
        only_previous = kwargs.get('only_previous', 1000)
        only_same_dataset_id = kwargs.get('only_same_dataset_id', True)
        allowed_train_and_val = kwargs.get('allowed_train_and_val', False)
        
        idxs = list(self.get_index().get_set_ids(phase))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(idxs)
        
        return self.get_index().ds_doc_page[current_i], self.get_index().get_nearest(idxs,
                                                                                     current_i,
                                                                                     only_previous,
                                                                                     only_same_dataset_id,
                                                                                     allowed_train_and_val)
    
    def produce_targets(self, doc, **kwargs):
        all_annotated_wordboxes_nearest = kwargs.get('all_annotated_wordboxes_nearest')
        cls_extract_type_to_ordered_ids = kwargs.get('cls_extract_type_to_ordered_ids')
        # target similarity matrix:
        target_reuse_similarity = np.zeros((len(doc['wordboxes']['bbox']), len(all_annotated_wordboxes_nearest), 1),
                                           dtype=np.float)
        assert 0 not in target_reuse_similarity.shape, "target reuse similarity shape {}".format(
            target_reuse_similarity.shape)
        for ids, cls_extract_type in zip(doc['annotation']['ids'], doc['annotation']['cls_extract_type']):
            if cls_extract_type in cls_extract_type_to_ordered_ids:
                same_cls_extract_types_i = cls_extract_type_to_ordered_ids[cls_extract_type]
                for i in ids:
                    target_reuse_similarity[i, same_cls_extract_types_i, 0] = 1
            # else this means that the nearest page found something that cannot be transported so well!
        # weights = np.ones_like(target_reuse_similarity, dtype=np.float)
        weights = target_reuse_similarity * self.binary_class_weights[1] + \
                  (1.0 - target_reuse_similarity) * self.binary_class_weights[0]
        targets = np.concatenate([target_reuse_similarity, weights], axis=-1)
        return targets
    
    def expand_data_call(self, *args, **kwargs):
        with closing(self.get_my_db_object()) as db:
            with closing(db.cursor()) as cursor:
                textfields_aabb = (0, 0, PAGE_WIDTH, PAGE_HEIGHT)
                
                phase = kwargs.get('phase')
                learning_phase = phase == 'train'
                
                shuffle_seed, current_i = args[0]
                
                (dataset_id, doc_id, settype, page_n), (
                    nearest_dataset_id, nearest_doc_id, nearest_settype, nearest_page_n) = \
                    self.get_doc_pair(shuffle_seed, current_i, **kwargs)
                
                doc = self.get_doc_data(dataset_id, doc_id, settype, page_n, cursor, textfields_aabb, learning_phase,
                                        self.use_pic)
                nearest_doc = self.get_doc_data(nearest_dataset_id, nearest_doc_id, nearest_settype, nearest_page_n,
                                                cursor, textfields_aabb, learning_phase, self.use_pic)
        
        all_annotated_wordboxes_nearest = sorted(list(set(sum(nearest_doc['annotation']['ids'], []))))
        annotated_id_to_order = {e: i for i, e in enumerate(all_annotated_wordboxes_nearest)}
        
        cls_extract_type_to_annotated_ids = {cls_extract_type: [] for cls_extract_type in nearest_doc['annotation']['cls_extract_type']}
        for i, cls_extract_type in enumerate(nearest_doc['annotation']['cls_extract_type']):
            cls_extract_type_to_annotated_ids[cls_extract_type].extend(nearest_doc['annotation']['ids'][i])
        cls_extract_type_to_ordered_ids = {cls_extract_type: [annotated_id_to_order[x] for x in cls_extract_type_to_annotated_ids[cls_extract_type]]
                                    for cls_extract_type in cls_extract_type_to_annotated_ids}
        
        # we will make more named targets and the model will just select those it wants then.
        #  Can be used to predict cls_extract_types or the similarity matrix
        
        targets = self.produce_targets(doc, all_annotated_wordboxes_nearest=all_annotated_wordboxes_nearest,
                                       cls_extract_type_to_ordered_ids=cls_extract_type_to_ordered_ids, nearest_doc=nearest_doc)
        
        datapoint = [{
            # sources
            'wb-text-features': doc['wordboxes']['text-features'],
            'wb-text-onehot': doc['wordboxes']['text-onehot'],
            'wb-bbox': doc['wordboxes']['bbox'],
            'wb-poso': doc['wordboxes']['posorder'],
            'nearest-wb-text-features': nearest_doc['wordboxes']['text-features'],
            'nearest-wb-text-onehot': nearest_doc['wordboxes']['text-onehot'],
            'nearest-wb-bbox': nearest_doc['wordboxes']['bbox'],
            'nearest-wb-poso': nearest_doc['wordboxes']['posorder'],
            # helpers
            'nearest-reuse-ids': all_annotated_wordboxes_nearest,
            # just for metrics computations outside NN:
            'annotations': doc['annotation'],
            'nearest-annotations': nearest_doc['annotation'],
            'nearest-cls_extract_type-to-ordered-ids': cls_extract_type_to_ordered_ids,
        }]
        if 'pic' in doc:  # === self.use_pic
            datapoint[0]['pic'] = doc['pic']
            datapoint[0]['nearest-pic'] = nearest_doc['pic']
        self.add_targets(datapoint, targets)
        
        if self.pass_cls_extract_types:
            datapoint[0]['nearest-annotated'] = self.produce_ft_onehot(nearest_doc, self.pass_cls_extract_types)
        if self.use_neighbours > 0:
            docnids = np.stack(doc['wordboxes']['neighbours-ids'], axis=0)
            nearnids = np.stack(nearest_doc['wordboxes']['neighbours-ids'], axis=0)
            datapoint[0]['neighbours-ids'] = self.use_neighours_count(docnids, self.use_neighbours)
            datapoint[0]['nearest-neighbours-ids'] = self.use_neighours_count(nearnids, self.use_neighbours)
        
        if self.use_fov_annotated > 0:
            datapoint[0]['fov_ids'] = produce_fov_ids(doc['wordboxes']['bbox'], nearest_doc['wordboxes']['bbox'],
                                                      self.use_fov_annotated)
        
        return self._filter_datapoint(datapoint, **kwargs)
        """
        The network is required to predict matchings from all wordboxes in doc to nonbackground wordboxes in nearest_doc.
        That is a matrix NxM to be predicted. (todo idea - +1 for the prediciton of 'it is a background class'?)
        Also it needs the input to contain the indexes of the (M) boxes in nearest_doc.
        
        Lets aggregate nearest_ based on cls_extract_type (there can be more instances of each class! ) #todo is it really?
        
        todo - if not using softmax we guess it will not perform as well...?
        """
    
    def get_output_types(self, **kwargs):
        otypes = ({
                      'wb-text-features': tf.float32,
                      'wb-text-onehot': tf.float32,
                      'wb-bbox': tf.float32,
                      'wb-poso': tf.float32,
                      'nearest-wb-text-features': tf.float32,
                      'nearest-wb-text-onehot': tf.float32,
                      'nearest-wb-bbox': tf.float32,
                      'nearest-wb-poso': tf.float32,
                      # targets & helpers
                      'nearest-reuse-ids': tf.int32,
                      # just for metrics computations outside NN:
                      'annotations': list,
                      # for future work: make this compatible to tf data dataflow (as soon as it gets the memoryleak fixed)
                      'nearest-annotations': list,
                      'nearest-cls_extract_type-to-ordered-ids': list,
                  },
                  tf.float32,
            # tf.float32,
        )
        if self.use_pic:
            otypes[0]['pic'] = tf.float32
            otypes[0]['nearest-pic'] = tf.float32
        if self.pass_cls_extract_types:
            otypes[0]['nearest-annotated'] = tf.float32
        if self.use_neighbours > 0:
            otypes[0]['neighbours-ids'] = tf.int32
            otypes[0]['nearest-neighbours-ids'] = tf.int32
        if self.use_fov_annotated > 0:
            otypes[0]['fov_ids'] = tf.int32
        return self._filter_datapoint(otypes, **kwargs)
    
    def get_batchpadded_shapes(self, **kwargs):
        txtfeats = features_from_text_len(values_scales=self.TEXT_VALUES_SCALE, scale=self.TEXT_SCALE)
        shapes = ({
                      'wb-text-features': [None, txtfeats],
                      'wb-text-onehot': [None, self.CHARACTERS_LEN, len(default_char_list)],
                      'wb-bbox': [None, 4],
                      'wb-poso': [None, 4],
                      'nearest-wb-text-features': [None, txtfeats],
                      'nearest-wb-text-onehot': [None, self.CHARACTERS_LEN, len(default_char_list)],
                      'nearest-wb-bbox': [None, 4],
                      'nearest-wb-poso': [None, 4],
                      # targets & helpers
                      'nearest-reuse-ids': [None],
                      # just for metrics computations outside NN:
                      'annotations': [None],
                      'nearest-annotations': [None],
                      'nearest-cls_extract_type-to-ordered-ids': [None]
                  },
                  [None, None, 2],
            # [None, None]
        )
        if self.use_pic:
            shapes[0]['pic'] = [None, None, 1]
            shapes[0]['nearest-pic'] = [None, None, 1]
        if self.pass_cls_extract_types:
            shapes[0]['nearest-annotated'] = [None, len(self.pass_cls_extract_types)]
        if self.use_neighbours > 0:
            shapes[0]['neighbours-ids'] = [None, 4 * self.use_neighbours]
            shapes[0]['nearest-neighbours-ids'] = [None, 4 * self.use_neighbours]
        if self.use_fov_annotated > 0:
            shapes[0]['fov_ids'] = [None, self.use_fov_annotated]
        return self._filter_datapoint(shapes, **kwargs)
    
    def get_batchpadded_values(self, **kwargs):
        bvals = ({
                     'wb-text-features': 0.0,
                     'wb-text-onehot': 0.0,
                     'wb-bbox': 0.0,
                     'wb-poso': 0.0,
                     'nearest-wb-text-features': 0.0,
                     'nearest-wb-text-onehot': 0.0,
                     'nearest-wb-bbox': 0.0,
                     'nearest-wb-poso': 0.0,
                     # targets & helpers
                     'nearest-reuse-ids': -1,
                     # this is important to be negative because GatherFromIndices will mask it will zeroes then
                     # just for metrics computations outside NN:
                     'annotations': None,
                     'nearest-annotations': None,
                     'nearest-cls_extract_type-to-ordered-ids': None,
                 },
                 0.0,
            # 0.0
        )
        if self.use_pic:
            bvals[0]['pic'] = 0.0
            bvals[0]['nearest-pic'] = 0.0
        if self.pass_cls_extract_types:
            bvals[0]['nearest-annotated'] = 0.0
        if self.use_neighbours > 0:
            bvals[0]['neighbours-ids'] = -1  # will make tensorflow put there zeroes in our actual gather layer.
            bvals[0]['nearest-neighbours-ids'] = -1
        if self.use_fov_annotated > 0:
            bvals[0]['fov_ids'] = -1
        return self._filter_datapoint(bvals, **kwargs)


class FtypesTrgtDocsTextsSqlite(DocsTextsSqliteBase):
    """
    Produce an unannotated and nearest target but with cls_extract_types as a target value.
    """
    
    def __init__(self, sqlite_source,
                 filter_doctype,
                 df_proc_num, batch_size, df_batches_to_prefetch,
                 shuffle=True,
                 binary_class_weights=(1.0, 1.0),  # for positive ang negative class MUTLIPLIED with provided ft weights
                 limit=None,
                 verbose_progress=True,
                 use_neighbours=0,
                 pass_cls_extract_types='all',
                 use_pic=False,
                 texts_method=None,
                 ft_weights='auto',
                 ):
        assert pass_cls_extract_types is not None
        if (ft_weights not in ['auto', None]):
            assert pass_cls_extract_types != 'all', \
                "if ft_weights are specified and not None, pass cls_extract_types must be specified also"
        DocsTextsSqliteBase.__init__(self, sqlite_source,
                                     filter_doctype,
                                     df_proc_num, batch_size, df_batches_to_prefetch,
                                     shuffle,
                                     binary_class_weights,
                                     limit,
                                     verbose_progress,
                                     use_neighbours,
                                     pass_cls_extract_types,
                                     use_pic,
                                     texts_method
                                     )
        self.ft_weights = ft_weights
    
    @classmethod
    def weights_for_ft_onehot(cls, ft_onehot, binary_class_weights, pass_cls_extract_types, ft_weights):
        ft_w = np.zeros_like(ft_onehot)
        if ft_weights is None:  # only use binary w:
            for id in range(len(ft_onehot)):
                ft_w[ft_onehot[:, id] >= 0.5, id] = binary_class_weights[1]
                ft_w[ft_onehot[:, id] < 0.5, id] = binary_class_weights[0]
        else:
            if not isinstance(ft_weights, np.ndarray):
                ft_weights = np.array(ft_weights)
            assert len(pass_cls_extract_types) == ft_weights.shape[0]
            assert 2 == ft_weights.shape[1]
            for id, weights in enumerate(ft_weights):
                ft_w[ft_onehot[:, id] >= 0.5, id] = weights[1] * binary_class_weights[1]
                ft_w[ft_onehot[:, id] < 0.5, id] = weights[0] * binary_class_weights[0]
        return ft_w
    
    def produce_targets(self, doc, **kwargs):
        ft_targets = self.produce_ft_onehot(doc, self.pass_cls_extract_types)
        ft_weights = self.weights_for_ft_onehot(ft_targets, self.binary_class_weights, self.pass_cls_extract_types,
                                                self.ft_weights)
        targets = np.concatenate([ft_targets, ft_weights], axis=-1)
        return targets
    
    def _compute_index(self):
        DocsTextsSqliteBase._compute_index(self)
        
        if self.ft_weights == 'auto':
            self._compute_cls_extract_type_counts()
            
            if self.ft_weights == 'auto':
                counts_bin = [[self._all_wordboxes - self._all_ft_counts[ft], self._all_ft_counts[ft]]
                              for ft in self.pass_cls_extract_types]
                self.ft_weights = class_weights_from_counts_binary(counts_bin, norm_weights_to_one=True)
                if self.verbose_progress:
                    print("passing automatically computed classweights: {}".format(self.ft_weights))


class FtypesTrgtDocsTextsSqliteSeparated(FtypesTrgtDocsTextsSqlite):
    """
    Produce an unannotated and nearest target but with cls_extract_types as a target value.
    """
    
    def __init__(self, sqlite_source,
                 filter_doctype,
                 df_proc_num, batch_size, df_batches_to_prefetch,
                 shuffle=True,
                 binary_class_weights=(1.0, 1.0),
                 # for positive ang negative class MUTLIPLIED with provided ft weights
                 limit=None,
                 verbose_progress=True,
                 use_neighbours=0,
                 pass_cls_extract_types='all',
                 use_pic=False,
                 texts_method=None,
                 ft_weights='auto',
                 ):
        FtypesTrgtDocsTextsSqlite.__init__(self, sqlite_source,
                                           filter_doctype,
                                           df_proc_num, batch_size, df_batches_to_prefetch,
                                           shuffle,
                                           binary_class_weights,
                                           # for positive ang negative class MUTLIPLIED with provided ft weights
                                           limit,
                                           verbose_progress,
                                           use_neighbours,
                                           pass_cls_extract_types,
                                           use_pic,
                                           texts_method,
                                           ft_weights)
    
    def produce_targets(self, doc, **kwargs):
        ft_targets = self.produce_ft_onehot(doc, self.pass_cls_extract_types)
        
        # as originally in the article:
        ft_weights = multiclass_temporal_class_weights(ft_targets, self.ft_weights)
        assert ft_targets.shape[0] == ft_weights.shape[0] and ft_weights.ndim == 1
        targets = [ft_targets, ft_weights]
        return targets
    
    def get_output_types(self, **kwargs):
        otypes = ({
                      'wb-text-features': tf.float32,
                      'wb-text-onehot': tf.float32,
                      'wb-bbox': tf.float32,
                      'wb-poso': tf.float32,
                      # just for metrics computations outside NN:
                      'annotations': list,
                  },
                  tf.float32, tf.float32
        )
        if self.use_neighbours > 0:
            otypes[0]['neighbours-ids'] = tf.int32
        if self.use_pic:
            otypes[0]['pic'] = tf.float32
        return self._filter_datapoint(otypes, **kwargs)
    
    def get_batchpadded_shapes(self, **kwargs):
        txtfeats = features_from_text_len(values_scales=self.TEXT_VALUES_SCALE, scale=self.TEXT_SCALE)
        shapes = ({
                      'wb-text-features': [None, txtfeats],
                      'wb-text-onehot': [None, self.CHARACTERS_LEN, len(default_char_list)],
                      'wb-bbox': [None, 4],
                      'wb-poso': [None, 4],
                      # just for metrics computations outside NN:
                      'annotations': [None],
                  },
                  [None, None], [None]
        )
        if self.use_neighbours > 0:
            shapes[0]['neighbours-ids'] = [None, 4 * self.use_neighbours]
        if self.use_pic:
            shapes[0]['pic'] = [None, None, 1]
        return self._filter_datapoint(shapes, **kwargs)
    
    def get_batchpadded_values(self, **kwargs):
        bvals = ({
                     'wb-text-features': 0.0,
                     'wb-text-onehot': 0.0,
                     'wb-bbox': 0.0,
                     'wb-poso': 0.0,
                     # just for metrics computations outside NN:
                     'annotations': None,
                 },
                 0.0, 0.0
        )
        if self.use_neighbours > 0:
            bvals[0]['neighbours-ids'] = -1  # will make tensorflow put there zeroes in our actual gather layer.
        if self.use_pic:
            bvals[0]['pic'] = 0.0
        return self._filter_datapoint(bvals, **kwargs)


class DocsTextsSqliteWeightedFtypes(DocsTextsSqliteNearest):
    """
    Produce an unannotated and nearest target but with cls_extract_types as a target value.
    """
    
    def __init__(self, sqlite_source,
                 filter_doctype,
                 df_proc_num, batch_size, df_batches_to_prefetch,
                 shuffle=True,
                 binary_class_weights=(1.0, 1.0),  # for positive ang negative class MUTLIPLIED with provided ft weights
                 limit=None,
                 verbose_progress=True,
                 use_neighbours=0,
                 use_fov_annotated=0,
                 pass_cls_extract_types='all',
                 use_pic=False,
                 texts_method=None,
                 ft_weights='auto',
                 ):
        assert pass_cls_extract_types is not None
        if (ft_weights not in ['auto', None]):
            assert pass_cls_extract_types != 'all', \
                "if ft_weights are specified and not None, pass cls_extract_types must be specified also"
        DocsTextsSqliteNearest.__init__(self, sqlite_source,
                                        filter_doctype,
                                        df_proc_num, batch_size, df_batches_to_prefetch,
                                        shuffle,
                                        binary_class_weights,
                                        limit,
                                        verbose_progress,
                                        use_neighbours,
                                        use_fov_annotated,
                                        pass_cls_extract_types,
                                        use_pic,
                                        texts_method,
                                        )
        self.ft_weights = ft_weights
    
    def _compute_index(self):
        DocsTextsSqliteNearest._compute_index(self)
        
        if self.ft_weights == 'auto':
            self._compute_cls_extract_type_counts()
            
            if self.ft_weights == 'auto':
                counts_bin = [[self._all_wordboxes - self._all_ft_counts[ft], self._all_ft_counts[ft]]
                              for ft in self.pass_cls_extract_types]
                self.ft_weights = class_weights_from_counts_binary(counts_bin, norm_weights_to_one=True)
                if self.verbose_progress:
                    print("passing automatically computed classweights: {}".format(self.ft_weights))
        
        self.ft_weights_dict = {ft: weight for ft, weight in zip(self.pass_cls_extract_types, self.ft_weights)}
    
    def produce_targets(self, doc, **kwargs):
        all_annotated_wordboxes_nearest = kwargs.get('all_annotated_wordboxes_nearest')
        cls_extract_type_to_ordered_ids = kwargs.get('cls_extract_type_to_ordered_ids')
        
        assert len(cls_extract_type_to_ordered_ids.keys()) <= len(
            self.pass_cls_extract_types), "we should pass only the cls_extract_types specified"
        
        # target similarity matrix:
        target_reuse_similarity = np.zeros((len(doc['wordboxes']['bbox']), len(all_annotated_wordboxes_nearest), 1),
                                           dtype=np.float)
        
        weights = np.zeros_like(target_reuse_similarity)
        # initialize to weights of zero-classes:
        for cls_extract_type in cls_extract_type_to_ordered_ids:
            same_cls_extract_types_i = cls_extract_type_to_ordered_ids[cls_extract_type]
            weights[:, same_cls_extract_types_i] = self.ft_weights_dict[cls_extract_type][0] * self.binary_class_weights[0]
        
        assert 0 not in target_reuse_similarity.shape, "target reuse similarity shape {}".format(
            target_reuse_similarity.shape)
        for ids, cls_extract_type in zip(doc['annotation']['ids'], doc['annotation']['cls_extract_type']):
            if cls_extract_type in cls_extract_type_to_ordered_ids:
                same_cls_extract_types_i = cls_extract_type_to_ordered_ids[cls_extract_type]
                for i in ids:
                    target_reuse_similarity[i, same_cls_extract_types_i, 0] = 1
                    weights[i, same_cls_extract_types_i, 0] = self.ft_weights_dict[cls_extract_type][1] * self.binary_class_weights[1]
        
        targets = np.concatenate([target_reuse_similarity, weights], axis=-1)
        return targets


class DocsTextsSqliteWeightedFtypesDebug(DocsTextsSqliteWeightedFtypes):
    """
    Produce an unannotated and nearest target but with cls_extract_types as a target value.
    """

    def produce_targets(self, doc, **kwargs):
        """
        
        THIS IS A DEBUG GENERATOR - PRODUCES 1s for similarity, if the bboxe's texts have the same lengths:
        """
        all_annotated_wordboxes_nearest = kwargs.get('all_annotated_wordboxes_nearest')
        cls_extract_type_to_ordered_ids = kwargs.get('cls_extract_type_to_ordered_ids')
    
        assert len(cls_extract_type_to_ordered_ids.keys()) <= len(
            self.pass_cls_extract_types), "we should pass only the cls_extract_types specified"
    
        # target similarity matrix:
        target_reuse_similarity = np.zeros((len(doc['wordboxes']['bbox']), len(all_annotated_wordboxes_nearest), 1),
                                           dtype=np.float)
    
        weights = np.ones_like(target_reuse_similarity)
    
        assert 0 not in target_reuse_similarity.shape, "target reuse similarity shape {}".format(
            target_reuse_similarity.shape)
        for ibbox, ohtext in enumerate(doc['wordboxes']['text-onehot']):
            for nibbox in range(len(all_annotated_wordboxes_nearest)):
                
                nohtext = kwargs['nearest_doc']['wordboxes']['text-onehot'][all_annotated_wordboxes_nearest[nibbox]]
                assert ohtext.shape == nohtext.shape
                if np.sum(ohtext) == np.sum(nohtext):
                    target_reuse_similarity[ibbox, nibbox, 0] = 1
    
        targets = np.concatenate([target_reuse_similarity, weights], axis=-1)
        return targets
    



class FtypesTrgtDocsTextsSqliteNearest(DocsTextsSqliteWeightedFtypes):
    """
    Produce an unannotated and nearest target but with cls_extract_types as a target value.
    """
    
    def __init__(self, sqlite_source,
                 filter_doctype,
                 df_proc_num, batch_size, df_batches_to_prefetch,
                 shuffle=True,
                 binary_class_weights=(1.0, 1.0),  # for positive ang negative class MUTLIPLIED with provided ft weights
                 limit=None,
                 verbose_progress=True,
                 use_neighbours=0,
                 use_fov_annotated=0,
                 pass_cls_extract_types='all',
                 use_pic=False,
                 texts_method=None,
                 ft_weights='auto',
                 ):
        DocsTextsSqliteWeightedFtypes.__init__(self, sqlite_source,
                                               filter_doctype,
                                               df_proc_num, batch_size, df_batches_to_prefetch,
                                               shuffle,
                                               binary_class_weights,
                                               limit,
                                               verbose_progress,
                                               use_neighbours,
                                               use_fov_annotated,
                                               pass_cls_extract_types,
                                               use_pic,
                                               texts_method,
                                               ft_weights
                                               )
    
    def produce_targets(self, doc, **kwargs):
        ft_targets = self.produce_ft_onehot(doc, self.pass_cls_extract_types)
        ft_weights = FtypesTrgtDocsTextsSqlite.weights_for_ft_onehot(ft_targets, self.binary_class_weights,
                                                                     self.pass_cls_extract_types, self.ft_weights)
        targets = np.concatenate([ft_targets, ft_weights], axis=-1)
        return targets


class FtypesTrgtDocsTextsSqliteNearestSeparated(FtypesTrgtDocsTextsSqliteNearest):
    """
    Produce an unannotated and nearest target but with cls_extract_types as a target value.
    """
    
    def __init__(self, sqlite_source,
                 filter_doctype,
                 df_proc_num, batch_size, df_batches_to_prefetch,
                 shuffle=True,
                 binary_class_weights=(1.0, 1.0),  # for positive ang negative class MUTLIPLIED with provided ft weights
                 limit=None,
                 verbose_progress=True,
                 use_neighbours=0,
                 use_fov_annotated=0,
                 pass_cls_extract_types='all',
                 use_pic=False,
                 texts_method=None,
                 ft_weights='auto',
                 ):
        FtypesTrgtDocsTextsSqliteNearest.__init__(self, sqlite_source,
                                                  filter_doctype,
                                                  df_proc_num, batch_size, df_batches_to_prefetch,
                                                  shuffle,
                                                  binary_class_weights,
                                                  limit,
                                                  verbose_progress,
                                                  use_neighbours,
                                                  use_fov_annotated,
                                                  pass_cls_extract_types,
                                                  use_pic,
                                                  texts_method,
                                                  ft_weights)
    
    def produce_targets(self, doc, **kwargs):
        ft_targets = self.produce_ft_onehot(doc, self.pass_cls_extract_types)
        
        # as originally in the article:
        ft_weights = multiclass_temporal_class_weights(ft_targets, self.ft_weights)
        assert ft_targets.shape[0] == ft_weights.shape[0] and ft_weights.ndim == 1
        targets = [ft_targets, ft_weights]
        return targets
    
    def get_output_types(self, **kwargs):
        otypes = ({
                      'wb-text-features': tf.float32,
                      'wb-text-onehot': tf.float32,
                      'wb-bbox': tf.float32,
                      'wb-poso': tf.float32,
                      'nearest-wb-text-features': tf.float32,
                      'nearest-wb-text-onehot': tf.float32,
                      'nearest-wb-bbox': tf.float32,
                      'nearest-wb-poso': tf.float32,
                      # targets & helpers
                      'nearest-reuse-ids': tf.int32,
                      # just for metrics computations outside NN:
                      'annotations': list,
                      # for future work: make this compatible to tf data dataflow (as soon as it gets the memoryleak fixed)
                      'nearest-annotations': list,
                      'nearest-cls_extract_type-to-ordered-ids': list,
                  },
                  tf.float32, tf.float32,
        )
        if self.use_pic:
            otypes[0]['pic'] = tf.float32
            otypes[0]['nearest-pic'] = tf.float32
        if self.pass_cls_extract_types:
            otypes[0]['nearest-annotated'] = tf.float32
        if self.use_neighbours > 0:
            otypes[0]['neighbours-ids'] = tf.int32
            otypes[0]['nearest-neighbours-ids'] = tf.int32
        if self.use_fov_annotated > 0:
            otypes[0]['fov_ids'] = tf.int32
        return self._filter_datapoint(otypes, **kwargs)
    
    def get_batchpadded_shapes(self, **kwargs):
        txtfeats = features_from_text_len(values_scales=self.TEXT_VALUES_SCALE, scale=self.TEXT_SCALE)
        shapes = ({
                      'wb-text-features': [None, txtfeats],
                      'wb-text-onehot': [None, self.CHARACTERS_LEN, len(default_char_list)],
                      'wb-bbox': [None, 4],
                      'wb-poso': [None, 4],
                      'nearest-wb-text-features': [None, txtfeats],
                      'nearest-wb-text-onehot': [None, self.CHARACTERS_LEN, len(default_char_list)],
                      'nearest-wb-bbox': [None, 4],
                      'nearest-wb-poso': [None, 4],
                      # targets & helpers
                      'nearest-reuse-ids': [None],
                      # just for metrics computations outside NN:
                      'annotations': [None],
                      'nearest-annotations': [None],
                      'nearest-cls_extract_type-to-ordered-ids': [None]
                  },
                  [None, None, 2], [None]
        )
        if self.use_pic:
            shapes[0]['pic'] = [None, None, 1]
            shapes[0]['nearest-pic'] = [None, None, 1]
        if self.pass_cls_extract_types:
            shapes[0]['nearest-annotated'] = [None, len(self.pass_cls_extract_types)]
        if self.use_neighbours > 0:
            shapes[0]['neighbours-ids'] = [None, 4 * self.use_neighbours]
            shapes[0]['nearest-neighbours-ids'] = [None, 4 * self.use_neighbours]
        if self.use_fov_annotated > 0:
            shapes[0]['fov_ids'] = [None, self.use_fov_annotated]
        return self._filter_datapoint(shapes, **kwargs)
    
    def get_batchpadded_values(self, **kwargs):
        bvals = ({
                     'wb-text-features': 0.0,
                     'wb-text-onehot': 0.0,
                     'wb-bbox': 0.0,
                     'wb-poso': 0.0,
                     'nearest-wb-text-features': 0.0,
                     'nearest-wb-text-onehot': 0.0,
                     'nearest-wb-bbox': 0.0,
                     'nearest-wb-poso': 0.0,
                     # targets & helpers
                     'nearest-reuse-ids': -1,
                     # this is important to be negative because GatherFromIndices will mask it will zeroes then
                     # just for metrics computations outside NN:
                     'annotations': None,
                     'nearest-annotations': None,
                     'nearest-cls_extract_type-to-ordered-ids': None,
                 },
                 0.0, 0.0
        )
        if self.use_pic:
            bvals[0]['pic'] = 0.0
            bvals[0]['nearest-pic'] = 0.0
        if self.pass_cls_extract_types:
            bvals[0]['nearest-annotated'] = 0.0
        if self.use_neighbours > 0:
            bvals[0]['neighbours-ids'] = -1  # will make tensorflow put there zeroes in our actual gather layer.
            bvals[0]['nearest-neighbours-ids'] = -1
        if self.use_fov_annotated > 0:
            bvals[0]['fov_ids'] = -1
        return self._filter_datapoint(bvals, **kwargs)


def measure_classcounts(sqlite_source):
    dfobj = DocsTextsSqliteNearest(sqlite_source, 'pdf', df_proc_num=4,
                                   batch_size=1,
                                   df_batches_to_prefetch=4,  # choose at least = df_proc_num of batch size is 1
                                   limit=None)
    df, size = dfobj.dataflow_packer('all')
    df.reset_state()
    
    ones = 0
    total = 0
    
    wordbox_counts = []
    reuse_ids_counts = []
    
    progress = tqdm(df.get_data(), total=size)
    for item in progress:
        ones += sum(sum(item[1][:, :, 0]))
        total += item[1].shape[0] * item[1].shape[1]
        wordbox_counts.append(len(item[0]['wb-bbox']))
        reuse_ids_counts.append(len(item[0]['nearest-reuse-ids']))
    
    print("in total there are {} predicions, {} class 1 and {} class 0".format(total, ones, total - ones))
    print("Wordboxes mean {}, max {}, min {}, median {}".format(np.mean(wordbox_counts),
                                                                np.max(wordbox_counts),
                                                                np.min(wordbox_counts),
                                                                np.median(wordbox_counts)))
    print("Reusing ids mean {}, max {}, min {}, median {}".format(np.mean(reuse_ids_counts),
                                                                  np.max(reuse_ids_counts),
                                                                  np.min(reuse_ids_counts),
                                                                  np.median(reuse_ids_counts)))
    
    """
    Querying index...
    100%|███████████████████████████████████| 31272/31272 [00:25<00:00, 1242.13it/s]
    Index of 42617 items of byte size: 10926320
    (3679 pages dropped due to no content)
    Index computed
    100%|███████████████████████
    
    in total there are 274925766 predicions, 8655430.0 class 1 and 266270336.0 class 0
    (all cls_extract_types:)
    Wordboxes mean 141.71415162963137, max 930, min 1, median 124.0
    Reusing ids mean 45.41680550015252, max 897, min 0, median 34.0
    ^^ because of tables!
    
    after filtering:
    Wordboxes mean 141.76504439329167, max 930, min 1, median 124.0
    Reusing ids mean 32.37990322732184, max 167, min 1, median 32.0

    """


def all_pairs(total):
    for i in range(total):
        for j in range(i):
            yield i, j


def put_item(i, j, embs_all, dists_all2all):
    dst = (embs_all[i] - embs_all[j])
    dists_all2all[i, j] = np.sum(dst * dst)


def all_pairs_batched(total, bsize=5000):
    batch = []
    for i in range(total):
        for j in range(i):
            batch.append([i, j])
            if len(batch) >= bsize:
                yield np.array(batch)
                batch = []
    if len(batch) > 0:
        yield np.array(batch)


def put_item_batched(batch, embs_all, dists_all2all):
    dst = (embs_all[batch[:, 0]] - embs_all[batch[:, 1]])
    dists_all2all[batch[:, 0], batch[:, 1]] = np.sum(dst * dst, axis=-1)


def try_precompute_nearest(sqlite_source):
    dfobj = DocsTextsSqliteNearest(sqlite_source, 'pdf', df_proc_num=2,
                                   batch_size=1,
                                   df_batches_to_prefetch=1, limit=1000)
    dfobj.get_index().precompute_embcache('/tmp/try_numpy_cache', n_jobs=4)
    """
    datakwargs = {'only_previous': 1000}
    times = []

    df = dfobj.get_indices(phase='train')
    df.reset_state()
    progress = tqdm(df.get_data(), total=len(df))
    for item in progress:
        bef = time.time()
        shuffle_seed, current_i = item
        current_i = len(df) - 1
    
        (dataset_id, doc_id, settype, page_n), (
            nearest_dataset_id, nearest_doc_id, nearest_settype, nearest_page_n) = \
            dfobj.get_doc_pair(shuffle_seed, current_i, **datakwargs)
        after = time.time()
    
        times.append(after - bef)

    print("measured to do {} iterations / second".format(progress.n / (progress.last_print_t - progress.start_t)))
    """


def try_find_nearest(sqlite_source):
    dfobj = DocsTextsSqliteNearest(sqlite_source, 'pdf', df_proc_num=2,
                                   batch_size=1,
                                   df_batches_to_prefetch=1, limit=400)
    
    datakwargs = {'only_previous': 1000}
    times = []
    
    df = dfobj.get_indices(phase='train')
    df.reset_state()
    progress = tqdm(df.get_data(), total=len(df))
    for item in progress:
        bef = time.time()
        shuffle_seed, current_i = item
        current_i = len(df) - 1
        
        (dataset_id, doc_id, settype, page_n), (
            nearest_dataset_id, nearest_doc_id, nearest_settype, nearest_page_n) = \
            dfobj.get_doc_pair(shuffle_seed, current_i, **datakwargs)
        after = time.time()
        
        times.append(after - bef)
    
    print("measured to do {} iterations / second".format(progress.n / (progress.last_print_t - progress.start_t)))
    
    # linear correlation between predictions and outputs
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.title('times')
    # plt.axhline(y=0.0, color='black', linestyle='-')
    plt.plot(times, color='red', linewidth=1,
             label='times')
    # plt.legend(loc='upper left', fontsize=30)
    # plt.ylabel('SSE model score', fontsize=30)
    # plt.xlabel('Data corruption percentage', fontsize=30)
    plt.savefig('timesplt.png')


def try_dflow(sqlite_source):
    dfobj = DocsTextsSqliteNearest(sqlite_source, 'pdf', df_proc_num=2,
                                   batch_size=1,
                                   df_batches_to_prefetch=1, limit=400)
    df, size = dfobj.dataflow_packer('train')
    df.reset_state()
    
    progress = tqdm(df.get_data(), total=size)
    x = 0
    for item in progress:
        # print(item)
        # break
        x += 1
        if x > 100:
            break
    print("measured to do {} iterations / second".format(progress.n / (progress.last_print_t - progress.start_t)))


def try_dflow_FtypesTrgtDocsTextsSqlite(sqlite_source):
    dfobj = FtypesTrgtDocsTextsSqliteNearest(sqlite_source, 'pdf', df_proc_num=2,
                                             batch_size=1,
                                             df_batches_to_prefetch=1, limit=400, pass_cls_extract_types='all',
                                             ft_weights='auto',
                                             verbose_progress=True
                                             )
    df, size = dfobj.dataflow_packer('train')
    df.reset_state()
    
    progress = tqdm(df.get_data(), total=size)
    x = 0
    for item in progress:
        # print(item)
        # break
        x += 1
        if x > 100:
            break
    print("measured to do {} iterations / second".format(progress.n / (progress.last_print_t - progress.start_t)))


def run_dflow_epochs(sqlite_source,epochs=8):
    from pympler import asizeof
    dfobj = DocsTextsSqliteNearest(sqlite_source, 'pdf', df_proc_num=8,
                                   batch_size=8,
                                   df_batches_to_prefetch=3, limit=None)
    df, size = dfobj.dataflow_packer('train')
    df.reset_state()
    
    print("total size of dfobj: {} and generator: {}".format(asizeof.asizeof(dfobj), asizeof.asizeof(df)))
    
    for epoch in range(epochs):
        progress = tqdm(df.get_data(), total=size)
        for item in progress:
            continue
        print("measured to do {} iterations / second".format(progress.n / (progress.last_print_t - progress.start_t)))
        print("total size of dfobj: {} and generator: {}".format(asizeof.asizeof(dfobj), asizeof.asizeof(df)))


def run_dflow_final_epochs(sqlite_source, epochs=8):
    from pympler import asizeof
    dfobj = DocsTextsSqliteNearest(sqlite_source, 'pdf', df_proc_num=8,
                                   batch_size=8,
                                   df_batches_to_prefetch=3, limit=None, use_fov_annotated=1)
    df, size = dfobj.get_final_dataflow_dataset('train')
    
    print("total size of dfobj: {} and generator: {}".format(asizeof.asizeof(dfobj), asizeof.asizeof(df)))
    
    for epoch in range(epochs):
        progress = tqdm(df, total=size)
        for item in progress:
            continue
        print("measured to do {} iterations / second".format(progress.n / (progress.last_print_t - progress.start_t)))
        print("total size of dfobj: {} and generator: {}".format(asizeof.asizeof(dfobj), asizeof.asizeof(df)))


def try_tf_dataset(sqlite_source):
    dfobj = DocsTextsSqliteNearest(sqlite_source, 'pdf', 1, 2, 1, limit=400)
    tfds, size = dfobj.get_final_tf_data_dataset('train')
    iteratetf = tf_dataset_as_iterator(tfds)
    
    progress = tqdm(iteratetf, total=size)
    x = 0
    for item in progress:
        # print(item)
        # break
        x += 1
        if x > 100:
            break
    print("measured to do (batch size {}) {} iterations / second".format(dfobj.batch_size,
                                                                         progress.n / (
                                                                                 progress.last_print_t - progress.start_t)))


def try_generator(sqlite_source):
    """
    table_docs_def = create table if not exists docs(dataset text, docid text, url text, npages integer,
                            settype text, doctype text, primary key(dataset, docid));
                            
    table_texts_def = create table if not exists texts(docid text, page integer, itemorder integer,
                          bbox_l float,bbox_t float,bbox_r float,bbox_b float,
                          content text, 
                          row_readings_pos_1 integer, row_readings_pos_2 integer, col_readings_pos_1 integer, col_readings_pos_2 integer,
                          primary key(docid, page, itemorder));
                       
    table_annotations_def = create table if not exists annotations(docid text, page integer, itemorder integer,
                          cls_extract_type integer, content text, texts_list_ids text,
                          primary key (docid, page, itemorder));
    """
    con = sqlite3.connect(sqlite_source, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    
    try:
        # cur.execute("Select count(*) from texts")
        cur.execute("Select * from docs where settype=='train' and doctype=='pdf' limit 2")
        bef = time.time()
        records = cur.fetchall()
        after = time.time()
        print("Total rows are:  ", len(records))
        print("Total time to fetch: ", after - bef)
        
        cur.close()
    
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if con:
            con.close()
            print("The SQLite connection is closed")


def count_wordboxes_total(sqlite_source):
    """
    table_docs_def = create table if not exists docs(dataset text, docid text, url text, npages integer,
                            settype text, doctype text, primary key(dataset, docid));

    table_texts_def = create table if not exists texts(docid text, page integer, itemorder integer,
                          bbox_l float,bbox_t float,bbox_r float,bbox_b float,
                          content text,
                          row_readings_pos_1 integer, row_readings_pos_2 integer, col_readings_pos_1 integer, col_readings_pos_2 integer,
                          primary key(docid, page, itemorder));

    table_annotations_def = create table if not exists annotations(docid text, page integer, itemorder integer,
                          cls_extract_type integer, content text, texts_list_ids text,
                          primary key (docid, page, itemorder));
    """
    con = sqlite3.connect(sqlite_source, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    
    try:
        # cur.execute("Select count(*) from texts")
        # per doc: Select count(texts.itemorder) as cnt, docs.docid from texts,
        # docs group by docs.docid where (docs.docid == texts.docid) and (docs.doctype=='pdf')"
        # total:
        allcount = cur.execute("Select count(texts.itemorder) as cnt, count(distinct docs.docid) from texts,"
                               " docs where (docs.docid == texts.docid) and (docs.doctype=='pdf')").fetchall()
        print(allcount)
        
        cur.close()
    
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if con:
            con.close()
            print("The SQLite connection is closed")


def count_wordboxes_per_cls_extract_type(sqlite_source):
    """
    table_docs_def = create table if not exists docs(dataset text, docid text, url text, npages integer,
                            settype text, doctype text, primary key(dataset, docid));

    table_texts_def = create table if not exists texts(docid text, page integer, itemorder integer,
                          bbox_l float,bbox_t float,bbox_r float,bbox_b float,
                          content text,
                          row_readings_pos_1 integer, row_readings_pos_2 integer, col_readings_pos_1 integer, col_readings_pos_2 integer,
                          primary key(docid, page, itemorder));

    table_annotations_def = create table if not exists annotations(docid text, page integer, itemorder integer,
                          cls_extract_type integer, content text, texts_list_ids text,
                          primary key (docid, page, itemorder));
    """
    con = sqlite3.connect(sqlite_source, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    
    a_counts_per_ft = defaultdict(list)
    
    try:
        # cur.execute("Select count(*) from texts")
        allpdfs = cur.execute("Select * from docs where doctype=='pdf'").fetchall()
        allpdfids = {item[1] for item in allpdfs}
        
        annotations_total = cur.execute(
            "Select count(*) from annotations").fetchone()[0]
        
        annotations_on_page = cur.execute(
            "Select * from annotations where texts_list_ids!='[]' order by docid, page, itemorder asc")
        
        for annotation in tqdm(annotations_on_page, total=annotations_total):
            if annotation[0] in allpdfids:  # is docid pdf?
                cls_extract_type = annotation[-3]
                ids_wordboxes = json.loads(annotation[-1])
                a_counts_per_ft[cls_extract_type].append(len(ids_wordboxes))
        
        cur.close()
        
        for cls_extract_type in a_counts_per_ft:
            counts = a_counts_per_ft[cls_extract_type]
            print("{}: \t  min {},\t mean {}, \t median {}, \t max {} \t total {} \t totally in docs {}".
                  format(cls_extract_type, np.min(counts), np.mean(counts), np.median(counts),
                         np.max(counts), np.sum(counts), len(counts)))
    
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if con:
            con.close()
            print("The SQLite connection is closed")
    """
    100%|█████████████████████████████| 2374704/2374704 [00:10<00:00, 236950.57it/s]
    bank_num: 	  min 1,	 mean 1.101681311253802, 	 median 1.0, 	 max 5 	 total 26079 	 totally in docs 23672
    date_issue: 	  min 1,	 mean 1.177041139676504, 	 median 1.0, 	 max 6 	 total 42716 	 totally in docs 36291
    sender_name: 	  min 1,	 mean 1.968275495279883, 	 median 2.0, 	 max 16 	 total 74017 	 totally in docs 37605
    account_num: 	  min 1,	 mean 1.1384656714126722, 	 median 1.0, 	 max 16 	 total 29665 	 totally in docs 26057
    tax_detail_base: 	  min 1,	 mean 1.0977286644769804, 	 median 1.0, 	 max 4 	 total 39437 	 totally in docs 35926
    date_due: 	  min 1,	 mean 1.1484052939900196, 	 median 1.0, 	 max 7 	 total 26465 	 totally in docs 23045
    sender_dic: 	  min 1,	 mean 1.0739916876426858, 	 median 1.0, 	 max 8 	 total 18347 	 totally in docs 17083
    var_sym: 	  min 1,	 mean 1.0790346210193538, 	 median 1.0, 	 max 10 	 total 13994 	 totally in docs 12969
    tax_detail_total: 	  min 1,	 mean 1.1249949777009924, 	 median 1.0, 	 max 6 	 total 28000 	 totally in docs 24889
    tax_detail_rate: 	  min 1,	 mean 1.0476791638530871, 	 median 1.0, 	 max 3 	 total 35685 	 totally in docs 34061
    amount_total_tax: 	  min 1,	 mean 1.0831276848551321, 	 median 1.0, 	 max 6 	 total 23701 	 totally in docs 21882
    sender_addrline: 	  min 1,	 mean 2.03876977590765, 	 median 2.0, 	 max 14 	 total 147295 	 totally in docs 72247
    recipient_ic: 	  min 1,	 mean 1.1152362584378013, 	 median 1.0, 	 max 5 	 total 16191 	 totally in docs 14518
    tax_detail_tax: 	  min 1,	 mean 1.0797632124984744, 	 median 1.0, 	 max 4 	 total 35386 	 totally in docs 32772
    recipient_dic: 	  min 1,	 mean 1.0399717014503007, 	 median 1.0, 	 max 4 	 total 14700 	 totally in docs 14135
    phone_num: 	  min 1,	 mean 1.6332049004452853, 	 median 1.0, 	 max 14 	 total 70788 	 totally in docs 43343
    order_id: 	  min 1,	 mean 1.1268500523247122, 	 median 1.0, 	 max 21 	 total 30150 	 totally in docs 26756
    sender_ic: 	  min 1,	 mean 1.1820368143557487, 	 median 1.0, 	 max 7 	 total 25558 	 totally in docs 21622
    recipient_name: 	  min 1,	 mean 1.7374038157478655, 	 median 1.0, 	 max 9 	 total 65931 	 totally in docs 37948
    amount_total: 	  min 1,	 mean 1.1250795855761995, 	 median 1.0, 	 max 41 	 total 38876 	 totally in docs 34554
    const_sym: 	  min 1,	 mean 1.0909964315124898, 	 median 1.0, 	 max 3 	 total 7949 	 totally in docs 7286
    invoice_id: 	  min 1,	 mean 1.042090156220591, 	 median 1.0, 	 max 10 	 total 44293 	 totally in docs 42504
    date_uzp: 	  min 1,	 mean 1.1501219087425985, 	 median 1.0, 	 max 6 	 total 16510 	 totally in docs 14355
    recipient_addrline: 	  min 1,	 mean 1.8294614034094128, 	 median 2.0, 	 max 13 	 total 161514 	 totally in docs 88285
    amount_total_base: 	  min 1,	 mean 1.0863448049550843, 	 median 1.0, 	 max 6 	 total 26484 	 totally in docs 24379
    amount_due: 	  min 1,	 mean 1.1329333025617354, 	 median 1.0, 	 max 48 	 total 39272 	 totally in docs 34664
    recipient_address: 	  min 1,	 mean 4.656514026278601, 	 median 4.0, 	 max 29 	 total 154871 	 totally in docs 33259
    sender_address: 	  min 1,	 mean 3.983216148786573, 	 median 4.0, 	 max 29 	 total 140496 	 totally in docs 35272
    terms: 	  min 1,	 mean 2.1470148395260553, 	 median 1.0, 	 max 21 	 total 18664 	 totally in docs 8693
    recipient_vat_id: 	  min 1,	 mean 1.0797584017345516, 	 median 1.0, 	 max 13 	 total 6972 	 totally in docs 6457
    sender_vat_id: 	  min 1,	 mean 1.1692812550024012, 	 median 1.0, 	 max 5 	 total 14609 	 totally in docs 12494
    sender_other_addrline: 	  min 1,	 mean 2.171418260555756, 	 median 2.0, 	 max 13 	 total 24068 	 totally in docs 11084
    sender_other_name: 	  min 1,	 mean 1.610406937958639, 	 median 1.0, 	 max 30 	 total 12070 	 totally in docs 7495
    sender_other_address: 	  min 1,	 mean 3.3687357630979498, 	 median 3.0, 	 max 36 	 total 23662 	 totally in docs 7024
    page_total: 	  min 1,	 mean 1.0449403852033017, 	 median 1.0, 	 max 109 	 total 10254 	 totally in docs 9813
    spec_sym: 	  min 1,	 mean 1.0856924254016833, 	 median 1.0, 	 max 8 	 total 1419 	 totally in docs 1307
    page_current: 	  min 1,	 mean 1.0263466805791053, 	 median 1.0, 	 max 3 	 total 15738 	 totally in docs 15334
    bic: 	  min 1,	 mean 1.146814662811305, 	 median 1.0, 	 max 12 	 total 24590 	 totally in docs 21442
    iban: 	  min 1,	 mean 1.438227925987782, 	 median 1.0, 	 max 24 	 total 32724 	 totally in docs 22753
    amount_rounding: 	  min 1,	 mean 1.057414291615035, 	 median 1.0, 	 max 5 	 total 5120 	 totally in docs 4842
    recipient_other_addrline: 	  min 1,	 mean 1.7998270109971581, 	 median 2.0, 	 max 14 	 total 14566 	 totally in docs 8093
    recipient_other_address: 	  min 1,	 mean 4.567793773016405, 	 median 4.0, 	 max 30 	 total 13644 	 totally in docs 2987
    currency_code: 	  min 1,	 mean 1.0282861896838602, 	 median 1.0, 	 max 3 	 total 6798 	 totally in docs 6611
    customer_id: 	  min 1,	 mean 1.0581084764240938, 	 median 1.0, 	 max 11 	 total 19411 	 totally in docs 18345
    recipient_delivery_addrline: 	  min 1,	 mean 1.8949878738884398, 	 median 2.0, 	 max 11 	 total 23441 	 totally in docs 12370
    recipient_other_name: 	  min 1,	 mean 1.8509299781181618, 	 median 1.0, 	 max 9 	 total 6767 	 totally in docs 3656
    recipient_delivery_name: 	  min 1,	 mean 1.8831615120274914, 	 median 2.0, 	 max 10 	 total 8768 	 totally in docs 4656
    document_ref_num: 	  min 1,	 mean 1.0627705627705628, 	 median 1.0, 	 max 9 	 total 491 	 totally in docs 462
    recipient_delivery_address: 	  min 1,	 mean 4.902331720193577, 	 median 4.0, 	 max 18 	 total 22286 	 totally in docs 4546
    tax_detail_currency_tax: 	  min 1,	 mean 1.0323325635103926, 	 median 1.0, 	 max 2 	 total 894 	 totally in docs 866
    tax_detail_currency_rate: 	  min 1,	 mean 1.0184994861253853, 	 median 1.0, 	 max 3 	 total 991 	 totally in docs 973
    amount_paid: 	  min 1,	 mean 1.032381380706094, 	 median 1.0, 	 max 3 	 total 4591 	 totally in docs 4447
    internal_notes: 	  min 1,	 mean 3.0, 	 median 3.0, 	 max 5 	 total 12 	 totally in docs 4
    tax_detail_currency_base: 	  min 1,	 mean 1.0463659147869675, 	 median 1.0, 	 max 3 	 total 835 	 totally in docs 798
    tax_detail_currency_total: 	  min 1,	 mean 1.0502873563218391, 	 median 1.0, 	 max 3 	 total 731 	 totally in docs 696
    
    maybe ignore:
    line_item_amount_base: 	  min 1,	 mean 1.0256147540983607, 	 median 1.0, 	 max 3 	 total 12012 	 totally in docs 11712
    line_item_amount_total_base: 	  min 1,	 mean 1.0454506766533322, 	 median 1.0, 	 max 78 	 total 12283 	 totally in docs 11749
    line_item_description: 	  min 1,	 mean 2.239815242494226, 	 median 2.0, 	 max 21 	 total 48492 	 totally in docs 21650
    line_item_code: 	  min 1,	 mean 1.0737327188940091, 	 median 1.0, 	 max 9 	 total 8621 	 totally in docs 8029
    line_item_uom: 	  min 1,	 mean 1.0198665893271461, 	 median 1.0, 	 max 15 	 total 7033 	 totally in docs 6896
    line_item_amount_total: 	  min 1,	 mean 1.036768384192096, 	 median 1.0, 	 max 3 	 total 8290 	 totally in docs 7996
    line_item_other: 	  min 1,	 mean 1.2454911433172302, 	 median 1.0, 	 max 14 	 total 15469 	 totally in docs 12420
    line_item_amount: 	  min 1,	 mean 1.0131524866420059, 	 median 1.0, 	 max 2 	 total 2465 	 totally in docs 2433
    line_item_tax: 	  min 1,	 mean 1.032114624505929, 	 median 1.0, 	 max 2 	 total 4178 	 totally in docs 4048
    line_item_quantity: 	  min 1,	 mean 1.0176691452550983, 	 median 1.0, 	 max 3 	 total 13823 	 totally in docs 13583
    line_item_rate: 	  min 1,	 mean 1.0132061628760087, 	 median 1.0, 	 max 2 	 total 8286 	 totally in docs 8178
    
    definitely ban:
    table_body: 	  min 1,	 mean 75.06005685856432, 	 median 43.0, 	 max 897 	 total 422438 	 totally in docs 5628
    table_header: 	  min 1,	 mean 10.08849557522124, 	 median 9.0, 	 max 473 	 total 59280 	 totally in docs 5876
    table_footer: 	  min 1,	 mean 6.768347478781827, 	 median 4.0, 	 max 87 	 total 13557 	 totally in docs 2003
    
    
    we have in total:
    [(6394295, 31272)] wordboxes and docs.
    """


if __name__ == '__main__':
    # try_precompute_nearest()
    # try_find_nearest()
    # try_dflow_FtypesTrgtDocsTextsSqlite()
    try_generator()
    # try_dflow()
    # try_tf_dataset()
    # measure_classcounts()
    # count_wordboxes_per_cls_extract_type()
    # run_dflow_epochs()
    # run_dflow_final_epochs()
    # count_wordboxes_total()
