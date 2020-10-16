#
# COPYRIGHT Martin Holecek 2019
#

import sqlite3
from collections import defaultdict

import click
import random
import io
import numpy as np
import os
from tqdm import tqdm

try:
    from utils.boxgeometry import produce_annotations_for_page
except:
    print("skipping")

"""
EXAMPLE FILE ON HOW TO CREATE AND FILL THE SQLITE DATABASE FROM A CUSTOM FORMAT.

This script is the basis of converting a custom dataset into the sqlite cache from which thetraining is run.

As a bridge to a priprietary code, here are provided function stubs, that need to be filled with Your data.
"""

class ModelBase:
    def __init__(self):
        raise AttributeError('Please provide an implementation to the database')

def load_pdf(docid):
    raise AttributeError('Please provide an implementation to the database')

def streaming_dataset_from_file(filename, load_annotation):
    raise AttributeError('Please provide an implementation to the database')

def train_val_split(ds):
    raise AttributeError('Please provide an implementation to the database')

def doc_info(doc):
    raise AttributeError('Please provide an implementation to the database')

def pic_data_for_page(doc):
    raise AttributeError('Please provide an implementation to the database')

def doc_type(doc):
    raise AttributeError('Please provide an implementation to the database')


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


@click.command()
@click.option('--sqlite_target', default="new.sqlite", )
@click.option('--embedding_model', default=None )
@click.option('--verbose', is_flag=True )
def add_embeddings_to_pages(sqlite_target,
                            embedding_model,
                            verbose,
                            ):
    table_pages_def = """create table if not exists pages(docid text, ipage integer,
                        embedding array, primary key(docid, ipage));
                      """
    
    emodel = ModelBase.load(embedding_model)
    emodel.build()
    
    con = sqlite3.connect(sqlite_target, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute(table_pages_def)
    
    cur.execute("Select * from docs")
    records = cur.fetchall()
    for record in tqdm(records, disable=not verbose):  # add all pages in range of pages...
        ds_id = record[0]
        doc_id = record[1]
        
        # we need just the image for embeddings
        document = load_pdf(doc_id)
        with document as doc:
            for page_i in range(record[3]):
                page = doc.page(page_i)
                page_embed = emodel.embed_page(page)
                insert = (doc_id, page_i, page_embed)
                cur.execute(
                    "insert into pages (docid, ipage, embedding) values (?, ?, ?)",
                    insert)
    cur.close()
    con.commit()
    con.close()


@click.command()
@click.option('--sqlite_target', default="new.sqlite", )
@click.option('--ds_filename', default=None, )
@click.option('--verbose', is_flag=True)
def store_pic_data(sqlite_target, ds_filename, verbose):
    pics_def = """create table if not exists pics(docid text, ipage integer,
                        pic_array array, primary key(docid, ipage));
                      """
    
    con = sqlite3.connect(sqlite_target, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute(pics_def)
    
    ds = streaming_dataset_from_file(ds_filename, load_labels=False)
    
    for document in tqdm(ds, disable=not verbose):
        with document as doc:
            n_pages = doc.page_count()
            doc_id = doc.id()
            
            for page_i in range(n_pages):
                with doc.page(page_i) as page:
                    xydata = pic_data_for_page(page)
                    xydata = (xydata * 255).astype(np.uint8)
                    
                    insert = (doc_id, page_i, xydata)
                    cur.execute(
                        "insert into pics (docid, ipage, pic_array) values (?, ?, ?)",
                        insert)
    
    cur.close()
    con.commit()
    con.close()


@click.command()
@click.option('--sqlite_target', default="new.sqlite", )
@click.option('--sqlite_erase', is_flag=True )
@click.option('--ds_filename', default=None, )
@click.option('--use_neighbours', default=3)
@click.option('--verbose', is_flag=True )
def dataset_to_sqlite(sqlite_target,
                      sqlite_erase,
                      ds_filename,
                      use_neighbours,
                      verbose,
                      ):
    if sqlite_erase and os.path.exists(sqlite_target):
        os.remove(sqlite_target)
    
    table_docs_def = """create table if not exists docs(dataset text, docid text, url text, npages integer,
                            settype text, doctype text, primary key(dataset, docid));
                     """
    table_texts_def = """create table if not exists texts(docid text, page integer, itemorder integer,
                          bbox_l float,bbox_t float,bbox_r float,bbox_b float,
                          content text, 
                          row_readings_pos_1 integer, row_readings_pos_2 integer,
                          col_readings_pos_1 integer, col_readings_pos_2 integer,
                          neighbours_ids array,
                          primary key(docid, page, itemorder));
                       """
    table_annotations_def = """create table if not exists annotations(docid text, page integer, itemorder integer,
                          cls_extract_type integer, content text, texts_list_ids text,
                          primary key (docid, page, itemorder));
                       """
    con = sqlite3.connect(sqlite_target, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute(table_docs_def)
    cur.execute(table_texts_def)
    cur.execute(table_annotations_def)
    
    ds = streaming_dataset_from_file(ds_filename, load_labels=True)
    ds_train, ds_val = train_val_split(ds)
    
    for type, ds in zip(['train', 'val'], [ds_train, ds_val]):
        for doc in tqdm(ds, disable=not verbose):
            n_pages = doc.page_count()
            insert = (
                ds_filename, doc.id(), doc_info(doc), n_pages, type,
                doc_type(doc))
            cur.execute("insert into docs (dataset, docid, url, npages, settype, doctype) values (?, ?, ?, ?, ?, ?)",
                        insert)
            
            """
            if n_pages > 1:
                processed_layouts = Parallel(n_jobs=min(4, n_pages), verbose=0)(
                    delayed(produce_annotations_for_page)(doc, page_i, [0.5, 0.5], 0.2)
                    for page_i in range(n_pages)
                )
            """
            
            for page_n in range(n_pages):
                # here we can create wordboxes if not present!
                
                # if n_pages <= 1:
                texts, annotations = produce_annotations_for_page(doc, page_n, reading_percents=[0.5, 0.5],
                                                                  overlap_classes_percents=0.2,
                                                                  use_neighbours=use_neighbours)
                # else:
                #    texts, annotations = processed_layouts[page_n]
                
                for i, annotbox in enumerate(annotations):
                    insert = (
                        doc.id(), page_n, i, annotbox['cls_extract_type'], annotbox['text'],
                        str(annotbox['covered_wordboxes']))
                    cur.execute("insert into annotations (docid, page, itemorder, cls_extract_type, content, texts_list_ids)"
                                " values (?, ?, ?, ?, ?, ?)",
                                insert)
                
                for i, wordbox in enumerate(texts):
                    insert = ([doc.id(), page_n, i]
                              + list(wordbox['bbox']) + [wordbox['text']] +
                              list(wordbox['row_readings_pos']) + list(wordbox['col_readings_pos'])
                              + [wordbox['neighbours']]
                              )
                    cur.execute("insert into texts (docid, page, itemorder, "
                                "bbox_l, bbox_t, bbox_r, bbox_b, "
                                "content, "
                                "row_readings_pos_1, row_readings_pos_2, col_readings_pos_1, col_readings_pos_2,"
                                "neighbours_ids) "
                                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                insert)
    
    cur.close()
    con.commit()
    con.close()


@click.command()
@click.option('--sqlite_target', default="new.sqlite", )
@click.option('--verbose', is_flag=True)
def sqlite_split_val_test(sqlite_target,
                      verbose,
                      ):
    con = sqlite3.connect(sqlite_target, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    alldocs = cur.execute("select docid, settype, doctype from docs").fetchall()
    o_vals = [doc for doc in alldocs if doc[1] in ['val', 'test']]
    #    o_trains = [doc for doc in alldocs if doc[1] == 'train']

    split = int(len(o_vals) / 2)
    print(split)

    cur.execute("update docs set settype=='val' where settype=='test'")
    for doc in o_vals[split:]:
        cur.execute("update docs set settype=='test' where docid=(?)",(doc[0], ))

    updated = cur.execute("select count(*) from docs where settype == 'test'").fetchall()[0]
   
    print(updated)
    cur.close()
    con.commit()
    con.close()


@click.command()
@click.option('--sqlite_target', default="new.sqlite", )
@click.option('--verbose', is_flag=True)
def sqlite_anonymize_texts(sqlite_target,
                          verbose,
                          ):
    con = sqlite3.connect(sqlite_target, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    uqds = cur.execute(f"select DISTINCT(dataset) from docs").fetchall()
    print(uqds)
    
    alldocs = cur.execute("select content, rowid from annotations").fetchall()
    allchars = set(list("".join([text[0] for text in alldocs])))
    print(allchars)
    
    if len(allchars) > 2:
        for text, rowid in tqdm(alldocs):
            parts = text.split(" ")
            new_parts = ["a"*len(part) for part in parts]
            new = " ".join(new_parts)
    
            cur.execute("update annotations set content=? where rowid=?", [new, rowid])
            con.commit()

    cur.execute(f"update docs set dataset='2020'")
    cur.execute(f"update docs set url = ''")
    con.commit()

    alltexts = cur.execute("select content, rowid from texts").fetchall()

    allchars = set(list("".join([text[0] for text in alltexts])))
    print(allchars)
    if len(allchars) > 2:
        for text, rowid in tqdm(alltexts):
            parts = text.split(" ")
            new_parts = ["a"*len(part) for part in parts]
            new = " ".join(new_parts)
    
            cur.execute("update texts set content=? where rowid=?", [new, rowid])
            con.commit()
        
    cur.execute("VACUUM;")
    cur.close()
    con.commit()
    con.close()


@click.command()
@click.option('--sqlite_target', default="new.sqlite", )
@click.option('--verbose', is_flag=True)
def sqlite_super_anonymize_texts(sqlite_target,
                           verbose,
                           ):
    con = sqlite3.connect(sqlite_target, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    
    uqds = cur.execute(f"select DISTINCT(dataset) from docs").fetchall()
    print(uqds)
    
    cur.execute("update annotations set content=''")
    con.commit()
    
    cur.execute(f"update docs set dataset='2020'")
    cur.execute(f"update docs set url = ''")
    con.commit()
    
    """
    Specic anonymization technique is kept proprietal.
    """
    
    print(allchars)
    
    con.commit()
    cur.execute("VACUUM;")
    cur.close()
    con.commit()
    con.close()

input_char_list = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.-+:/%?$£€#()&\''
lowers = u'abcdefghijklmnopqrstuvwxyz'
uppers = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
digits = u'0123456789'
specials = u",.-+:/%?$£€#()&\'"

def range_by_char(char):
    if char.isdigit():
        return digits
    elif char.isalpha():
        if char.islower():
            return lowers
        elif char.isupper():
            return uppers
        else:
            return lowers
    else:
        return specials

import random

def anonymize(text):
    """The actual anonymization"""
    return u"".join([random.choice(range_by_char(char)) for char in text])


def ask_memory_anonymize(text, memory):
    """Solves saving anonymizations to memory if missing & querying the memory"""
    if text not in memory:
        memory[text] = anonymize(text)
    return memory[text]

def anonymize_memorize(text, memory):
    """Solves tokenization - split based on whitespace"""
    parts = text.split(" ")
    new_parts = [ask_memory_anonymize(part, memory) for part in parts]
    return " ".join(new_parts)

@click.command()
@click.option('--sqlite_target', default="new.sqlite", )
@click.option('--verbose', is_flag=True)
def sqlite_anonymize_texts_as_dict(sqlite_target,
                           verbose, debug=True
                           ):
    con = sqlite3.connect(sqlite_target, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    cur.execute(f"update docs set dataset='2020'")
    cur.execute(f"update docs set url = ''")
    
    uqds = cur.execute(f"select DISTINCT(dataset) from docs").fetchall()
    print(uqds)
    
    alldocs = cur.execute("select content, rowid from annotations").fetchall()
    allchars = set(list("".join([text[0] for text in alldocs])))
    print(allchars)
    
    memory = {}
    if len(allchars) > 2:
        for text, rowid in tqdm(alldocs):
            new = anonymize_memorize(text, memory)
            cur.execute("update annotations set content=? where rowid=?", [new, rowid])
    
    alltexts = cur.execute("select content, rowid from texts").fetchall()
    
    allchars = set(list("".join([text[0] for text in alltexts])))
    print(allchars)
    if len(allchars) > 2:
        for text, rowid in tqdm(alltexts):
            new = anonymize_memorize(text, memory)
            
            cur.execute("update texts set content=? where rowid=?", [new, rowid])
    
    if not debug:
        con.commit()
        cur.execute("VACUUM;")
    cur.close()

    if not debug:
        con.commit()
    con.close()


@click.group()
def clisap():
    pass


clisap.add_command(dataset_to_sqlite)
clisap.add_command(add_embeddings_to_pages)
clisap.add_command(store_pic_data)
clisap.add_command(sqlite_split_val_test)
clisap.add_command(sqlite_anonymize_texts)
clisap.add_command(sqlite_anonymize_texts_as_dict)
clisap.add_command(sqlite_super_anonymize_texts)

if __name__ == "__main__":
    clisap()
