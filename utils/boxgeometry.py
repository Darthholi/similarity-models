#
# COPYRIGHT Martin Holecek 2019
#

import numpy as np

from utils.generic_utils import equal_ifarray


class BoundingBox:
    """Provides some basic operations with bounding box edges."""
    # edges - indexes in the common bbox tuple:
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    
    # axes
    HORIZONTAL = 0
    VERTICAL = 1


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


def filter_seen_boxes(viewer, world, key_f=None,
                      ortho_filter=True, max_sqr_dist=None,
                      order_by_dist=True, max_slots=3):
    '''
    Unoptimized function for fast prototyping of experiments.
    '''
    vbbox = key_f(viewer)
    filter = []
    for item in world:
        ibbox = key_f(item)
        if equal_ifarray(ibbox, vbbox) or viewer == item:
            continue  # lets skip the bbox itself
        x_dim_prop, y_dim_prop, edge_id = analyze_bboxes_positions(vbbox, ibbox)
        if ortho_filter:
            if min(x_dim_prop[0], y_dim_prop[0]) > 0:  # does not overlap in strict projection
                continue
        x = max(x_dim_prop[0], 0)
        y = max(y_dim_prop[0], 0)
        dist = x * x + y * y
        if max_sqr_dist is not None:
            if dist > max_sqr_dist:
                continue
        filter.append((item, edge_id, dist))
    if order_by_dist:  # asc
        filter.sort(key=lambda x: x[-1])
    bboxes_by_slots = {0: [], 1: [], 2: [], 3: []}
    for item in filter:
        if max_slots is None or len(bboxes_by_slots[item[1]]) < max_slots:
            bboxes_by_slots[item[1]].append(item)
        else:
            if all([len(bboxes_by_slots[slot]) >= max_slots for slot in bboxes_by_slots]):
                break
    return bboxes_by_slots


def range_overlap_size(a_min, a_max, b_min, b_max):
    assert a_min <= a_max and b_min <= b_max, str((a_min, a_max, b_min, b_max))
    return (min(a_max, b_max) - max(a_min, b_min))


def range_overlap_percent(a_min, a_max, b_min, b_max):
    diff = (a_max - a_min)
    if diff <= 0:
        return 0.0
    return range_overlap_size(a_min, a_max, b_min, b_max) / diff


def range_overlap_percent_min(a_min, a_max, b_min, b_max, zero_limit=True):
    min_d = min(a_max - a_min, b_max - b_min)
    if min_d <= 0.0001:
        if zero_limit:
            return 0.0
        else:
            return range_overlap_size(a_min, a_max, b_min, b_max)
    return range_overlap_size(a_min, a_max, b_min, b_max) / min_d


def produce_ordered_and_overlapping_chunks(items, key_f=None, percent=0.5, dim=1):
    """
    Produces items in chunks, where a chunk is defined as - all items, whose bboxes are overlapped by something else
     (bbox or chunked thing) by more than given number of percent.
    """
    # take care to call it only with filtered labels
    if key_f is None:
        def key_f(x):
            return x.bbox
    
    items = sorted(items, key=lambda bb: key_f(bb)[dim])  # should copy it and not modify original...
    chunk_min = None
    chunk_max = None
    last_produced_idx = -1
    
    for i, item in enumerate(items):
        bb = key_f(item)
        assert bb[dim] <= bb[dim + 2]
        # ^^ we have some assumptions on what should the key_f do. Assert because it can be optimized out on OK code.
        if None in [chunk_min, chunk_max]:
            chunk_min = bb[dim]
            chunk_max = bb[dim + 2]
        elif range_overlap_percent(bb[dim], bb[dim + 2], chunk_min, chunk_max) > percent:
            chunk_min = min(bb[dim], chunk_min)
            chunk_max = max(bb[dim + 2], chunk_max)
        else:
            # found first from the top view, that does NOT overlap
            chunk_min = bb[dim]
            chunk_max = bb[dim + 2]
            yield items[last_produced_idx + 1: i]  # yields item objects, ordered from top to bottom (by their bboxes)!
            last_produced_idx = i - 1
    yield items[last_produced_idx + 1:]


def search_lines_in_bboxes(items, key_f=None, percent_thr=0.2, otherdim=BoundingBox.HORIZONTAL, method_soft=True):
    """
    When we have a chunk of items (defined geometrically by their bboxes got by calling key_f(item)), then we want to
    search for line-reading items by the following premise:
    method_soft: True:
    - Each item has 2 its own left and right neighbours. The first neighbours are the closest items on the sides of the item,
     where their dimensions (specified by otherdim) do overlap by more than percent_thr AND for the second neighbour, it is
     the item, that has the biggest overlap (if bigger  than threshold).
     method_soft: False:
     - Each item has 1 its own left and right neighbours. The neighbour is
     the item, that has the biggest overlap (if bigger than threshold).
    - This defines a graph,  where each node connects up to 4 items on each side and in the graph we need to find lines
     as connected components.

    Thee logic with the 4 neighbours was selected after a bit of exploration, where the closest items should matter,
    but the item should also have a chance to be connected to something very far if aligned nicely.

    items, key_f - see  get_items_reading_order
    percent_thr: when two labels do overlap on a projection to otherdim axis by more than this threshold, they are considered
     to be neighbouring (at least as candidates).
    Call on otherdim-ordered bboxes only!
    """
    assert otherdim in [BoundingBox.HORIZONTAL, BoundingBox.VERTICAL]
    
    if key_f is None:
        def key_f(x):
            return x.bbox
    
    # definition of the graph is in connections, while group is the final group id decision found by the DFS
    bboxes_with_ids_avail = [{'orig_i': i, 'bbox': key_f(bbox), 'connections': set(), 'group': -1}
                             for i, bbox in enumerate(items)]
    dim = (otherdim + 1) % 2
    bboxes_with_ids_avail.sort(key=lambda item: item['bbox'][dim])  # orig_id, bbox, True
    
    def find_overlaps(x, fromrange):
        thisx = bboxes_with_ids_avail[x]
        avail_ids_overlaps = [(i, range_overlap_percent(thisx['bbox'][otherdim], thisx['bbox'][otherdim + 2],
                                                        bboxes_with_ids_avail[i]['bbox'][otherdim],
                                                        bboxes_with_ids_avail[i]['bbox'][otherdim + 2],
                                                        ))
                              for i in fromrange if i != x]
        if not avail_ids_overlaps:
            return []  # there are no candidates for left overlap
        # for maximum:
        candidate = max(avail_ids_overlaps, key=lambda it: it[1])
        if candidate[1] < percent_thr:
            return []  # if the maximal overlap was not positive (or bigger than thr), lets not go further
        all_connectings = [candidate[0]]
        
        if method_soft:  # in soft variant we have 2 neighbours to return
            # for 'first, that satisfies threshold'
            for ibox in avail_ids_overlaps:
                if ibox[1] > percent_thr:
                    all_connectings.append(ibox[0])
                    break
        return all_connectings
    
    def find_left_overlaps(x):
        return find_overlaps(x, range(x, 0, -1))  # from x to 0
    
    def find_right_overlaps(x):
        return find_overlaps(x, range(x + 1, len(bboxes_with_ids_avail), 1))
    
    for inode, node in enumerate(bboxes_with_ids_avail):
        news = {new for new in find_left_overlaps(inode) + find_right_overlaps(inode) if new is not None}
        node['connections'] = node['connections'].union(news)
        
        for new in news:  # make edges go both ways
            bboxes_with_ids_avail[new]['connections'].add(inode)
    
    group_id = 0
    while True:
        # take the bbox with the lowest id, that is the first one from the top:
        remaining = [inode for inode, node in enumerate(bboxes_with_ids_avail) if node['group'] == -1]
        if len(remaining) <= 0:
            break
        topmost = min(remaining, key=lambda inode: bboxes_with_ids_avail[inode]['orig_i'])
        
        # combine those that have something in common with him
        stack = [topmost]  # stack of IDS to array 'bboxes_with_ids_avail'
        while len(stack) > 0:
            node = stack.pop()
            bboxes_with_ids_avail[node]['group'] = group_id
            news = bboxes_with_ids_avail[node]['connections']  # new nodes
            for new in news:
                if bboxes_with_ids_avail[new]['group'] == -1:  # unvisited
                    stack.append(new)
        
        # at this point we have found the topmost line! ... let available remain and those with false let go as
        # the first line
        linef = [items[item['orig_i']] for item in bboxes_with_ids_avail if item['group'] == group_id]
        if len(linef) > 0:
            yield linef
            # is it sorted (as linef.sort(key=lambda item: key_f(item)[0]) would be)? Yes it is!
        else:
            break
        group_id += 1


def get_items_reading_layout(items, key_f=None, percents_thr_chunk=0.0, percent_thr_inline=0.5,
                             row_by_row=True, method_soft=True):
    """
    Algorithm that would return the reading order of bboxes, parameterizable with overlapping percentages.

    How the algorithm works:
    - lets separate boxes into y-overlapping (in y projection) chunks with threshold = percents_thr_chunk
    - inside the chunk it will happen, that one bbox overlaps with more others.
    - Take the topmost (by edge) bbox
    - Assign him to one at the left and one at the right that he overlaps the most from that side.
     - BFS deeper on his 2 neighbours
     - each bbox has 2 hands like this (because each one can decide differently).
       - Then remove all hands-holding from the chunk as being the next line.
    - repeat until nothing is in the chunk.

    items: The items list
    key_f: A function, that returns a (l,t,r,b) bbox when applied on an items member
    percents_thr_chunk: First stage - separate items hungrily in chunks with this percentage threshold
     on projection overlappings
    percent_thr_inline: threshold for producing edges for the BFS second stage
    row_by_row: produce row by row; False - produce col by col layout
    method_soft: use softer method for the second stage (default). False option is for experimental reasons.
    """
    if key_f is None:
        def key_f(x):
            return x.bbox
    lines = []
    
    for chunk in produce_ordered_and_overlapping_chunks(items, key_f, percents_thr_chunk,
                                                        dim=1 if row_by_row else 0):
        for line in search_lines_in_bboxes(chunk, key_f, percent_thr_inline,
                                           otherdim=(1 if row_by_row else 0),
                                           method_soft=method_soft):
            lines.append(line)
    return lines


def range_overlap(a_min, a_max, b_min, b_max):
    """Neither range is completely greater than the other"""
    assert a_min <= a_max and b_min <= b_max
    return (a_min < b_max) and (a_max > b_min)


def overlaps(r1, r2):
    """Overlapping rectangles overlap both horizontally & vertically"""
    return range_overlap(r1[0], r1[2], r2[0], r2[2]) and range_overlap(r1[1], r1[3], r2[1], r2[3])


def overlaps_ids(source, targets, overlap_percent=0.2):
    result = []
    for i, overlapper in enumerate(targets):
        if overlap_percent <= 0.0:
            if overlaps(source.bbox, overlapper.bbox):
                result.append(i)
        else:
            if cut_percentage(source.bbox, overlapper.bbox) >= overlap_percent:
                result.append(i)
    return result


def sanitize_bbox(bbox, pic_shape):
    """
    Makes sure the bounding box is within picture and not flipped.
    If it's empty, None is returned.
    """
    left, top, right, bottom = np.array(bbox, dtype=float)
    if pic_shape is not None:
        height, width = pic_shape[:2]
        fixed_bbox = np.array([
            max(min(left, right), 0),
            max(min(top, bottom), 0),
            min(max(left, right), width - 1),
            min(max(top, bottom), height - 1),
        ], dtype=float)
    else:
        fixed_bbox = np.array([
            max(min(left, right), 0),
            max(min(top, bottom), 0),
            max(left, right),
            max(top, bottom),
        ], dtype=float)
    
    if (fixed_bbox[2] - fixed_bbox[0] > 0) and (fixed_bbox[3] - fixed_bbox[1] > 0):
        return fixed_bbox
    else:
        return None


def sanitize_bboxes(bboxes, pic_shape):
    fixed_bboxes = []
    for bbox in bboxes:
        fixed_bbox = sanitize_bbox(bbox, pic_shape)
        if fixed_bbox is not None:
            fixed_bboxes.append(fixed_bbox)
    return np.array(fixed_bboxes)


def get_doc_annotations(doc):
    raise Exception("Implement for your data structures")


def produce_annotations_for_page(doc, page_n, reading_percents=[0.5, 0.5], overlap_classes_percents=0.2,
                                 use_neighbours=0):
    annots_text_o = [annotation for annotation in get_doc_annotations(doc) if
                     annotation.page == page_n and annotation.cls_extract_type == 'text']
    annots_text = []
    for annot in annots_text_o:
        fixed_bbox = sanitize_bbox(annot.bbox, None)
        if fixed_bbox is not None:
            annot.bbox = fixed_bbox
            annots_text.append(annot)
    
    # first lets order all the boxes by the reading layout
    annots_text = sum(get_items_reading_layout(annots_text, key_f=lambda annot: annot.bbox,
                                               percent_thr_inline=reading_percents[0]), [])
    text_features = [{'bbox': annot.bbox,
                      'text': annot.text,
                      } for annot in annots_text]
    
    boxdata_to_lrtb = lambda item: item['bbox']
    # then lets assign coordinates based on the reding order (and also rotated in y dir)
    for i, group in enumerate(get_items_reading_layout(text_features, key_f=boxdata_to_lrtb,
                                                       percent_thr_inline=reading_percents[0])):
        for j, annot_feature in enumerate(group):
            annot_feature['row_readings_pos'] = (i, j)
    for i, group in enumerate(
            get_items_reading_layout(text_features, key_f=boxdata_to_lrtb, row_by_row=False,
                                     percent_thr_inline=reading_percents[1])):
        for j, annot_feature in enumerate(group):
            annot_feature['col_readings_pos'] = (i, j)
    
    wboxes_annotations = [annot for annot in get_doc_annotations(doc) if
                          annot.page == page_n and annot.cls_extract_type != 'text']
    annotations_features = [{'bbox': annot.bbox,
                             'cls_extract_type': annot.cls_extract_type,
                             'covered_wordboxes': overlaps_ids(annot, annots_text, overlap_percent=0.2),
                             'text': annot.text,
                             } for annot in wboxes_annotations]
    
    if use_neighbours > 0:
        # each annotation has its own data in the whole sequence AND the datas of 4*use_neighbours neighbours
        def key_bbox(item):
            return text_features[item]['bbox']
        
        all_seen_data = []
        for ith, label in enumerate(text_features):
            all_seen_data.append((label, filter_seen_boxes(ith, list(range(len(text_features))),
                                                           key_f=key_bbox,
                                                           ortho_filter=False, max_sqr_dist=1000.0 * 1000.0,
                                                           order_by_dist=True, max_slots=use_neighbours)))
            # filter_seen returns {edge_id: (item, edge_id, dist)}
        defaulted_neigh_ids = []
        use_self = False  # will index to self data , false will index to minus
        for tid, seens in enumerate(all_seen_data):
            np_neigh_ids = np.full((4 * use_neighbours,), tid if use_self else -1, dtype=np.int32)
            # ^^ a default value for a neighbour, that is not present will be -1
            # (the index of exactly this annotation can also be used, but then the default batcher cannot do things easily)
            for edge_n in range(4):
                for nid, seen in enumerate(seens[1][edge_n]):  # max count of seens is known to be 'use_neighbours'
                    np_neigh_ids[edge_n * use_neighbours + nid] = seen[0]
                    # this format affectgs `def use_neighours_count` in sqlite_experiment_generator.py
                    # from (item, edge_id, dist) we take item which is id
            # defaulted_neigh_ids.append(np_neigh_ids)
            text_features[tid]['neighbours'] = np_neigh_ids
        
        # f_neigh_ids = np.array(defaulted_neigh_ids, dtype=np.int32)
        # we can check if is produced an array of a len,4 size
    
    return text_features, annotations_features


def area(bbox):
    return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def cut_percentage(box1, box2):
    """Bboxes as [left top right bottom]."""
    lx, tx, rx, bx = box1
    ly, ty, ry, by = box2
    
    # XXX: If the bottom-right corner is a part of the box, use this:
    # rx += 1
    # bx += 1
    # ry += 1
    # by += 1
    
    li = max(lx, ly)
    ti = max(tx, ty)
    ri = min(rx, ry)
    bi = min(bx, by)
    
    intersect = max(ri - li, 0) * max(bi - ti, 0)
    smaller = min(area(box1), area(box2))
    
    return (1.0 * intersect / max(smaller, 1e-5)) if intersect else 0.0
