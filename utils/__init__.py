import matplotlib
matplotlib.use('Agg')
from matplotlib import patches as mpatch, pyplot as plt

pic_inpgrid_width = 620
pic_inpgrid_height = 877


def produce_drawings(istep, b, cls_extract_types, truth, pred, positions,
                     nearest_truth, nearest_positions,
                     plots_prefix):
    if istep not in [1, 10, 100, 1000, 10000]:
        return
    
    if plots_prefix is None:
        return
    
    rectangles = []
    for wb_i, (wtruth, wpred, pos) in enumerate(zip(truth, pred, positions)):
        goods = []
        extras = []
        misses = []
        for i, cls_extract_type in enumerate(cls_extract_types):
            t = wtruth[i] > 0.5
            p = wpred[i] > 0.5
            if not t and not p:
                continue
            if t and p:
                goods.append(cls_extract_type)
            elif p and not t:
                extras.append(cls_extract_type)
            else:
                misses.append(cls_extract_type)
        cls_extract_type = None
        if len(misses) > 0:
            color = 'red'
            cls_extract_type = misses[0]
        elif len(extras) > 0:
            color = 'blue'
            cls_extract_type= extras[0]
        elif len(goods) > 0:
            cls_extract_type = goods[0]
            color = 'green'
        if cls_extract_type is not None:
            rectangles.append([cls_extract_type,  #xy w h:
                               mpatch.Rectangle((pos[0], pos[1]),
                                                (pos[2] - pos[0]),
                                                (pos[3] - pos[1]), color=color)])
        else:
            rectangles.append(["",  # xy w h:
                               mpatch.Rectangle((pos[0], pos[1]),
                                                (pos[2] - pos[0]),
                                                (pos[3] - pos[1]), color='yellow')])

    def draw_rectangles(rectangles, ax):
        for rct in rectangles:
            ax.add_artist(rct[-1])
            rx, ry = rct[-1].get_xy()
            cx = rx + rct[-1].get_width() / 2.0
            cy = ry + rct[-1].get_height() / 2.0
            ax.annotate(rct[0], (cx, cy), color='black', weight='bold',
                        fontsize=2, ha='center', va='center')
    
    fig, ax = plt.subplots(figsize=(int(pic_inpgrid_width/100), int(pic_inpgrid_height/100)))

    draw_rectangles(rectangles, ax)

    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("{}plt{}-{}.svg".format(plots_prefix, istep, b))
    plt.close()
    
    # now paint the nearest page:
    
    if nearest_truth is not None and nearest_positions is not None:
        nearest_full = []
        for wb_i, (ntruth, pos) in enumerate(zip(nearest_truth, nearest_positions)):
            ftypes = []
            for i, cls_extract_type in enumerate(cls_extract_types):
                t = ntruth[i] > 0.5
                if t:
                    ftypes.append(cls_extract_type)
            cls_extract_type = " ".join(ftypes)
            if len(ftypes) > 0:
                nearest_full.append([cls_extract_type,  # xy w h:
                                   mpatch.Rectangle((pos[0], pos[1]),
                                                    (pos[2] - pos[0]),
                                                    (pos[3] - pos[1]), color='green')])
            else:
                nearest_full.append(["",  # xy w h:
                                   mpatch.Rectangle((pos[0], pos[1]),
                                                    (pos[2] - pos[0]),
                                                    (pos[3] - pos[1]), color='yellow')])
    
        fig, ax = plt.subplots(figsize=(int(pic_inpgrid_width / 100), int(pic_inpgrid_height / 100)))
    
        draw_rectangles(nearest_full, ax)
    
        ax.set_aspect('equal')
        plt.axis('off')
        plt.savefig("{}pltnearest{}-{}.svg".format(plots_prefix, istep, b))
        plt.close()