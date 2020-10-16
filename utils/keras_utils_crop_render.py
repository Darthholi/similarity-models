#
# COPYRIGHT Martin Holecek 2019
#

import itertools

import numpy as np
import tensorflow as tf
from keras.layers import Lambda


def tf_crop_and_resize_batches_of_arrays(image_input, boxes_input, crop_size):
    """
    Crops and resizes batch of images using batch_bbox bounding boxes for each image.
    Some boxes might be bogus (set to zero).
    image_input - Array of images, tensor with shape [batch, image_width, image_height, color_channels]
    boxes_input - Array of bounding boxes for each image, each box is defined by top-left and bottom right corner,
        tensor with shape [batch, batch_bbox, 4]
    # note that tf docs https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    # just rename width and height and flip ltrb -> tlbr, which is just a different notation.
    crop_size - a tuple defining (cropped_image_width, cropped_image_height) sizes that all crops will be resized to.

    The dimension of the output will be
        [batch, batch_bbox, cropped_image_width, cropped_image_height, color_channels] crops.

    To note, you can freely change the meaning of image_width and image_height written here, it is just a name of a size
        in that specified dimension.

    Examples of inputs:
    image_input = tf.ones((2, 200, 200, 1))
    boxes_input = tf.constant([[[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]],
                               [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]]])
    crop_size = (10, 10)
    """
    bboxes_per_batch = tf.shape(boxes_input)[1]
    batch_size = tf.shape(boxes_input)[0]
    # should be the same as image_input.shape[0]:
    with tf.control_dependencies([tf.assert_equal(batch_size, tf.shape(image_input)[0])]):
        # the goal is to create a [batch, maxnumperbatch] field of values,
        #  which are the same across batch and equal to the batch_id
        # and then to reshape it in the same way as we do reshape the boxes_input to just tell tf about
        #  each bboxes batch (and image).
        index_to_batch = tf.tile(tf.expand_dims(tf.range(batch_size), -1), (1, bboxes_per_batch))
        
        # now both get reshaped as tf wants it:
        boxes_processed = tf.reshape(boxes_input, (-1, 4))
        box_ind_processed = tf.reshape(index_to_batch, (-1,))
        
        # w = tf.Variable(tf.random_uniform([2, 2])) # init_op = w.initializer
        
        # the method wants boxes = [num_boxes, 4], box_ind = [num_boxes] to index into the batch
        # the method returns [num_boxes, crop_height, crop_width, depth]
        
        tf_produced_crops = tf.image.crop_and_resize(
            image_input,
            boxes_processed,
            box_ind_processed,
            crop_size,
            method='bilinear',
            extrapolation_value=0,
            name=None
        )
        new_shape = tf.concat([tf.stack([batch_size, bboxes_per_batch]), tf.shape(tf_produced_crops)[1:]], axis=0)
        crops_resized_to_original = tf.reshape(tf_produced_crops,
                                               new_shape)
        return crops_resized_to_original
    
    # devel comments, maybe not that much usefull:
    # https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    # but here it is ONE image and bboxes are from that one image
    # which 'should' not matter. ... i.e. I think that a test that would show me that it crops the bboxes correctly
    # would be the right thing
    # box_ind = tf.zeros_like
    # can we paralelize it to make it work like we want?
    
    # ok a better approach would be to really care for the batch dimension...
    # lets expect bboxes to be [batch, num(max_num_per_batch), 4]
    # images would be [batch, imgsize0, imgsize1]
    # the method wants boxes = [num_boxes, 4], box_ind = [num_boxes] to index into the batch
    # the method returns [num_boxes, crop_height, crop_width, depth]
    # we want to return things again in [batch, num(max_num_per_batch), 4]


def keras_crop_and_resize_batches_of_arrays(image_input, boxes_input, crop_size):
    """
    A helper function for tf_crop_and_resize_batches_of_arrays,
     assuming, that the crop_size would be a constant and not a tensorflow operation.
    """
    
    def f_crop(packed):
        image, boxes = packed
        return tf_crop_and_resize_batches_of_arrays(image, boxes, crop_size)
    
    return Lambda(f_crop)([image_input, boxes_input])


def render_bboxes_pyfunc_2d(elems, target_shape):
    """
    2d only numpy + tf.py_func replacement for render_nd_bboxes_tf_spreading.
    For testing purposes.
    The only faster way can then be using custom kernels https://www.tensorflow.org/guide/extend/op
    """
    
    # target_shape = [dimx, dimy,....]
    
    def py_render_boxes_2d(x_boxes_data, out_shape):
        # x will be a numpy array with the contents of the placeholder below
        if len(x_boxes_data.shape) <= 2:
            result = np.zeros(list(out_shape) + [x_boxes_data.shape[-1] - 2 * 2], dtype=np.float32)
            for box in x_boxes_data:
                result[box[0]:box[2], box[1]: box[3], :] += box[4:]
        else:  # also batch dimension is provided
            result = np.zeros([x_boxes_data.shape[0]] + list(out_shape) + [x_boxes_data.shape[-1] - 2 * 2],
                              dtype=np.float32)
            for i, batch in enumerate(x_boxes_data):
                for box in batch:
                    result[i, box[0]:box[2], box[1]: box[3], :] += box[4:]
        return result
    
    return tf.py_func(py_render_boxes_2d, [elems, target_shape], tf.float32)


def render_nd_bboxes_tf_spreading(elems, target_shape, ndim=2):
    """
    elems: tensor of size [..., n_boxes, 2*ndim + val_dim], where in the last dimension,
     there are packed edge coordinates and values (of val_dim) to be filled in the specified box.
    target_shape: list/tuple of ndim entries.
    returns: rendered image of size [elems(...), target_shape..., val_dim]
    ('elems(...)' usually means batch_size)
    """
    assert_shape_ndim = tf.Assert(tf.equal(tf.size(target_shape), ndim), [target_shape])
    assert_nonempty_data = tf.Assert(tf.greater(tf.shape(elems)[-1], 2 * ndim), [elems])
    
    with tf.control_dependencies([assert_shape_ndim, assert_nonempty_data]):
        """
        +1 ...... -1      ++++++      ++++++
        ...........       ......      ++++++
        ...........    -> ......   -> ++++++
        ...........       ------      ++++++
        -1        +1
        in 3d there must be another wall of minuses. looking like that:

        -   +
        .....
        +   -

        so when indexing [0, 1] to ltrb... pluses are when there is even number of 0s, - when odd.
        """
        el_ndim = len(elems.shape)
        # we do not access this property in tensorflow runtime, but in 'compile time', because, well,
        # number of dimensions
        # should be known before
        
        assert el_ndim >= 2 and el_ndim <= 3, "elements should be in the form of [batch, n, coordinates] or [n, " \
                                              "coordinates]"
        if el_ndim == 3:  # we use batch_size dimension also!
            bboxes_per_batch = tf.shape(elems)[1]
            batch_size = tf.shape(elems)[0]  # should be the same as image_input.shape[0]
            index_to_batch = tf.tile(tf.expand_dims(tf.range(batch_size), -1), (1, bboxes_per_batch))
            index_to_batch = tf.reshape(index_to_batch, (-1, 1))
        else:
            index_to_batch = None
        
        val_vector_size = tf.shape(elems)[-1] - 2 * ndim
        
        corner_ids = list(itertools.product([0, 1], repeat=ndim))
        corners_lists = []
        corners_values = []
        for corner in corner_ids:
            plus = sum(corner) % 2 == 0
            id_from_corner = [i + ndim * c for i, c in
                              enumerate(corner)]  # indexes a corner into [left, top, right, bottom] notation
            corner_coord = tf.gather(elems[..., 0: 2 * ndim], id_from_corner, axis=-1)
            corner_value = elems[..., 2 * ndim:] * (1 if plus else -1)  # last dimension is == val_vector_size
            if index_to_batch is not None:
                # if the operation is called in batches, remember to rehape it all into one long list for scatter_nd
                # and add (concatenate) the batch ids
                corner_coord = tf.concat([index_to_batch, tf.reshape(corner_coord, (-1, 2))], axis=-1)
                corner_value = tf.reshape(corner_value, (-1, val_vector_size))
            corners_lists.append(corner_coord)
            corners_values.append(corner_value)
        
        indices = tf.concat(corners_lists, axis=0)
        updates = tf.concat(corners_values, axis=0)
        shape = tf.concat([tf.shape(elems)[:-2], target_shape, [val_vector_size]], axis=0)
        
        dense_orig = tf.scatter_nd(
            indices,
            updates,
            shape=shape,
        )
        
        dense = dense_orig
        for dim in range(ndim):
            # we want to start from the axis before the last one. The last one is the value dimension, and
            # the first dimensions hidden in the '...' might be the batched dimensions
            dense = tf.cumsum(dense, axis=-2 - dim, exclusive=False, reverse=False, name=None)
        
        return dense
