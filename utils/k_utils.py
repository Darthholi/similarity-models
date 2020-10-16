#
# COPYRIGHT Martin Holecek 2019
#

import tensorflow as tf
from keras.engine.saving import load_model
from keras.layers import Layer

from utils.attention import AttentionTransformer, SinCosPositionalEmbedding


class GatherFromIndices(Layer):
    """
    To have a graph convolution (over a fixed/fixed degree kernel) from a given sequence of nodes, we need to gather
    the data of each node's neighbours before running a simple Conv1D/conv2D,
     that would be effectively a defined convolution (or even TimeDistributed(Dense()) can be used - only
     based on data format we would output).
    This layer should do exactly that.

    Does not support non integer values, values lesser than 0 zre automatically masked.
    """
    
    def __init__(self, mask_value=0, include_self=True, flatten_indices_features=False, **kwargs):
        Layer.__init__(self, **kwargs)
        self.mask_value = mask_value
        self.include_self = include_self
        self.flatten_indices_features = flatten_indices_features
    
    def get_config(self):
        config = {'mask_value': self.mask_value,
                  'include_self': self.include_self,
                  'flatten_indices_features': self.flatten_indices_features,
                  }
        base_config = super(GatherFromIndices, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    # def build(self, input_shape):
    # self.built = True
    
    def compute_output_shape(self, input_shape):
        inp_shape, inds_shape = input_shape
        indices = inds_shape[-1]
        if self.include_self:
            indices += 1
        features = inp_shape[-1]
        if self.flatten_indices_features:
            if indices is None:
                return tuple(list(inds_shape[:-1]) + [None])
            return tuple(list(inds_shape[:-1]) + [indices * features])
        else:
            return tuple(list(inds_shape[:-1]) + [indices, features])
    
    def call(self, inputs, training=None):
        inp, inds = inputs
        # assumes input in the shape of (inp=[...,batches, sequence_len, features],
        #  inds = [...,batches,sequence_ind_len, neighbours]... indexing into inp)
        # for output we want to get  [...,batches,sequence_ind_len, indices,features]
        
        assert_shapes = tf.Assert(tf.reduce_all(tf.equal(tf.shape(inp)[:-2], tf.shape(inds)[:-2])), [inp])
        assert_positive_ins_shape = tf.Assert(tf.reduce_all(tf.greater(tf.shape(inds), 0)), [inds])
        # the shapes need to be the same (with the exception of the last dimension)
        with tf.control_dependencies([assert_shapes, assert_positive_ins_shape]):
            inp_shape = tf.shape(inp)
            inds_shape = tf.shape(inds)
            
            features_dim = -1
            # ^^ for future variablility of the last dimension, because maybe can be made to take not the last
            # dimension as features, but something else.
            
            inp_p = tf.reshape(inp, [-1, inp_shape[features_dim]])
            ins_p = tf.reshape(inds, [-1, inds_shape[features_dim]])
            
            # we have lost the batchdimension by reshaping, so we save it by adding the size to the respective indexes
            # we do it because we use the gather_nd as nonbatched (so we do not need to provide batch indices)
            resized_range = tf.range(tf.shape(ins_p)[0])
            different_seqs_ids_float = tf.scalar_mul(1.0 / tf.to_float(inds_shape[-2]), tf.to_float(resized_range))
            different_seqs_ids = tf.to_int32(tf.floor(different_seqs_ids_float))
            different_seqs_ids_packed = tf.scalar_mul(inp_shape[-2], different_seqs_ids)
            thseq = tf.expand_dims(different_seqs_ids_packed, -1)
            
            # in case there are negative indices, make them all be equal to -1
            #  and add masking value to the ending of inp_p - that way, everything that should be masked
            #  will get the masking value as features.
            mask = tf.greater_equal(ins_p,
                                    0)  # extract where minuses are, because the will all default to default value
            # .. before the mod operation, if provided greater id numbers, to wrap correctly small sequences
            offset_ins_p = tf.mod(ins_p, inp_shape[-2]) + thseq  # broadcast to ins_p
            minus_1 = tf.scalar_mul(tf.shape(inp_p)[0], tf.ones_like(mask, dtype=tf.int32))
            '''
            On GPU, if we use index = -1 anywhere it would throw a warning:
            OP_REQUIRES failed at gather_nd_op.cc:50 : Invalid argument:
            flat indices = [-1] does not index into param.
            Which is a warning, that there are -1s. We are using that as feature and know about that.
            '''
            offset_ins_p = tf.where(mask, offset_ins_p, minus_1)
            # also possible to do something like  tf.multiply(offset_ins_p, mask) + tf.scalar_mul(-1, mask)
            mask_value_last = tf.zeros((inp_shape[-1],))
            if self.mask_value != 0:
                mask_value_last += tf.constant(self.mask_value)  # broadcasting if needed
            inp_p = tf.concat([inp_p, tf.expand_dims(mask_value_last, 0)], axis=0)
            
            # expand dims so that it would slice n times instead having slice of length n indices
            neighb_p = tf.gather_nd(inp_p, tf.expand_dims(offset_ins_p, -1))  # [-1,indices, features]
            
            out_shape = tf.concat([inds_shape, inp_shape[features_dim:]], axis=-1)
            neighb = tf.reshape(neighb_p, out_shape)
            # ^^ [...,batches,sequence_len, indices,features]
            
            if self.include_self:  # if is set, add self at the 0th position
                self_originals = tf.expand_dims(inp, axis=features_dim - 1)
                # ^^ [...,batches,sequence_len, 1, features]
                neighb = tf.concat([neighb, self_originals], axis=features_dim - 1)
            
            if self.flatten_indices_features:
                neighb = tf.reshape(neighb, tf.concat([inds_shape[:-1], [-1]], axis=-1))
            
            return neighb


def tile_for_product_matrix(vect_inp1, vect_inp2):
    Nboxes1 = tf.shape(vect_inp1)[-2]
    Nboxes2 = tf.shape(vect_inp2)[-2]
    
    v1 = tf.expand_dims(vect_inp1, -3)
    v1 = tf.tile(v1, [1, Nboxes2, 1, 1])
    
    v2 = tf.expand_dims(vect_inp2, -2)
    v2 = tf.tile(v2, [1, 1, Nboxes1, 1])
    
    return [v1, v2]


def tile_for_product_matrix_shape(shape1, shape2):
    return [(shape1[0], shape2[1], shape1[1], shape1[2]),
            (shape2[0], shape2[1], shape1[1], shape2[2])]


def tile_to_match(tile, tomatch):
    Nboxes1 = tf.shape(tomatch)[-2]
    
    v2 = tf.expand_dims(tile, -2)
    v2 = tf.tile(v2, [1, 1, Nboxes1, 1])
    
    return v2


def tile_to_match_shape(shape1, shape2):
    return [(shape2[0], shape2[1], shape1[1], shape2[2])]


def make_product_matrix(vect_inp1, vect_inp2):
    '''
    vect_inp1,2: [batches, Nboxes1, Cfeatures1], [batches, Nboxes2, Cfeatures2]
    should return [batches, Nboxes1, Nboxes2, Cfeatures1 + Cfeatures2]
    '''
    v1, v2 = tile_for_product_matrix(vect_inp1, vect_inp2)
    ret = tf.concat([v1, v2], -1)
    
    # assert_op = tf.Assert(tf.equal(tf.shape(ret)[-2], tf.shape(ret)[-3]), [ret])
    # with tf.control_dependencies([assert_op]):
    return ret


def load_our_model(model_name):
    return load_model(model_name, custom_objects={"SinCosPositionalEmbedding": SinCosPositionalEmbedding,
                                                  "AttentionTransformer": AttentionTransformer})


def load_weights_as_possible(model, start_weights_from):
    """
    Load weights with same layer names if possible.
    """
    try:
        model.load_weights(start_weights_from, by_name=True,
                           skip_mismatch=True, reshape=False)
    except Exception as e:
        xmodel = load_our_model(start_weights_from)
        xdict = {}
        for layer in xmodel.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                xdict[layer.name] = weights
        for layer in model.layers:
            if layer.name in xdict:
                try:
                    layer.set_weights(xdict[layer.name])
                except Exception as e:
                    pass
