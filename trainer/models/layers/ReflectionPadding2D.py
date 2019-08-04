from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
import tensorflow as tf
from tensorflow.keras.layers import InputSpec

class ReflectionPadding2D(Layer):
    def __init__(self, data_format='channels_last', padding=(1, 1), mode='REFLECT', **kwargs):
        self.padding = tuple(padding)
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.mode = mode
        
        assert self.data_format in ['channels_last', 'channels_first']
        assert self.mode in ['SYMMETRIC', 'REFLECT']
        
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        if self.data_format == 'channels_last':
            return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])
        if self.data_format == 'channels_first':
            return (s[0], s[1], s[2] + 2 * self.padding[0], s[3] + 2 * self.padding[1])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        if self.data_format == 'channels_last':
            return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], self.mode)
        if self.data_format == 'channels_first':
            return tf.pad(x, [[0,0], [0,0], [h_pad,h_pad], [w_pad,w_pad] ], self.mode)

class Conv2D:
    def __init__(self, filters, kernel_size, strides=(1,1), padding='symmetric', dilation_rate=(1,1), **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.kwargs = kwargs
        
    def __call__(self, x):
        if self.padding in ['symmetric', 'reflect']:
            p = int(self.dilation_rate[0]*(self.kernel_size[0]-1)/2)            
            x = ReflectionPadding2D(mode=self.padding.upper(), padding=(p,p))(x)
            return tf.keras.layers.Conv2D(self.filters, self.kernel_size, 
                                          strides=self.strides, padding='valid', 
                                          dilation_rate=self.dilation_rate, **self.kwargs)(x)
        else:
            return tf.keras.layers.Conv2D(self.filters, self.kernel_size, 
                                          strides=self.strides, padding=self.padding, 
                                          dilation_rate=self.dilation_rate, **self.kwargs)(x)
