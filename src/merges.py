from keras.layers.merge import _Merge
import tensorflow as tf


class SubtractAbs(_Merge):
    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')
        return tf.abs(inputs[0] - inputs[1])


class AddMean(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output / len(inputs)
