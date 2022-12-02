# https://towardsdatascience.com/xception-from-scratch-using-tensorflow-even-better-than-inception-940fb231ced9

import tensorflow as tf

class Xception(tf.keras.Model):
    def __init__(self):
        super(Xception, self).__init__()

    def BatchNormalization(self, x, filters, kernel_size, strides=1):
        x = tf.keras.layers.Conv2D(filters=filters, 
                kernel_size = kernel_size, 
                strides=strides, 
                padding = 'same', 
                use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def SeparableBatchNormalization(self, x, filters, kernel_size, strides=1):
        x = tf.keras.layers.SeparableConv2D(filters=filters, 
                            kernel_size = kernel_size, 
                            strides=strides, 
                            padding = 'same', 
                            use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def entry_flow(self, x):
        x = self.BatchNormalization(x, filters = 32, kernel_size = 3, strides = 2)
        x = tf.keras.layers.ReLU()(x)
        x = self.BatchNormalization(x, filters = 64, kernel_size = 3, strides = 1)
        tensor = tf.keras.layers.ReLU()(x)
        x = self.SeparableBatchNormalization(tensor, filters = 128, kernel_size = 3)
        x = tf.keras.layers.ReLU()(x)
        x = self.SeparableBatchNormalization(x, filters = 128, kernel_size =3)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides = 2, padding = 'same')(x)
        tensor = self.BatchNormalization(tensor, filters = 128, kernel_size = 1,strides = 2)
        x = tf.keras.layers.Add()([tensor, x])
        x = tf.keras.layers.ReLU()(x)
        x = self.SeparableBatchNormalization(x, filters = 256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = self.SeparableBatchNormalization(x, filters = 256, kernel_size=3)
        x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
        tensor = self.BatchNormalization(tensor, filters = 256, kernel_size = 1,strides = 2)
        x = tf.keras.layers.Add()([tensor,x])
        x = tf.keras.layers.ReLU()(x)
        x = self.SeparableBatchNormalization(x, filters = 728, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = self.SeparableBatchNormalization(x, filters = 728, kernel_size=3)
        x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
        tensor = self.BatchNormalization(tensor, filters = 728, kernel_size = 1,strides = 2)
        x = tf.keras.layers.Add()([tensor,x])
        return x

    def middle_flow(self, tensor):
        for _ in range(8):
            x = tf.keras.layers.ReLU()(tensor)
            x = self.SeparableBatchNormalization(x, filters = 728, kernel_size = 3)
            x = tf.keras.layers.ReLU()(x)
            x = self.SeparableBatchNormalization(x, filters = 728, kernel_size = 3)
            x = tf.keras.layers.ReLU()(x)
            x = self.SeparableBatchNormalization(x, filters = 728, kernel_size = 3)
            x = tf.keras.layers.ReLU()(x)
            tensor = tf.keras.layers.Add()([tensor,x])
        return tensor
    
    def exit_flow(self, tensor):
        x = tf.keras.layers.ReLU()(tensor)
        x = self.SeparableBatchNormalization(x, filters = 728,  kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = self.SeparableBatchNormalization(x, filters = 1024,  kernel_size=3)
        x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding ='same')(x)
        tensor = self.BatchNormalization(tensor, filters =1024, kernel_size=1, strides = 2)
        x = tf.keras.layers.Add()([tensor,x])
        x = self.SeparableBatchNormalization(x, filters = 1536,  kernel_size = 3)
        x = tf.keras.layers.ReLU()(x)
        x = self.SeparableBatchNormalization(x, filters = 2048,  kernel_size = 3)
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Dense (units = 1000, activation = 'softmax')(x)
        return x
    
    def call(self, input):
        x = self.entry_flow(input)
        x = self.middle_flow(x)
        output = self.exit_flow(x)
        return output