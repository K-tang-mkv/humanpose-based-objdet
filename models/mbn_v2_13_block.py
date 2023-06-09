import tensorflow as tf
from tensorflow.keras.models import Model
from keras.applications import imagenet_utils


class BlazePose():
    def __init__(self, num_keypoints: int):
        self.num_keypoints, self.alpha = num_keypoints, 0.75
        self.input_1 = tf.keras.layers.Input(shape=(192, 192, 3))
        self.mobilenet_v2_skip_block(self.input_1, self.alpha)

    def fuse_1(self, fm8, fm16):
        # conv_fpn1_d
        decode_up1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation=None)(fm8)
        decode_up1 = tf.keras.layers.BatchNormalization()(decode_up1)
        decode_up1 = tf.keras.layers.ReLU()(decode_up1)
        decode_up1 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                     kernel_size=3, padding="same", strides=(1, 1))(decode_up1)
        # conv_fpn1_e
        encode_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation=None)(fm16)
        encode_1 = tf.keras.layers.BatchNormalization()(encode_1)
        encode_1 = tf.keras.layers.ReLU()(encode_1)
        fuse_1 = tf.keras.layers.Add()([encode_1, decode_up1])
        return fuse_1

    def fuse_2(self, fuse_1, fm32):
        # conv_fpn2_d
        decode_up2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None)(fuse_1)
        decode_up2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation=None)(decode_up2)
        decode_up2 = tf.keras.layers.BatchNormalization()(decode_up2)
        decode_up2 = tf.keras.layers.ReLU()(decode_up2)
        decode_up2 = tf.keras.layers.Conv2DTranspose(filters=32,
                                                     kernel_size=3, padding="same", strides=(2, 2))(decode_up2)
        # conv_fpn2_e
        encode_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation=None)(fm32)
        encode_2 = tf.keras.layers.BatchNormalization()(encode_2)
        encode_2 = tf.keras.layers.ReLU()(encode_2)
        fuse_2 = tf.keras.layers.Add()([encode_2, decode_up2])
        return fuse_2

    def fuse_3(self, fuse_2, fm64):
        # conv_fpn3_d
        decode_up3 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None)(fuse_2)
        decode_up3 = tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)(decode_up3)
        decode_up3 = tf.keras.layers.BatchNormalization()(decode_up3)
        decode_up3 = tf.keras.layers.ReLU()(decode_up3)
        decode_up3 = tf.keras.layers.Conv2DTranspose(filters=24,
                                                     kernel_size=3, padding="same", strides=(2, 2))(decode_up3)
        # conv_fpn3_e
        encode_3 = tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)(fm64)
        encode_3 = tf.keras.layers.BatchNormalization()(encode_3)
        encode_3 = tf.keras.layers.ReLU()(encode_3)
        fuse_3 = tf.keras.layers.Add()([encode_3, decode_up3])
        return fuse_3

    def conv_final(self, fuse_3):
        y = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None)(fuse_3)
        y = tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None)(y)
        y = tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Conv2D(filters=self.num_keypoints, kernel_size=1, activation=None)(y)
        return y

    def mobilenet_v2_skip_block(self, input_1, alpha):

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
            """Inverted ResNet block."""
            x, in_channels, channel_axis = inputs, inputs.shape[-1], -1
            pointwise_filters = _make_divisible(int(filters * alpha), 8)
            prefix = 'block_{}_'.format(block_id)

            if block_id:
                # Expand
                x = tf.keras.layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
                x = tf.keras.layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
                x = tf.keras.layers.ReLU(6., name=prefix + 'expand_relu')(x)
            else:
                prefix = 'expanded_conv_'

            # Depthwise
            if stride == 2:
                x = tf.keras.layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, 3), name=prefix + 'pad')(x)
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise')(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
            x = tf.keras.layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

            # Project
            x = tf.keras.layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

            if in_channels == pointwise_filters and stride == 1:
                return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])
            return x

        first_block_filters = _make_divisible(32 * alpha, 8)
        x = tf.keras.layers.Conv2D(first_block_filters, kernel_size=3, strides=2, padding='same', use_bias=False, name='Conv1')(input_1)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
        self.Conv1_relu = tf.keras.layers.ReLU(6., name='Conv1_relu')(x)

        # block_0
        self.expanded_conv = _inverted_res_block(self.Conv1_relu, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

        # block_1
        self.block_1 = _inverted_res_block(self.expanded_conv, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
        # block_2
        self.block_2 = _inverted_res_block(self.block_1, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

        # block_3
        self.block_3 = _inverted_res_block(self.block_2, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
        # block_4
        self.block_4 = _inverted_res_block(self.block_3, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
        # block_5
        self.block_5 = _inverted_res_block(self.block_4, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

        # block_6
        self.block_6 = _inverted_res_block(self.block_5, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
        # block_7
        self.block_7 = _inverted_res_block(self.block_6, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
        # block_8
        self.block_8 = _inverted_res_block(self.block_7, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
        # block_9
        self.block_9 = _inverted_res_block(self.block_8, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

        # block_10
        self.block_10 = _inverted_res_block(self.block_9, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
        # block_11
        self.block_11 = _inverted_res_block(self.block_10, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
        # block_12
        self.block_12 = _inverted_res_block(self.block_11, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

        # out_relu
        if alpha > 1.0:
            last_block_filters = _make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280
        x = tf.keras.layers.Conv2D(filters=last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(self.block_12)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        self.out_relu = tf.keras.layers.ReLU(6., name='out_relu')(x)

    def build_model(self, model_type):

        # # 从encode部分取出4组特征图
        fm64 = self.block_2
        fm32 = self.block_5
        fm16 = self.block_9
        fm8 = self.out_relu

        fuse_1 = self.fuse_1(fm8, fm16)
        fuse_2 = self.fuse_2(fuse_1, fm32)
        fuse_3 = self.fuse_3(fuse_2, fm64)
        y = self.conv_final(fuse_3)

        heatmap = tf.keras.layers.Activation("sigmoid", name="heatmap")(y)

        if model_type == "HEATMAP":
            return Model(inputs=self.input_1, outputs=heatmap)
        else:
            raise ValueError("Wrong model type.")


if __name__ == '__main__':
    model = BlazePose(14).build_model("HEATMAP")
    print(model.summary())
    input_tensor = tf.random.normal((1, 192, 192, 3))
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
    gmacs = flops / 1e9
    print("FLOPs:", flops)
    print("GMACs:", gmacs)
