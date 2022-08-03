import numpy as np
import tensorflow as tf


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def compute_paddings(resample_kernel, up, down, kernel_size=None, factor=2, gain=1):
    assert not (up and down)

    if kernel_size is not None:
        is_conv = True
    else:
        is_conv = False
    k = [1] * factor if resample_kernel is None else resample_kernel
    if up:
        k = _setup_kernel(k) * (gain * (factor ** 2))
        if is_conv:
            p = (k.shape[0] - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
        else:
            p = k.shape[0] - factor
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2
    elif down:
        k = _setup_kernel(k) * gain
        if is_conv:
            p = (k.shape[0] - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
        else:
            p = k.shape[0] - factor
            pad0 = (p + 1) // 2
            pad1 = p // 2
    else:
        k = resample_kernel
        pad0, pad1 = 0, 0
    return k, pad0, pad1


def upsample_2d(x, resample_kernel, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    x_res = x.shape[2]
    k, pad0, pad1 = compute_paddings(resample_kernel, up=True, down=False, factor=factor, gain=gain)
    return _simple_upfirdn_2d(x, x_res, k, up=factor, pad0=pad0, pad1=pad1)


def downsample_2d(x, resample_kernel, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    x_res = x.shape[2]
    k, pad0, pad1 = compute_paddings(resample_kernel, up=False, down=True, factor=factor, gain=gain)
    return _simple_upfirdn_2d(x, x_res, k, down=factor, pad0=pad0, pad1=pad1)


def upsample_conv_2d(x, w, kernel_size, resample_kernel, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    x_res = x.shape[2]
    # Check weight shape.
    w = tf.convert_to_tensor(w)
    assert w.shape.rank == 4
    # convH = w.shape[0]
    # convW = w.shape[1]
    inC = tf.shape(w)[2]
    outC = tf.shape(w)[3]

    convW = kernel_size
    convH = kernel_size


    # Determine data dimensions.
    stride = [1, 1, factor, factor]
    output_shape = [tf.shape(x)[0], outC, (x_res - 1) * factor + convH, (x_res - 1) * factor + convW]
    num_groups = tf.shape(x)[1] // inC

    # Transpose weights.
    w = tf.reshape(w, [convH, convW, inC, num_groups, -1])
    w = tf.transpose(w[::-1, ::-1], [0, 1, 4, 3, 2])
    w = tf.reshape(w, [convH, convW, -1, num_groups * inC])

    # Execute.
    x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=stride, padding='VALID', data_format='NCHW')
    new_x_res = output_shape[2]

    k, pad0, pad1 = compute_paddings(resample_kernel, up=True, down=False, kernel_size=kernel_size, factor=factor, gain=gain)

    return _simple_upfirdn_2d(x, new_x_res, k, pad0=pad0, pad1=pad1)


def conv_downsample_2d(x, w, kernel_size, resample_kernel, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    x_res = x.shape[2]
    w = tf.convert_to_tensor(w)
    # convH, convW, _inC, _outC = w.shape.as_list()
    convW = kernel_size
    convH = kernel_size

    s = [1, 1, factor, factor]
    k, pad0, pad1 = compute_paddings(resample_kernel, up=False, down=True, kernel_size=kernel_size, factor=factor, gain=gain)
    x = _simple_upfirdn_2d(x, x_res, k, pad0=pad0, pad1=pad1)
    return tf.nn.conv2d(x, w, strides=s, padding='VALID', data_format='NCHW')


def _simple_upfirdn_2d(x, x_res, k, up=1, down=1, pad0=0, pad1=0):
    assert x.shape.rank == 4
    y = x
    y = tf.reshape(y, [-1, x_res, x_res, 1])
    y = upfirdn_ref(y, k, up_x=up, up_y=up, down_x=down, down_y=down, pad_x0=pad0, pad_x1=pad1, pad_y0=pad0, pad_y1=pad1)
    y = tf.reshape(y, [-1, tf.shape(x)[1], tf.shape(y)[1], tf.shape(y)[2]])
    return y


def upfirdn_ref(x, k, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    in_height, in_width = tf.shape(x)[1], tf.shape(x)[2]
    minor_dim = tf.shape(x)[3]
    kernel_h, kernel_w = k.shape

    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, in_height, 1, in_width, 1, minor_dim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, up_y - 1], [0, 0], [0, up_x - 1], [0, 0]])
    x = tf.reshape(x, [-1, in_height * up_y, in_width * up_x, minor_dim])

    # Pad (crop if negative).
    x = tf.pad(x, [
        [0, 0],
        [tf.math.maximum(pad_y0, 0), tf.math.maximum(pad_y1, 0)],
        [tf.math.maximum(pad_x0, 0), tf.math.maximum(pad_x1, 0)],
        [0, 0]
    ])
    x = x[:, tf.math.maximum(-pad_y0, 0): tf.shape(x)[1] - tf.math.maximum(-pad_y1, 0),
          tf.math.maximum(-pad_x0, 0): tf.shape(x)[2] - tf.math.maximum(-pad_x1, 0), :]

    # Convolve with filter.
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, 1, in_height * up_y + pad_y0 + pad_y1, in_width * up_x + pad_x0 + pad_x1])
    w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
    x = tf.reshape(x, [-1,
                       minor_dim,
                       in_height * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                       in_width * up_x + pad_x0 + pad_x1 - kernel_w + 1])
    x = tf.transpose(x, [0, 2, 3, 1])

    # Downsample (throw away pixels).
    return x[:, ::down_y, ::down_x, :]


def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = tf.reshape(x, shape=[-1, C, H, 1, W, 1])
    x = tf.tile(x, multiples=[1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, shape=[-1, C, H * factor, W * factor])
    return x


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = tf.reshape(x, shape=[-1, C, H // factor, factor, W // factor, factor])
    x = tf.reduce_mean(x, axis=[3, 5])
    return x
