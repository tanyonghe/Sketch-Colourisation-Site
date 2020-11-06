# import required libraries
import os
import tensorflow as tf


# generator model configurations
OUTPUT_CHANNELS = 3
IMG_HEIGHT = 256
IMG_WIDTH = 256


def downsample(filters, size, shape, apply_batchnorm=True):
    """
    Returns a downsampling layer followed by batch normalization (added by default) and Leaky ReLU activation function.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', batch_input_shape=shape, 
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, shape, apply_dropout=False):
    """
    Returns an upsampling layer followed by dropout (not added by default) and ReLU activation function.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2, batch_input_shape=shape,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def buildGenerator():
    """
    Returns a generator model containing downsampling layers, upsampling layers, and its skip connections.
    
    The generator part of a GAN learns to create fake data by incorporating feedback from the discriminator. 
    It learns to make the discriminator classify its output as real.
    """
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 4, (None, 256, 256, 3), apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4, (None, 128, 128, 64)), # (bs, 64, 64, 128)
        downsample(256, 4, (None, 64, 64, 128)), # (bs, 32, 32, 256)
        downsample(512, 4, (None, 32, 32, 256)), # (bs, 16, 16, 512)
        downsample(512, 4, (None, 16, 16, 512)), # (bs, 8, 8, 512)
        downsample(512, 4, (None, 8, 8, 512)), # (bs, 4, 4, 512)
        downsample(512, 4, (None, 4, 4, 512)), # (bs, 2, 2, 512)
        downsample(512, 4, (None, 2, 2, 512)), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, (None, 1, 1, 512), apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, (None, 2, 2, 1024), apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, (None, 4, 4, 1024), apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4, (None, 8, 8, 1024)), # (bs, 16, 16, 1024)
        upsample(256, 4, (None, 16, 16, 1024)), # (bs, 32, 32, 512)
        upsample(128, 4, (None, 32, 32, 512)), # (bs, 64, 64, 256)
        upsample(64, 4, (None, 64, 64, 256)), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
    
def resize(image, height, width):
    image = tf.image.resize(image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

def normalize(image):
    image = (image / 127.5) - 1
    return image

def unnormalize(image):
    image = (image + 1) * 127.5
    return image

def save_image(image, filename, height, width):
    image = resize(image, height, width)
    image = tf.cast(unnormalize(image), tf.uint8)
    image = tf.image.encode_png(image)
    tf.io.write_file(app.config['DOWNLOAD_FOLDER'] + filename, image)
    return image

def generate_image(model, input):
    prediction = model(input, training=False)
    return prediction[0]

def load_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32) 
    return image

def process_image(image):
    image = load_image(image)
    height, width, _ = image.shape
    if height > width:
      scaled_height = (height // IMG_HEIGHT + 1) * IMG_HEIGHT
      image = resize(image, scaled_height, scaled_height)
    else:
      scaled_width = (width // IMG_WIDTH + 1) * IMG_WIDTH
      image = resize(image, scaled_width, scaled_width)
    image = np.float32(normalize(image))[:,:,:3]
    return image, height, width

def process_file(path, filename):
    image, height, width = process_image(os.path.join(path, filename))
    image = generate_image(new_generator, tf.expand_dims(image, 0))
    save_image(image, filename, height, width)