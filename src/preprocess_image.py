import tensorflow as tf

def preprocess_image(image, label, image_size=(32, 32), normalize=True):
    """Resize and normalize an image-label pair."""
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    if normalize:
        image /= 255.0
    return image, label
