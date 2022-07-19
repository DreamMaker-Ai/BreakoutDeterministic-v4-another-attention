import numpy as np
import tensorflow as tf


def preprocess_frame(frame):
    # frame: (210,160,3), uint8
    image = tf.cast(tf.convert_to_tensor(frame), tf.float32)  # (210,160,3), float32
    image_gray = tf.image.rgb_to_grayscale(image)  # (210,160,1)
    image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)  # (160,150,1)
    image_resize = tf.image.resize(image_crop, [84, 84])  # (84,84,1)
    image_scaled = tf.divide(image_resize, 255)  # (84,84,1)

    frame = image_scaled.numpy()[:, :, 0]  # (84,84)

    return frame


class RunningStats:
    """baselinesの実装より
        https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    """

    def __init__(self, shape=None):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 0 + 1e-4  #: サンプル数

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    @staticmethod
    def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


if __name__ == "__main__":
    stats = RunningStats(shape=(1,))
    x = np.arange(10).reshape(-1, 1)
    stats.update(x)
    print(stats.mean, stats.var, stats.count)
    x = np.arange(10).reshape(-1, 1)
    stats.update(x)
    print(stats.mean, stats.var, stats.count)
