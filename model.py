import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class AttentionModel(tf.keras.Model):
    """ Attention model """

    def __init__(self, action_space):
        super(AttentionModel, self).__init__()

        self.action_space = action_space

        # Value module 1
        self.conv1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4,
                                   activation='relu',
                                   kernel_initializer='Orthogonal')
        )

        self.conv2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                   activation='relu',
                                   kernel_initializer='Orthogonal')
        )

        self.conv3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                   activation='relu',
                                   kernel_initializer='Orthogonal')
        )

        self.flatten1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
        )

        self.dense1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=512, activation='relu', kernel_initializer='Orthogonal'))

        def get_source_layer1(tensor):
            return tf.expand_dims(tensor[:, -1, :], axis=1)

        def get_target_layer1(tensor):
            return tensor[:, :-1, :]

        self.source1 = tf.keras.layers.Lambda(get_source_layer1, output_shape=(-1, 512))
        self.target1 = tf.keras.layers.Lambda(get_target_layer1, output_shape=(-1, 512))

        self.attention1 = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=512
        )

        self.flatten2 = tf.keras.layers.Flatten()

        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # Policy module 1
        self.conv10 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4,
                                   activation='relu',
                                   kernel_initializer='Orthogonal')
        )

        self.conv20 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,
                                   activation='relu',
                                   kernel_initializer='Orthogonal')
        )

        self.conv30 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                   activation='relu',
                                   kernel_initializer='Orthogonal')
        )

        self.flatten10 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
        )

        self.dense10 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=512, activation='relu', kernel_initializer='Orthogonal'))

        def get_source_layer10(tensor):
            return tf.expand_dims(tensor[:, -1, :], axis=1)

        def get_target_layer10(tensor):
            return tensor[:, :-1, :]

        self.source10 = tf.keras.layers.Lambda(get_source_layer10, output_shape=(-1, 512))
        self.target10 = tf.keras.layers.Lambda(get_target_layer10, output_shape=(-1, 512))

        self.attention10 = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=512
        )

        self.flatten20 = tf.keras.layers.Flatten()

        self.concat10 = tf.keras.layers.Concatenate(axis=-1)

        # Value module 2
        self.dense4 = tf.keras.layers.Dense(
            units=512,
            activation='relu',
            kernel_initializer='Orthogonal'
        )

        self.value = tf.keras.layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer='Orthogonal'
        )

        # Policy module 2
        self.dense40 = tf.keras.layers.Dense(
            units=512,
            activation='relu',
            kernel_initializer='Orthogonal'
        )

        self.logits = tf.keras.layers.Dense(
            units=self.action_space,
            activation='linear',
            kernel_initializer='Orthogonal'
        )

        self.softmax = tf.keras.layers.Softmax()

    @tf.function
    def call(self, x):  # x:(3,5,84,84,4)

        # Value module 1
        x0 = self.conv1(x)  # (3,5,20,20,32)
        x0 = self.conv2(x0)  # (3,5,9,9,64)
        x0 = self.conv3(x0)  # (3,5,7,7,64)
        x0 = self.flatten1(x0)  # (3,5,3136)

        x1 = self.dense1(x0)  # (3,5,64)

        source1 = self.source1(x1)  # (3,1,512)
        target1 = self.target1(x1)  # (3,4,512)

        x2, score1 = self.attention1(source1, target1,
                                     return_attention_scores=True)  # (3,1,512), (3,4,1,4)

        x4 = self.flatten2(x2)  # (3,512)

        x5 = self.concat1([x1[:, -1, :], x4])  # (3,1024)

        y1 = self.dense4(x5)  # (3,512)
        val = self.value(y1)  # (3,1)

        # Policy module 1
        x00 = self.conv10(x)  # (3,5,20,20,32)
        x00 = self.conv20(x00)  # (3,5,9,9,64)
        x00 = self.conv30(x00)  # (3,5,7,7,64)
        x00 = self.flatten10(x00)  # (3,5,3136)

        x10 = self.dense10(x00)  # (3,5,512)

        source10 = self.source10(x10)  # (3.1,512)
        target10 = self.target10(x10)  # (3,4,512)

        x20, score10 = self.attention10(source10, target10,
                                        return_attention_scores=True)  # (3,5,512), (3,4,5,5)

        x40 = self.flatten20(x20)  # (3,512)

        x50 = self.concat10([x10[:, -1, :], x40])  # (3,1024)

        y10 = self.dense40(x50)  # (3,512)
        logits = self.logits(y10)  # (3,action=space)=(3,2)
        action_prob = self.softmax(logits)  # (3,2)

        return val, action_prob, score1, score10

    def sample_action(self, state):
        """
        states: (batch, time, feature)
        """
        _, action_prob, _, _ = self(state)

        cdist = tfp.distributions.Categorical(probs=action_prob)
        action = cdist.sample()

        return action.numpy()[0]


if __name__ == '__main__':
    """ model.summary() and plot_model() 
        Need comment out 'tf.function', otherwise cause error !! """
    seq_len = 5
    n_frames = 4
    dummy_input = tf.keras.layers.Input(shape=(seq_len, 84, 84, n_frames))

    policy = AttentionModel(action_space=2)
    policy = tf.keras.models.Model(inputs=dummy_input,
                                   outputs=policy.call(dummy_input))
    policy.summary()

    tf.keras.utils.plot_model(policy,
                              to_file='attention_model.png',
                              show_shapes=True,
                              show_layer_activations=True)

    """ agent build test """
    states = np.random.rand(3, seq_len, 84, 84, n_frames)
    states.astype(np.float32)

    agent = AttentionModel(action_space=2)
    agent(states)

    """ agent sample_action test """
    state = np.random.rand(1, seq_len, 84, 84, n_frames)

    for _ in range(5):
        act = agent.sample_action(state)
        print(act, " ", end="")
