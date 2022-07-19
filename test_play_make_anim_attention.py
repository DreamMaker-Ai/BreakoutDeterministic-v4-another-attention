import os.path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ray
import gym
import numpy as np
import tensorflow as tf
import util
from collections import deque

from big_architecture.model import AttentionModel as MyModel


class MakeAnimation_Attention:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.scores_sequence = []
        self.num_heads = None

    def add_frame(self, scores):
        """ scores: (1, heads, seq_len, seq_len)=(1,4,50,50)
            -> (num_heads, seq_len, seq-len) """

        scores = np.squeeze(scores, axis=0)  # (4,50,50)
        scores = np.expand_dims(scores, axis=3)  # (4,50,50,1)
        scores = np.concatenate([scores, scores, scores], axis=3)  # (4,50,50,3), RGB

        self.scores_sequence.append(scores)
        self.num_heads = scores.shape[0]

    def generate_movies(self, heads_type):
        """ call this mothod """
        ims = []

        num_subplots = self.num_heads

        num_colms = int(np.ceil(np.sqrt(num_subplots)))
        num_rows = int(np.ceil(num_subplots / num_colms))

        # define fig & ax
        fig, ax = plt.subplots(num_rows, num_colms, figsize=(9.6, 9.6), tight_layout=True)

        # prepare subplots
        for row in range(num_rows):
            for colm in range(num_rows):
                head = row * num_rows + colm

                ax[(row, colm)].set_title(str(heads_type + '_' + str(head)))
                ax[(row, colm)].tick_params(
                    labelbottom=False, bottom=False, labelleft=False, left=False)

        # imshow subplots
        for idx in range(len(self.scores_sequence)):
            im = []
            for row in range(num_rows):
                for colm in range(num_rows):
                    head = row * num_rows + colm

                    img = ax[(row, colm)].imshow(self.scores_sequence[idx][head],
                                                 vmin=0, vmax=1, animated=True)
                    im += [img]

            ims.append(im)

        # make animation from ims
        anim = animation.ArtistAnimation(fig, ims,
                                         interval=50, blit=True,
                                         repeat_delay=3000, repeat=True)

        anim.save(self.save_dir + '/' + heads_type + '.mp4', writer='ffmpeg')


def test_play(policy, sequence_length, env_name, n_frames, save_dir=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    env = gym.make(env_name)

    frame = util.preprocess_frame(env.reset())  # (84,84)
    frames = deque([frame] * n_frames, maxlen=n_frames)  # [(84,84)]
    state = np.stack(frames, axis=2)  # (84,84,1)
    # state = np.expand_dims(state, axis=0)  # (1,84,84,1)

    state_seq = np.zeros(shape=(sequence_length, 84, 84, n_frames),
                         dtype=np.float32)  # (50,84,84,1)
    state_seq[-1] = state  # (84,84,1)

    total_rewards = 0

    make_animation_attention1 = MakeAnimation_Attention(save_dir)
    make_animation_attention2 = MakeAnimation_Attention(save_dir)

    while True:
        action = policy.sample_action(np.expand_dims(state_seq, axis=0))  # (1,50,4)
        next_frame, reward, done, _ = env.step(action)

        frames.append(util.preprocess_frame(next_frame))  # [(84,84)]
        next_state = np.stack(frames, axis=2)  # (84,84,1)
        # next_state = np.expand_dims(next_state, axis=0)  # (1,84,84,4)

        total_rewards += reward

        # get attention scores: anim_frame1, anim_frame2: (1,heds,seq,seq)=(1,4,50,50)
        _, _, anim_frame1, anim_frame2 = policy(np.expand_dims(state_seq, axis=0))

        make_animation_attention1.add_frame(scores=anim_frame1)
        make_animation_attention2.add_frame(scores=anim_frame2)

        if done:
            break
        else:
            state_seq = np.roll(state_seq, shift=-1, axis=0)  # (50,4)
            state_seq[-1] = next_state  # (4,)

    make_animation_attention1.generate_movies('val_heads')
    make_animation_attention2.generate_movies('policy_heads')

    return total_rewards


def main():
    env_name = 'BreakoutDeterministic-v4'
    sequence_length = 10  # same to 'main.py'
    n_frames = 1  # same to 'main.py'

    env = gym.make(env_name)
    action_space = env.action_space.n

    policy = MyModel(action_space)
    policy.build(input_shape=(None, sequence_length, 84, 84, n_frames))

    model_dir = 'big_architecture/result/policy_12000'
    tester = tf.keras.models.load_model(model_dir, compile=False)
    weights = tester.get_weights()

    policy.set_weights(weights)

    save_dir = 'anim_attention'

    test_play(policy, sequence_length=sequence_length, env_name=env_name,
              n_frames=n_frames, save_dir=save_dir)


if __name__ == '__main__':
    print("ray version:", ray.__version__)
    ray.init(local_mode=False)  # True for debug

    main()

    ray.shutdown()
