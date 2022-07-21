from pathlib import Path
import shutil

import ray
import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import deque

# from model import AttentionModel as MyModel
from model import AttentionModel_2 as MyModel

import util


# @ray.remote
@ray.remote(num_cpus=1)
class Agent:
    def __init__(self, agent_id, env_name, n_frames, trajectory_length, sequence_length):
        self.agent_id = agent_id
        self.env_name = env_name
        self.n_frames = n_frames
        self.env = gym.make(env_name)
        self.trajectory_length = trajectory_length

        self.reset_state()  # self.state: (84,84,1)

        self.sequence_length = sequence_length

        self.state_seq = np.zeros(shape=(self.sequence_length, 84, 84, self.n_frames),
                                  dtype=np.float32)  # (50,84,84,1)
        self.state_seq[-1] = self.state  # (84,84,1)

        self.action_space = self.env.action_space.n  # 4
        self.policy = MyModel(self.action_space)
        self.policy.build(input_shape=(None, self.sequence_length, 84, 84, self.n_frames))

    def reset_state(self):
        frame = util.preprocess_frame(self.env.reset())  # (84,84)
        self.frames = deque([frame] * self.n_frames,
                            maxlen=self.n_frames)  # [(84,84)]
        self.state = np.stack(self.frames, axis=2)  # (84,84,1)
        # self.state = np.expand_dims(state, axis=0)  # (1,84,84,1)

    def collect_trajectory(self, weights):
        """
        蓄積したtrajectoryの回収
        trajectory["s"], trajectory["s2"]: (trj, seq,84,84,1)=(32,50,84,84,1)
        trajectory["a"], trajectory["r"], trajectory["dones"]: (trj,1)=(32,1)
        """
        trajectory = self.rollout(weights=weights)
        return trajectory

    def rollout(self, weights):
        self.policy.set_weights(weights=weights)

        # 1. 初期化
        trajectory = {}
        trajectory["s"] = \
            np.zeros((self.trajectory_length, self.sequence_length,
                      84, 84, self.n_frames)).astype(np.float32)  # (None,50,84,84,4)
        trajectory["a"] = np.zeros((self.trajectory_length, 1)).astype(np.float32)  # (None,1)
        trajectory["r"] = np.zeros((self.trajectory_length, 1)).astype(np.float32)  # (None,1)
        trajectory["s2"] = \
            np.zeros((self.trajectory_length, self.sequence_length,
                      84, 84, self.n_frames)).astype(np.float32)  # (None,50,84,84,4)
        trajectory["dones"] = np.zeros((self.trajectory_length, 1)).astype(np.float32)  # (None,1)

        done = False
        lives = 5

        # 2. Rollout実施
        for i in range(self.trajectory_length):
            action = self.policy.sample_action(
                np.expand_dims(self.state_seq, axis=0)
            )  # self.state_seq: (50,84,84,4)

            next_frame, reward, done, info = self.env.step(action)  # action: int

            self.frames.append(util.preprocess_frame(next_frame))  # [(84,84)]
            next_state = np.stack(self.frames, axis=2)  # (84,84,1)
            # next_state = np.expand_dims(next_state, axis=0)  # (1,84,84,1)

            trajectory["s"][i] = self.state_seq  # self.state_seq: (50,84,84,1)
            trajectory["a"][i] = action  # action: int32
            trajectory["r"][i] = reward  # reward: float

            self.state_seq = np.roll(self.state_seq, shift=-1, axis=0)  # (50,84,84,1)
            self.state_seq[-1] = next_state  # (84,84,1)

            trajectory["s2"][i] = self.state_seq  # (50,84,84,1)
            trajectory["dones"][i] = done  # done: bool

            # livesが減った時はdone=Trueと見做す
            if info["ale.lives"] != lives:
                lives = info["ale.lives"]
                trajectory["dones"][i] = True
            else:
                trajectory["dones"][i] = done  # bool

            if done:
                self.reset_state()
                self.state_seq = np.zeros(shape=(self.sequence_length, 84, 84, self.n_frames),
                                          dtype=np.float32)  # (50,84,84,1)
                self.state_seq[-1] = self.state  # (84,84,1)
                trajectory["s2"][i] = self.state_seq  # (50,84,84,1)

        return trajectory


def compute_advantage(trajectories, policy, r_running_stats, gamma, gae_lambda):
    for trajectory in trajectories:
        """
        GAEの計算, Vol.42, Chap.17, copied from TRPO/Pendulum-v0
        trajectory_length=8
        trajectory["s"], trajectory["s2"]: 
            (trajectory_length, sequence_length, OBS_DIM)=(32,50,84,84,4)
        trajectory["r"], trajectory["dones"]: (trajectory_length,1)=(32,1)
        trajectory["a"]: (trajectory_length,ACTION_DIM)=(32,1)
        
        trajectory["vpred"], trajectory["vpred_next"]: (trajectory_length,1)=(32,1)
        trajectory["advantage"], trajectory["R"]: (trajectory_length,1)=(32,1)
        """

        trajectory["vpred"], _, _, _ = policy(trajectory["s"])
        trajectory["vpred"] = trajectory["vpred"].numpy()  # v(s): (8,1)

        trajectory["vpred_next"], _, _, _ = policy(trajectory["s2"])
        trajectory["vpred_next"] = trajectory["vpred_next"].numpy()  # v(s'): (8,1)

        is_nonterminal = 1 - trajectory["dones"]  # (8,1)

        # r + gamma * (1-dones) * v(s') - v(s)
        # normed_rewards = (trajectory["r"] / (np.sqrt(r_running_stats.var) + 1e-4))
        normed_rewards = trajectory["r"]

        deltas = normed_rewards + \
                 gamma * is_nonterminal * trajectory["vpred_next"] - trajectory["vpred"]  # (8,1)

        advantages = np.zeros_like(deltas, dtype=np.float32)  # (8,1)
        last_gae = 0
        for i in reversed(range(len(deltas))):  # len(deltas)=80
            last_gae = deltas[i] + gamma * gae_lambda * is_nonterminal[i] * last_gae
            advantages[i] = last_gae

        # GAE
        trajectory["advantage"] = advantages  # (8,1)

        # 期待リターン
        trajectory["R"] = trajectory["advantage"] + trajectory["vpred"]  # (8,1)

    return trajectories


def create_minibatch(trajectories):
    """
    num_agents * trajectory_length = 12 * 32 = 384, sequence_length=50
    states:(384,50,84,84,1), actions:(384,1), advantages:(384,1), vtargets:(384,1)
    """
    states = np.vstack([traj["s"] for traj in trajectories])
    actions = np.vstack([traj["a"] for traj in trajectories])
    advantages = np.vstack([traj["advantage"] for traj in trajectories])
    vtargets = np.vstack([traj["R"] for traj in trajectories])

    return states, actions, advantages, vtargets


def compute_logprobs(action_probs, actions, action_space):
    """
    action_probs, actions: (BATCH_SIZE,2)=(128,4)
    """
    selected_actions_onehot = \
        tf.one_hot(actions, depth=action_space, dtype=tf.float32)  # (128,4)

    log_probs = selected_actions_onehot * tf.math.log(action_probs + 1e-5)  # (128,4)

    selected_actions_log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)  # (128,1)

    return selected_actions_log_probs


def update_network(optimizer, policy, old_policy, states, actions, advantages, vtargets,
                   opt_iter, batch_size, clip_range, action_space, entropy_coef):
    """
    states
    """
    losses = []
    plosses = []
    vlosses = []
    entropies = []

    old_policy.set_weights(policy.get_weights())

    indices = np.random.choice(range(states.shape[0]), size=(opt_iter, batch_size))  # (3,128)

    for i in range(opt_iter):
        idx = indices[i]  # (128,)

        old_vpred, old_probs, _, _ = old_policy(states[idx])  # (128,1),(128,4)
        old_log_probs = compute_logprobs(old_probs, actions[idx, 0], action_space)  # (128,1)

        with tf.GradientTape() as tape:
            vpred, new_probs, _, _ = policy(states[idx])  # (128,1),(128,4)
            new_log_probs = compute_logprobs(new_probs, actions[idx, 0], action_space)  # (128,1)

            """ Compute policy loss """
            ratio = tf.exp(new_log_probs - old_log_probs)  # (128,1)

            ratio_clipped = tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range)  # (128,1)

            ploss_unclipped = ratio * advantages[idx]  # (128,1)
            ploss_clipped = ratio_clipped * advantages[idx]  # (128,1)

            ploss = tf.minimum(ploss_unclipped, ploss_clipped)  # (128,1)
            ploss = -tf.reduce_mean(ploss)

            """ Compute value loss """
            vpred_clipped = \
                old_vpred + tf.clip_by_value(vpred - old_vpred, -clip_range, clip_range)  # (128,1)

            vloss = tf.maximum(tf.square(vpred - vtargets[idx]),
                               tf.square(vpred_clipped - vtargets[idx]))  # (128,1)
            vloss = tf.reduce_mean(vloss)

            """ Compute entropy """
            entropy = - new_probs * tf.math.log(new_probs + 1e-5)  # (128,4)
            entropy = tf.reduce_mean(entropy, axis=1, keepdims=True)  # (128,1)
            entropy = tf.reduce_mean(entropy)

            """ Compute loss """
            loss = 1. * ploss + 0.5 * vloss - 1. * entropy_coef * entropy

        grads = tape.gradient(loss, policy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)

        optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        losses.append(loss)
        plosses.append(ploss)
        vlosses.append(vloss)
        entropies.append(entropy)

    return np.array(losses).mean(), np.array(plosses).mean(), np.array(vlosses).mean(), \
           np.array(entropies).mean(), policy


def learn(num_agents=10,
          env_name='BreakoutDeterministic-v4',
          n_frames=1,
          gamma=0.98,
          gae_lambda=0.95,
          clip_range=0.2,
          opt_iter=2,
          entropy_coef=0.01,
          trajectory_length=8,
          sequence_length=50,
          batch_size=768,
          num_update=5000000,
          lr=1e-4):
    env = gym.make(env_name)
    action_space = env.action_space.n

    # Make a learner network as global_policy
    global_policy = MyModel(action_space)
    global_policy.build(input_shape=(None, sequence_length, 84, 84, n_frames))
    global_policy.summary()

    old_global_policy = MyModel(action_space)
    old_global_policy.build(input_shape=(None, sequence_length, 84, 84, n_frames))

    # r_running_stats = util.RunningStats(shape=(action_space,))  # Originalはこうだが？
    r_running_stats = util.RunningStats(shape=(1,))

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Make worker agents
    agents = \
        [Agent.remote(agent_id=i, env_name=env_name, n_frames=n_frames,
                      trajectory_length=trajectory_length,
                      sequence_length=sequence_length)
         for i in range(num_agents)]

    # Make log dir
    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)

    summary_writer = tf.summary.create_file_writer(logdir=str(logdir))

    # Make result dir
    resultdir = Path(__file__).parent / "result"
    if resultdir.exists():
        shutil.rmtree(resultdir)

    """ Training loop """
    for n in tqdm(range(num_update)):
        """ global_policyから、重みを取得 """
        weights = global_policy.get_weights()

        """ agent毎にRolloutを実施しtrajectoryを蓄積。蓄積されたtrajectoryを収集 """
        # trajectories = [trajectory, trajectory, ..., trajectory], len=num_agents=5
        trajectories = ray.get([agent.collect_trajectory.remote(weights) for agent in agents])

        """ 即時報酬をスケーリングするrの標準偏差を更新 """
        for trajectory in trajectories:
            r_running_stats.update(trajectory["r"])

        """ GAE, 期待リターンRをtrajectoryに追加 """
        trajectories = \
            compute_advantage(trajectories, global_policy, r_running_stats, gamma, gae_lambda)

        """ trajectoriesを整理し直して、バッチの形に変更 """
        states, actions, advantages, vtargets = create_minibatch(trajectories)

        """ global_policy 更新 """
        loss, policy_loss, value_loss, entropy, global_policy = \
            update_network(optimizer, global_policy, old_global_policy,
                           states, actions, advantages, vtargets,
                           opt_iter, batch_size, clip_range, action_space, entropy_coef)

        """ post process """
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=n)
            tf.summary.scalar("policy_loss", policy_loss, step=n)
            tf.summary.scalar("value_loss", value_loss, step=n)
            tf.summary.scalar("entropy", entropy, step=n)
            tf.summary.scalar("advantage", advantages.mean(), step=n)

        if n % 50 == 0:
            episode_rewards = test_play(global_policy, sequence_length, env_name, n_frames)
            with summary_writer.as_default():
                tf.summary.scalar("test_reward", episode_rewards, step=n)

        if n % 1000 == 0:
            model_name = "/policy_" + str(n)
            global_policy.save(str(resultdir) + model_name)


def test_play(policy, sequence_length, env_name, n_frames, monitor_dir=None):
    env = gym.make(env_name)

    frame = util.preprocess_frame(env.reset())  # (84,84)
    frames = deque([frame] * n_frames, maxlen=n_frames)  # [(84,84)]
    state = np.stack(frames, axis=2)  # (84,84,1)
    # state = np.expand_dims(state, axis=0)  # (1,84,84,1)

    state_seq = np.zeros(shape=(sequence_length, 84, 84, n_frames),
                         dtype=np.float32)  # (50,84,84,1)
    state_seq[-1] = state  # (84,84,1)

    total_rewards = 0

    while True:
        action = policy.sample_action(np.expand_dims(state_seq, axis=0))
        next_frame, reward, done, _ = env.step(action)

        frames.append(util.preprocess_frame(next_frame))  # [(84,84)]
        next_state = np.stack(frames, axis=2)  # (84,84,1)
        # next_state = np.expand_dims(next_state, axis=0)  # (1,84,84,4)

        total_rewards += reward

        if done:
            break
        else:
            state_seq = np.roll(state_seq, shift=-1, axis=0)  # (50,84,84,1)
            state_seq[-1] = next_state  # (84,84,1)

    return total_rewards


if __name__ == '__main__':
    print("ray version:", ray.__version__)
    ray.init(local_mode=False)  # True for debug

    learn(num_agents=12,  # default=12
          trajectory_length=32,
          sequence_length=10,
          batch_size=128,
          opt_iter=3)

    ray.shutdown()
