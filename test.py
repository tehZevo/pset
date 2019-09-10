#it's a WIP

import numpy as np
import tensorflow as tf
import gym

from keras.layers import Input, Dense
from keras.models import Model

#TODO: let it run for a few steps before prediction
from ml_utils.viz import save_plot, viz_weights
from ml_utils.keras import apply_regularization

from pset.pset import create_traces, get_action_and_gradient, step_weights

def build_model(input_size, output_size, hidden_sizes=[32, 32], hidden_acti="relu", out_acti="tanh"):
  inputs = Input([input_size])
  x = inputs
  for size in hidden_sizes:
    x = Dense(size, activation=hidden_acti)(x)
  x = Dense(output_size, activation=out_acti)(x)
  return Model(inputs, x)


sampler = lambda p: np.random.choice(len(p), p=p)

#setup
env = gym.make("CartPole-v0")
#env = gym.make("LunarLander-v2")
is_discrete = env.action_space.shape == ()

input_size = env.observation_space.shape[0]
output_size = env.action_space.n if is_discrete else env.action_space.shape[0]
out_acti = "softmax" if is_discrete else "tanh"
#TODO: rescale action for continuous environments

#hparms
alpha = 1e-2
gamma = 1e-3
_lambda = 0.99
tau = 1e-2 #probably depends on input/output/matrix sizes...
epsilon = 1e-7 #fuzz factor
episode_end_reward = -1
reset_traces = True
alternate_trace_method = True #False
clip_advantage = None #1 #None
use_advantage = True
activation = "tanh" #"relu"
scale_exploration = False #True
regularization = 0#1e-5 * alpha
#print(regularization)

#runtime vars
model = build_model(input_size, output_size, [32, 32], hidden_acti=activation, out_acti=out_acti)
traces = create_traces(model)
state = env.reset()
reward_mean = 0
reward_deviation = 10

#metrics
episode_rewards = []
episode_reward = 0
step_actions = []
episode_actions = []
graph_episodes = 100
quantile = 0.00 #for reward graph

while True:
  #get action and delta weights
  action, d_theta = get_action_and_gradient(model, state, tau, scale_exploration)

  step_actions.append(action)

  #accumulate d_thetas into trace
  if alternate_trace_method:
    traces = [t * _lambda + dt for t, dt in zip(traces, d_theta)]
  else:
    traces = [t * _lambda + dt * (1 - _lambda) for t, dt in zip(traces, d_theta)]

  # step environment, get new state and reward
  #a = np.argmax(action) if is_discrete else action
  a = sampler(action) if is_discrete else action
  state, reward, done, info = env.step(a)

  if done:
    reward += episode_end_reward

  episode_reward += reward

  # calculate advantage
  d_reward = reward - reward_mean
  #technically, dont think we need to add epsilon since clipping infinity is well behaved
  advantage = d_reward / (reward_deviation + epsilon)
  if clip_advantage is not None:
    advantage = np.clip(advantage, -clip_advantage, clip_advantage)

  if not use_advantage:
    advantage = reward

  apply_regularization(model, regularization)
  step_weights(model, traces, advantage, alpha)

  # update running mean/absolute deviation
  reward_mean += d_reward * gamma
  reward_deviation += (np.abs(d_reward) - reward_deviation) * gamma

  if done:
    state = env.reset()
    episode_rewards.append(episode_reward)
    print("{}: {} ({}, {}) [{}]".format(
      len(episode_rewards),
      episode_reward,
      np.round(reward_mean, 2),
      np.round(reward_deviation, 2),
      np.round(np.mean([np.mean(np.abs(t)) for t in traces]), 6)
    ))
    episode_reward = 0
    if reset_traces:
      traces = create_traces(model)

    aaa = np.mean(step_actions, 0)
    episode_actions.append(aaa)
    step_actions = []

    if len(episode_rewards) % graph_episodes == 0:
      save_plot(episode_rewards, "Episode rewards", 0.01, q=quantile)

      save_plot(episode_actions, "Actions", 0.01)

      viz_weights(model.get_weights(), "weights.png")
