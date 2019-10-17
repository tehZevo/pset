import numpy as np
import tensorflow as tf
import gym

#WIP

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential

#TODO: let it run for a few steps before prediction
from ml_utils.viz import save_plot, viz_weights
from ml_utils.keras import apply_regularization

from pset.pset import create_traces
from pset.agent import Agent

tf.enable_eager_execution()

model = Sequential([
  Dense(64, input_shape=(8,), activation="relu"),
  Dense(4, activation="softmax"),
])

#setup
#env = gym.make("CartPole-v0")
env = gym.make("LunarLander-v2")
is_discrete = env.action_space.shape == ()

#hparms
alpha = 1e-2
gamma = 1e-3
_lambda = 0.99
tau = 1e-2 #probably depends on input/output/matrix sizes...
epsilon = 1e-7 #fuzz factor
episode_end_reward = -1
alternate_trace_method = True #False
clip_advantage = 1 #None
regularization = 0#1e-5 * alpha
reset_traces = False
#print(regularization)
action_repeat = 4 #TODO
#runtime vars

episodes_left = 9999999999

agent = Agent(model, alt_trace_method=alternate_trace_method, epsilon=epsilon,
  advantage_clip=clip_advantage, gamma=gamma, lr=alpha, lambda_=_lambda,
  regularization_scale=regularization, optimizer="adam", noise=0.1,
  initial_deviation=10, use_xavier=True)

#metrics
episode_rewards = []
episode_reward = 0
step_actions = []
episode_actions = []
graph_episodes = 100
quantile = 0.00 #for reward graph

state = env.reset()

while True:
  action = agent.get_action(state)

  step_actions.append(action)

  # step environment, get new state and reward
  a = np.argmax(action)
  reward = 0
  for i in range(action_repeat):
    state, r, done, info = env.step(a)
    reward += r
    env.render()
    if done:
      break

  if done:
    reward += episode_end_reward

  episode_reward += reward

  agent.train(reward)
  env.render()

  if done:
    episodes_left -= 1
    if episodes_left < 0:
      break
    state = env.reset()
    episode_rewards.append(episode_reward)
    print("{}: {} ({}, {})".format(
      len(episode_rewards),
      episode_reward,
      np.round(agent.reward_mean, 2),
      np.round(agent.reward_deviation, 2)
    ))
    episode_reward = 0
    if reset_traces:
      agent.traces = create_traces(model)

    aaa = np.mean(step_actions, 0)
    episode_actions.append(aaa)
    step_actions = []

    if len(episode_rewards) % graph_episodes == 0:
      save_plot(episode_rewards, "Episode rewards", 0.01, q=quantile)
      save_plot(episode_actions, "Actions", 0.01)
      viz_weights(model.get_weights(), "weights.png")
