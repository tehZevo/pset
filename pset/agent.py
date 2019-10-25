import tensorflow as tf
import numpy as np

from ml_utils.keras import get_states, set_states, apply_regularization
from ml_utils.model_builders import dense_stack

from .pset import create_traces, get_action_and_gradient, step_weights_opt

from pget import explore_continuous, explore_discrete, explore_multibinary

#TODO: saving/loading?
#TODO: args/kwargs for get_action/train, maybe accept "done" in train

class Agent():
  """Note: requires TF eager"""
  def __init__(self, model, optimizer=None, action_type="continuous",
      alt_trace_method=False, epsilon=1e-7, advantage_clip=1, gamma=0.99,
      lambda_=0.9, regularization=1e-6, noise=0.1, initial_deviation=10,
      use_xavier=True, use_squared_deviation=True):
    self.model = model

    #TODO: is this needed?
    self.input_shape = tuple(self.model.input_shape[1:])
    self.output_shape = tuple(self.model.output_shape[1:])

    #hyperparameters
    self.eps = epsilon
    self.advantage_clip = advantage_clip
    self.gamma = gamma
    self.lambda_ = lambda_
    self.alt_trace_method = alt_trace_method
    self.regularization = regularization
    self.noise = noise
    self.use_xavier = use_xavier
    self.action_type = action_type.lower()
    self.optimizer = (optimizer if optimizer is not None else
      tf.keras.optimizers.Adam(1e-3, clipnorm=1.0))
    self.use_squared_deviation = use_squared_deviation

    self.last_advantage = 0

    #initialization
    self.traces = create_traces(self.model)
    self.reward_mean = 0
    self.reward_deviation = initial_deviation

  def get_action(self, state):
    #housekeeping
    state = state.astype("float32")
    #calc action from state
    action, dtheta = get_action_and_gradient(self.model, state, self.noise, self.use_xavier)

    #update traces
    if self.alt_trace_method:
      self.traces = [t * self.lambda_ + dt for t, dt in zip(self.traces, dtheta)]
    else:
      self.traces = [t * self.lambda_ + dt * (1 - self.lambda_) for t, dt in zip(self.traces, dtheta)]

    #onehot encode actual action (via boltzmann exploration)
    if self.action_type == "discrete":
      action = explore_discrete(action, 0)
    elif self.action_type == "multibinary":
      action = explore_multibinary(action, 0)

    return action

  def train(self, reward):
    #scale/clip reward to calculate advantage
    delta_reward = reward - self.reward_mean
    advantage = delta_reward / (self.reward_deviation + self.eps)
    if self.advantage_clip is not None:
      advantage = np.clip(advantage, -self.advantage_clip, self.advantage_clip)

    #update reward mean/deviation
    self.reward_mean += delta_reward * (1 - self.gamma)
    #TODO: experimental square instead of abs
    if self.use_squared_deviation:
      self.reward_deviation += (delta_reward ** 2 - self.reward_deviation) * (1 - self.gamma)
    else:
      self.reward_deviation += (np.abs(delta_reward) - self.reward_deviation) * (1 - self.gamma)
    self.last_advantage = advantage

    #step network in direction of trace gradient * lr * reward
    apply_regularization(self.model, self.regularization)
    step_weights_opt(self.model, self.traces, advantage, self.optimizer)
