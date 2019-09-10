import numpy as np
import tensorflow as tf

def create_traces(model):
  return [np.zeros_like(w) for w in model.get_weights()]

def get_action_and_gradient(model, state, tau, scale_exploration=False):
  #get weights from model
  theta = model.get_weights()
  #modulate weights for exploration
  if scale_exploration == True:
    theta_pred = [w.copy() + np.random.normal(size=w.shape) * tau for w in theta]
  elif callable(scale_exploration):
    theta_pred = [w.copy() + np.random.normal(size=w.shape) * tau * scale_exploration(w) for w in theta]
  else:
    theta_pred = [np.random.normal(w, np.abs(w) * tau) for w in theta]
  model.set_weights(theta_pred)
  #predict action
  action = model.predict(np.expand_dims(state, 0))[0]
  #calculate "gradient"
  d_theta = [a - b for a, b in zip(theta_pred, theta)]

  #restore weights
  model.set_weights(theta)
  #return action and "gradient"
  return action, d_theta

#TODO: support using optimizers
def step_weights(model, traces, alpha, advantage):
  weights = model.get_weights()
  # update theta based on reward-modulated traces (training step)
  #TODO: should probably use assign_add
  weights = [w + t * advantage * alpha for w, t in zip(weights, traces)]

  model.set_weights(weights)
