import numpy as np
import tensorflow as tf

def create_traces(model):
  return [np.zeros(w.shape, dtype="float32") for w in model.trainable_variables]

#TODO: use truncated normal
#TODO: support more than 1/2d weights/bias
def xavier_exploration(theta, tau):
  shapes = [t.shape for t in theta]
  fan_ins = [s[0] if len(s) == 2 else np.prod(s) for s in shapes]
  fan_outs = [s[1] if len(s) == 2 else np.prod(s) for s in shapes]
  xavier_stdevs = [np.sqrt(2 / (fin + fout)) for fin, fout in zip(fan_ins, fan_outs)]
  d_theta = [np.random.normal(0, std * tau, size=shape).astype("float32") for std, shape in zip(xavier_stdevs, shapes)]

  return d_theta

#TODO: support recurrent
def get_action_and_gradient(model, state, tau, use_xavier=True):
  #get weights from model
  theta = model.get_weights()
  if use_xavier:
    d_theta = xavier_exploration(theta, tau)
  else:
    d_theta = [np.random.normal(0, tau, size=w.shape) for w in theta]

  #apply d_theta
  theta_temp = [w + d for w, d in zip(theta, d_theta)]
  model.set_weights(theta_temp)
  #predict action
  #action = model.predict(np.expand_dims(state, 0))[0]
  #https://github.com/keras-team/keras/issues/13118
  #https://github.com/tensorflow/tensorflow/issues/33009
  action = model.predict_on_batch(np.expand_dims(state, 0))[0]
  #restore weights
  model.set_weights(theta)

  #return action and "gradient"
  return action, d_theta

def step_weights(model, traces, lr, reward):
  #step direction
  alpha = lr * reward
  for weight, trace in zip(model.trainable_variables, traces):
    weight.assign_add(trace * alpha) #"gradient" ascent

def step_weights_opt(model, traces, reward, optimizer):
  traces2 = [(-t * advantage).astype("float32") for t in traces] #modulate by reward
  #then apply normally
  optimizer.apply_gradients(zip(traces, model.trainable_variables))
