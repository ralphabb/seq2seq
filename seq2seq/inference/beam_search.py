# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-Graph Beam Search Implementation.
"""
#My changes: 1) This implementation accepts time_ as a float64 scalar value (which can be a placeholder).
#Float64 was used to allow for high-precision computation. All other state values (lengths, log_probs, finished) can also be replaced by placeholders (i.e. define a state with placeholder parameters and feed that into this code), as I have done in my use of this implementation. The reason behind this change to tensor values is that I discovered, while using this code, that repeated calls to the beam_search_step function caused redundant nodes to be added to the computational graph and, as such, as more steps were called, more nodes were introduced and computational complexity was quadratic in the number of steps at best. Hence, I took it upon myself to "tensorize" the parameters, so that the nodes need only be defined once. To use this code, you 
#a) Define your state using placeholders of dynamic shape for every state parameter (An example can be given upon request, as I have used this in my own project)
#b) Define time as a float64 tensor
#c) Compute next_state values using sess.run() and save these, along with time+1. Then, feed these as the new feed_dict to the same function. 
#NOTE: The function beam_search_step is set up outside any loop over time/steps. Within the loop calculating the beam programs, only feed_dicts are constructed for the next step using the outputs of sess.run (An example is available upon request)

#2) The previous implementation does not support a beam size larger than the vocabulary size, and would crash when this is the case. Using conditionals, namely tf.cond since I "tensorised" the parameters of this code, I added functionality to detect when the beam size is larger than the number of candidates using log operations, and to set the size of the relevant arrays accordingly

#Feel free to contact me for more explanations and any questions
#Best,
#Ralph Abboud
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611


class BeamSearchState:
    def __init__(self,log_probs,finished,lengths):
        self.log_probs = log_probs
        self.finished = finished
        self.lengths = lengths

class BeamSearchStepOutput:
    def __init__(self,scores,predicted_ids,beam_parent_ids):
        self.predicted_ids = predicted_ids
        self.scores = scores
        self.beam_parent_ids = beam_parent_ids

class BeamSearchConfig:
    def __init__(self,beam_width,vocab_size,eos_token,length_penalty_weight, choose_successors_fn):
        self.beam_width = beam_width
        self.vocab_size = vocab_size
        self.eos_token = eos_token
        self.length_penalty_weight = length_penalty_weight
        self.choose_successors_fn = choose_successors_fn


def gather_tree_py(values, parents):
  """Gathers path through a tree backwards from the leave nodes. Used
  to reconstruct beams given their parents."""
  beam_length = values.shape[0]
  num_beams = values.shape[1]
  res = np.zeros_like(values)
  res[-1, :] = values[-1, :]
  for beam_id in range(num_beams):
    parent = parents[-1][beam_id]
    for level in reversed(range(beam_length - 1)):
      res[level, beam_id] = values[level][parent]
      parent = parents[level][parent]
  return np.array(res).astype(values.dtype)


def gather_tree(values, parents):
  """Tensor version of gather_tree_py"""
  res = tf.py_func(
      func=gather_tree_py, inp=[values, parents], Tout=values.dtype)
  res.set_shape(values.get_shape().as_list())
  return res

def create_initial_beam_state_numpy(config):
  """Creates an instance of `BeamState` that can be used on the first
  call to `beam_step`.
  Args:
    config: A BeamSearchConfig
  Returns:
    An instance of `BeamState`.
  """
  #Ralph Edit: This depends on vocabulary vs beam, what if beam > vocab
  width = np.minimum(config.beam_width, config.vocab_size)
  return BeamSearchState(
      log_probs=np.zeros(width),
      finished=np.zeros(
          width, dtype=np.bool),
      lengths=np.zeros(
          width, dtype=np.int32))


def create_initial_beam_state(config):
  """Creates an instance of `BeamState` that can be used on the first
  call to `beam_step`.
  Args:
    config: A BeamSearchConfig
  Returns:
    An instance of `BeamState`.
  """
  #Ralph Edit: This depends on vocabulary vs beam, what if beam > vocab
  width = np.minimum(config.beam_width, config.vocab_size)
  return BeamSearchState(
      log_probs=tf.zeros([width]),
      finished=tf.zeros(
          [width], dtype=tf.bool),
      lengths=tf.zeros(
          [width], dtype=tf.int32))


def length_penalty(sequence_lengths, penalty_factor):
  """Calculates the length penalty according to
  https://arxiv.org/abs/1609.08144
   Args:
    sequence_lengths: The sequence length of all hypotheses, a tensor
      of shape [beam_size, vocab_size].
    penalty_factor: A scalar that weights the length penalty.
  Returns:
    The length penalty factor, a tensor fo shape [beam_size].
   """
  return tf.div((5. + tf.to_float(sequence_lengths))**penalty_factor, (5. + 1.)
                **penalty_factor)


def hyp_score(log_probs, sequence_lengths, config):
  """Calculates scores for beam search hypotheses.
  """

  # Calculate the length penality
  length_penality_ = length_penalty(
      sequence_lengths=sequence_lengths,
      penalty_factor=config.length_penalty_weight)

  score = log_probs / length_penality_
  return score


def choose_top_k(scores_flat,time_,config): #Top k now accepts time as well to check for the k to use.
  """Chooses the top-k beams as successors.
  """
  #Ralph Edit to support initial beam > vocab_size
  VocabPowerTimePlusOne = tf.cast(tf.exp(tf.multiply(tf.cast(tf.log(config.vocab_size+0.0),dtype=tf.float64),time_+1)),dtype = tf.int32) #Compute vocabSize**(time+1)
  threshold = tf.convert_to_tensor((np.log(config.beam_width)/np.log(config.vocab_size)), dtype = tf.float64) #log of beam_width base vocab size. Float 64 is used for high-precision computation. This 
  #expression computes the time before the space of programs is larger than the beam size
  predicate2 = tf.greater(threshold,time_+1) 
  shape = tf.cond(predicate2, lambda: VocabPowerTimePlusOne, lambda: tf.convert_to_tensor(config.beam_width)) #If space of programs is still smaller than beam, only select the available space.
  next_beam_scores, word_indices = tf.nn.top_k(scores_flat, k=shape)
  return next_beam_scores, word_indices

def nest_map(inputs, map_fn, name=None):
  """Applies a function to (possibly nested) tuple of tensors.
  """
  if nest.is_sequence(inputs):
    inputs_flat = nest.flatten(inputs)
    y_flat = [map_fn(_) for _ in inputs_flat]
    outputs = nest.pack_sequence_as(inputs, y_flat)
  else:
    outputs = map_fn(inputs)
  if name:
    outputs = tf.identity(outputs, name=name)
  return outputs


def mask_probs(probs, eos_token, finished):
  """Masks log probabilities such that finished beams
  allocate all probability mass to eos. Unfinished beams remain unchanged.
  Args:
    probs: Log probabiltiies of shape `[beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to
    finished: A boolean tensor of shape `[beam_width]` that specifies which
      elements in the beam are finished already.
  Returns:
    A tensor of shape `[beam_width, vocab_size]`, where unfinished beams
    stay unchanged and finished beams are replaced with a tensor that has all
    probability on the EOS token.
  """
  vocab_size = tf.shape(probs)[1]
  finished_mask = tf.expand_dims(tf.to_float(1. - tf.to_float(finished)), 1)
  # These examples are not finished and we leave them
  non_finished_examples = finished_mask * probs
  # All finished examples are replaced with a vector that has all
  # probability on EOS
  finished_row = tf.one_hot(
      eos_token,
      vocab_size,
      dtype=tf.float32,
      on_value=0.,
      off_value=tf.float32.min)
  finished_examples = (1. - finished_mask) * finished_row
  return finished_examples + non_finished_examples


def beam_search_step(time_, logits, beam_state, config):
  """Performs a single step of Beam Search Decoding.
  Args:
    time_: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape `[B, vocab_size]`
    beam_state: Current state of the beam search. An instance of `BeamState`
    config: An instance of `BeamSearchConfig`
  Returns:
    A new beam state.
  """

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished

  probs = tf.nn.log_softmax(logits)
  probs = mask_probs(probs, config.eos_token, previously_finished)
  total_probs = tf.expand_dims(beam_state.log_probs, 1) + probs
  threshold = tf.convert_to_tensor((np.log(config.beam_width)/np.log(config.vocab_size)), dtype = tf.float64)
  VocabPowerTime = tf.cast(tf.exp(tf.multiply(tf.cast(tf.log(config.vocab_size+0.0),dtype=tf.float64),time_)),dtype = tf.int32)
  # Calculate the continuation lengths
  # We add 1 to all continuations that are not EOS and were not
  # finished previously
  #Ralph Edit to support initial beam > vocab_size
  predicate = tf.greater(threshold,time_)
  unit = tf.cond(predicate, lambda: VocabPowerTime, lambda: tf.convert_to_tensor(config.beam_width)) #Tensor conditional to find the shape dimension to use
  lengths_to_add = tf.one_hot([config.eos_token] * unit,
                              config.vocab_size, 0, 1)
  add_mask = (1 - tf.to_int32(previously_finished))
  lengths_to_add = tf.expand_dims(add_mask, 1) * lengths_to_add
  new_prediction_lengths = tf.expand_dims(prediction_lengths,
                                          1) + lengths_to_add

  # Calculate the scores for each beam
  scores = hyp_score(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      config=config)

  scores_flat = tf.reshape(scores, [-1])
  # During the first time step we only consider the initial beam
  scores_flat = tf.cond(
      tf.convert_to_tensor(time_) > 0, lambda: scores_flat, lambda: scores[0])
  # Pick the next beams according to the specified successors function
  next_beam_scores, word_indices = config.choose_successors_fn(scores_flat,time_,
                                                               config)
  #Ralph Edit to support initial beam > vocab_size
  predicate2 = tf.greater(threshold,time_+1)
  VocabPowerTimePlusOne = tf.cast(tf.exp(tf.multiply(tf.cast(tf.log(config.vocab_size+0.0),dtype=tf.float64),time_+1)),dtype = tf.int32)
  shape = tf.cond(predicate2, lambda: VocabPowerTimePlusOne, lambda: tf.convert_to_tensor(config.beam_width)) #Replaces the old non-tensor implementation
  newShape = tf.expand_dims(shape,axis=-1)
  tf.reshape(next_beam_scores,newShape)
  tf.reshape(word_indices,newShape)
  # Pick out the probs, beam_ids, and states according to the chosen predictions
  total_probs_flat = tf.reshape(total_probs, [-1], name="total_probs_flat")
  next_beam_probs = tf.gather(total_probs_flat, word_indices)
  tf.reshape(next_beam_probs,newShape)
  next_word_ids = tf.mod(word_indices, config.vocab_size)
  next_beam_ids = tf.div(word_indices, config.vocab_size)

  # Append new ids to current predictions
  next_finished = tf.logical_or(
      tf.gather(beam_state.finished, next_beam_ids),
      tf.equal(next_word_ids, config.eos_token))

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged
  # 2. Beams that are now finished (EOS predicted) remain unchanged
  # 3. Beams that are not yet finished have their length increased by 1
  lengths_to_add = tf.to_int32(tf.not_equal(next_word_ids, config.eos_token))
  lengths_to_add = (1 - tf.to_int32(next_finished)) * lengths_to_add
  next_prediction_len = tf.gather(beam_state.lengths, next_beam_ids)
  next_prediction_len += lengths_to_add

  next_state = BeamSearchState(
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished)

  output = BeamSearchStepOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      beam_parent_ids=next_beam_ids)

  return output, next_state
