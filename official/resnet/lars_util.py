# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Enable Layer-wise Adaptive Rate Scaling optimizer in ResNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from official.utils.lars_optimizer import LARSOptimizer

FLAGS = flags.FLAGS

# From imagenet_main.py
_NUM_TRAIN_IMAGES = 1281167

def poly_rate_schedule(current_epoch,
                       batch_size,
                       poly_rate=0.0):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.  After
  FLAGS.poly_warmup_epochs, we reach the base learning rate (scaled to account
  for batch size). The learning rate is then decayed using a polynomial rate
  decay schedule with power 2.0.

  Args:
    current_epoch: `Tensor` for current epoch.
    poly_rate: Polynomial decay rate.

  Returns:
    A scaled `Tensor` for current learning rate.
  """

  if batch_size <= 16384:
    plr = 25.0
    w_epochs = 5
  elif batch_size == 32768:
    plr = 34.0
    w_epochs = 12
  elif batch_size == 65536:
    plr = 35.0
    w_epochs = 34
  else:
    plr = 41.0
    w_epochs = 16

  # Override default poly learning rate and warmup epochs
  if poly_rate > 0.0:
    plr = poly_rate

  train_steps = (_NUM_TRAIN_IMAGES // batch_size) * FLAGS.train_epochs
  tf.logging.info('Total train steps: %d' % train_steps)

  wrate = (plr * current_epoch / w_epochs)
  w_steps = (w_epochs * _NUM_TRAIN_IMAGES // batch_size)
  min_step = tf.constant(1, dtype=tf.int64)
  global_step = tf.train.get_or_create_global_step()
  decay_steps = tf.maximum(min_step, tf.subtract(global_step, w_steps))
  poly_rate = tf.train.polynomial_decay(
      plr,
      decay_steps,
      train_steps - w_steps + 1,
      power=2.0)
  decay_rate = tf.where(current_epoch <= w_epochs, wrate, poly_rate)
  return decay_rate


def init_lars_optimizer(current_epoch, batch_size, momentum, weight_decay):
  """Initialize the LARS Optimizer."""

  learning_rate = poly_rate_schedule(current_epoch, batch_size, FLAGS.poly_rate)
  optimizer = LARSOptimizer(
      learning_rate,
      momentum=momentum,
      weight_decay=weight_decay,
      skip_list=['batch_normalization', 'bias'])
  return optimizer
