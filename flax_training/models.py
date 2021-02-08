import functools

import jax
from flax import linen as nn
from jax import numpy as jnp


NO_EPOCHS = 20
EMBEDDINGS_SIZE = 50
LSTM_STATE_SIZE = 25
BILSTM_STATE_SIZE = 2 * LSTM_STATE_SIZE
NO_STACK_FEATURES = 3
NO_BUFFER_FEATURES = 1
# The number of actions (left-arc, right-arc, shift).
NO_OUTPUT_CLASSES = 3

class SentenceEncoderScan(nn.Module):
  """
  Attributes:
    vocab_size: vocabulary size.
    length: sentence length.
  """
  vocab_size: int
  length: jnp.array

  @functools.partial(
    nn.transforms.scan,
    variable_broadcast='params',
    split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    embed = nn.Embed(self.vocab_size, EMBEDDINGS_SIZE)
    lstm_cell = nn.LSTMCell()
    c, h, step = carry
    embed_x = embed(x)
    (new_c, new_h), state = lstm_cell((c, h), embed_x)
    c = jnp.where(step < self.length, new_c, c)
    h = jnp.where(step < self.length, new_h, h)
    new_carry = c, h, step+1
    return new_carry, state


class SentenceEncoder(nn.Module):
  vocab_size: int

  @nn.compact
  def __call__(self, sentence, sentence_length):
    rng = jax.random.PRNGKey(0)
    fwd_sec = SentenceEncoderScan(self.vocab_size, sentence_length)
    bwd_sec = SentenceEncoderScan(self.vocab_size, sentence_length)
    fwd_c0, fwd_h0 = nn.LSTMCell.initialize_carry(rng, (), LSTM_STATE_SIZE)
    bwd_c0, bwd_h0 = nn.LSTMCell.initialize_carry(rng, (), LSTM_STATE_SIZE)
    fwd_carry = (fwd_c0, fwd_h0, 0)
    bwd_carry = (bwd_c0, bwd_h0, 0)
    _, fwd_states = fwd_sec(fwd_carry, sentence)
    reversed_sentence = jnp.flip(sentence)
    padding_length = sentence.shape[0] - sentence_length
    reversed_sentence = jnp.roll(reversed_sentence, -padding_length)
    _, bwd_states = bwd_sec(bwd_carry, reversed_sentence)
    return [jnp.concatenate(z) for z in zip(fwd_states, bwd_states)]
