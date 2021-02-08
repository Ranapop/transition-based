import functools

from absl.testing import absltest
import jax
from jax import numpy as jnp
from flax_training.models import SentenceEncoderScan, SentenceEncoder, BILSTM_STATE_SIZE

class ModelsTest(absltest.TestCase):

  def test_sentence_encoder_scan(self):

    vocab_size = 10
    length = jnp.array(3)
    lstm_state_size = 8
    sequence_length = 5

    rng = jax.random.PRNGKey(0)
    x = jnp.array([1, 2, 3, 4, 5])
    c = jnp.zeros(lstm_state_size)
    h = jnp.zeros(lstm_state_size)
    step = 0
    carry = c, h, step

    sec = SentenceEncoderScan(vocab_size, length)
    (carry, y), _ = sec.init_with_output(rng, carry, x)

    self.assertEqual(y.shape, (sequence_length, lstm_state_size))
    self.assertEqual(carry[0].shape, (lstm_state_size,))
    self.assertEqual(carry[1].shape, (lstm_state_size,))
    self.assertEqual(carry[2].shape, ())
    self.assertEqual(carry[2], 5)

  def test_sentence_encoder(self):
    vocab_size = 16
    length = jnp.array(3)
    max_length = 5
    enc = SentenceEncoder(vocab_size)
    sentence = jnp.array([1, 2, 3, 4, 5])
    rng = jax.random.PRNGKey(0)
    enc_sentence, _ = enc.init_with_output(rng, sentence, length)

    self.assertEqual(len(enc_sentence), max_length)
    for s in enc_sentence:
      self.assertEqual(s.shape, (BILSTM_STATE_SIZE,))

  def test_sentence_encoder_with_vmap(self):
    vocab_size = 100
    batch = jnp.array([
      [1, 2, 3, 4, 5, 0, 0],
      [2, 7, 6, 0, 0, 0, 0]
    ])
    lengths = jnp.array([5, 3])
    rng = jax.random.PRNGKey(0)
    enc = SentenceEncoder(vocab_size)
    init_fn = functools.partial(enc.init_with_output, rng)
    vmapped_enc = jax.vmap(init_fn)
    vmapped_enc(batch, lengths)

if __name__ == '__main__':
  absltest.main()