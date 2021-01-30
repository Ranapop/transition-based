from typing import List
import dynet as dy
from input_pipeline.data_extraction import load_from_file, DataSet
from input_pipeline.oracle import generate_sequence_of_actions, apply_action

NO_EPOCHS = 50
EMBEDDINGS_SIZE = 20
LSTM_NUM_OF_LAYERS = 1
LSTM_STATE_SIZE = 25
BILSTM_STATE_SIZE = 2 * LSTM_STATE_SIZE
NO_STACK_FEATURES = 3
NO_BUFFER_FEATURES = 1
# The number of actions (left-arc, right-arc, shift).
NO_OUTPUT_CLASSES = 3

UD_ENGLISH_PATH = 'data/ud-treebanks-v2.7/UD_English-'
ENGLISH_PARTUT = 'ParTUT/en_partut-ud-{0}.conllu'

class ArcStandardModel():

  def __init__(self, vocab_size):
    self.model = dy.Model()
    self.sentence_encoder = SentenceEncoder(self.model, vocab_size)
    self.action_classifier = ActionClassiffier(self.model)
    self.trainer = dy.SimpleSGDTrainer(self.model)
  
  def create_cg(self, sentence: List[int], action_types: List[int], train: bool):
    # Build new computational graph.
    dy.renew_cg()
    bilstm_repr = self.sentence_encoder.add_to_cg(sentence)
    n = len(sentence)
    stack = []
    buffer = list(range(n))
    losses = []
    predicted_actions = []
    arc_history = []
    correct_predictions = 0
    for gold_action_type in action_types:
      features = extract_features(stack, buffer, bilstm_repr)
      logits = self.action_classifier.add_to_cg(features)
      probs = dy.softmax(logits)
      loss = -dy.log(dy.pick(probs, gold_action_type.value))
      losses.append(loss)
      predicted_actions.append(probs.npvalue().argmax())
      #TODO: Arc history is not needed at this point
      apply_action(gold_action_type, stack, buffer, arc_history)
    loss = dy.esum(losses)
    loss_value = loss.value()
    no_actions = len(action_types)
    for i in range(no_actions):
      if action_types[i].value == predicted_actions[i]:
        correct_predictions += 1
    accuracy = correct_predictions/no_actions
    if train:
      loss.backward()
      self.trainer.update()
    return predicted_actions, loss_value, accuracy
class SentenceEncoder():
  
  def __init__(self, model: dy.Model, vocab_size: int):
    self.input_lookup = model.add_lookup_parameters((vocab_size, EMBEDDINGS_SIZE))
    self.enc_fwd_lstm = dy.LSTMBuilder(
      LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, LSTM_STATE_SIZE, model)
    self.enc_bwd_lstm = dy.LSTMBuilder(
      LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, LSTM_STATE_SIZE, model)
    
  def add_to_cg(self, tokens):
    """
    Returns:
      a list of BiLSTM representation (a representation for each token in the
      sentence). Each element is a vector of size BILSTM_STATE_SIZE
    """
    embeddings = [self.input_lookup[token] for token in tokens]
    fwd_init_state = self.enc_fwd_lstm.initial_state()
    bwd_init_state = self.enc_bwd_lstm.initial_state()
    fw_exps = fwd_init_state.transduce(embeddings)
    bw_exps = bwd_init_state.transduce(reversed(embeddings))
    bilstm_repr = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]
    return bilstm_repr

class ActionClassiffier():

  def __init__(self, model):
    no_features = NO_STACK_FEATURES + NO_BUFFER_FEATURES
    classifier_input_size = no_features * BILSTM_STATE_SIZE
    self.weights_classiffier = model.add_parameters(
        (NO_OUTPUT_CLASSES, classifier_input_size))
    self.bias_classifier = model.add_parameters(NO_OUTPUT_CLASSES)
  
  def add_to_cg(self, features):
    w = dy.parameter(self.weights_classiffier)
    b = dy.parameter(self.bias_classifier)
    return w * features + b

#TODO: typing bilstm repr
def extract_features(stack: List[int], buffer: List[int],
                     bilstm_repr):
  """
  Return
    For 3 stack tokens & 1 buffer token:
    concatenated [stack_n-2, stack_n-1, stack_n, buffer_0]
    In case the stack has only 2 elements (or less), put zeros:
    concatenated [zero-vector, stack_n-1, stack_n, buffer_0]
  """
  # Get the positions of the tokens in the sentence.
  stack_token_positions = stack[-NO_STACK_FEATURES:]
  buffer_tokens_positions = buffer[:NO_BUFFER_FEATURES]
  # Get the bilstm representations.
  stack_bilstms = [bilstm_repr[i] for i in stack_token_positions]
  buffer_bilstms = [bilstm_repr[i] for i in buffer_tokens_positions]
  # Add zero-valued vectors if not enough features.
  no_missing_stack_features = NO_STACK_FEATURES - len(stack_bilstms)
  stack_bilstms = [dy.zeros(BILSTM_STATE_SIZE)] * no_missing_stack_features + stack_bilstms
  no_missing_buffer_features = NO_BUFFER_FEATURES - len(buffer)
  buffer_bilstms += [dy.zeros(BILSTM_STATE_SIZE)] * no_missing_buffer_features
  # Put stack & buffer features together in a list.
  features_list = stack_bilstms + buffer_bilstms
  # Concatenate the feature vectors.
  features = dy.concatenate(features_list)
  return features

def train_one_epoch(arc_standard_model: ArcStandardModel, dataset: DataSet):
  sum_loss = 0
  sum_accuracy = 0
  valid_entries = 0
  i = 0
  for data_entry in dataset.dataset_entries:
    # if i>2:
    #   break
    # i+=1
    act_seq = generate_sequence_of_actions(data_entry.tokens,
                                           data_entry.heads,
                                           data_entry.labels)
    # Make sure the action sequence can be generated.
    if act_seq:
      act_types = act_seq[0]
      predicted, loss_value, accuracy = arc_standard_model.create_cg(data_entry.tokens,
                                                       act_types,
                                                       train=True)
      sum_loss += loss_value
      sum_accuracy += accuracy
      valid_entries += 1
  avg_loss = sum_loss / valid_entries
  avg_accuracy = sum_accuracy/valid_entries
  return avg_loss, avg_accuracy

def train_model(train_data: DataSet):
  vocab_size = len(train_data.tokens_vocab.keys())
  arc_standard_model = ArcStandardModel(vocab_size)
  for epoch in range(NO_EPOCHS):
    avg_loss, avg_accuracy = train_one_epoch(arc_standard_model, train_data)
    print('Epoch {0} Loss {1} Accuracy {2}'.format(epoch, avg_loss, avg_accuracy))
  return arc_standard_model

def evaluate_model(eval_data: DataSet, trained_model: ArcStandardModel):
  sum_loss = 0
  sum_accuracy = 0
  valid_entries = 0
  for data_entry in dataset.dataset_entries:
    act_seq = generate_sequence_of_actions(data_entry.tokens,
                                           data_entry.heads,
                                           data_entry.labels)
    # Should we only test on valid entries??
    if act_seq:
      act_types = act_seq[0]
      predicted, loss_value, accuracy = arc_standard_model.create_cg(data_entry.tokens,
                                                       act_types,
                                                       train=False)
      sum_loss += loss_value
      sum_accuracy += accuracy
      valid_entries += 1
  avg_loss = sum_loss / valid_entries
  avg_accuracy = sum_accuracy/valid_entries
  return avg_loss, avg_accuracy


if __name__ == '__main__':
  train_path = UD_ENGLISH_PATH + ENGLISH_PARTUT.format('train')
  train_data = load_from_file(train_path)
  arc_standard_model = train_model(train_data)