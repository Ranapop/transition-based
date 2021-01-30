from typing import List
import dynet as dy
import numpy as np
from input_pipeline.data_extraction import load_from_file, DataSet
from input_pipeline.oracle import generate_sequence_of_actions, apply_action,\
  ArcStandardAction, get_valid_actions_mask

NO_EPOCHS = 20
EMBEDDINGS_SIZE = 50
LSTM_NUM_OF_LAYERS = 1
LSTM_STATE_SIZE = 25
BILSTM_STATE_SIZE = 2 * LSTM_STATE_SIZE
NO_STACK_FEATURES = 3
NO_BUFFER_FEATURES = 1
# The number of actions (left-arc, right-arc, shift).
NO_OUTPUT_CLASSES = 3

UD_ENGLISH_PATH = 'data/ud-treebanks-v2.7/UD_English-'
ENGLISH_PARTUT = 'ParTUT/en_partut-ud-{0}.conllu'
ENGLISH_EWT = 'EWT/en_ewt-ud-{0}.conllu'

class ArcStandardModel():

  def __init__(self, vocab_size):
    self.model = dy.Model()
    self.sentence_encoder = SentenceEncoder(self.model, vocab_size)
    self.action_classifier = ActionClassiffier(self.model)
    self.trainer = dy.SimpleSGDTrainer(self.model)
  
  def create_cg(self, sentence: List[int], action_types: List[int], train: bool):
    # Build new computational graph.
    dy.renew_cg()
    bilstm_repr = self.sentence_encoder.add_to_cg(sentence, train)
    n = len(sentence)
    stack = []
    buffer = list(range(n))
    losses = []
    predicted_actions = []
    arc_history = []
    for gold_action_type in action_types:
      features = extract_features(stack, buffer, bilstm_repr)
      logits = self.action_classifier.add_to_cg(features)
      probs = dy.softmax(logits)
      loss = -dy.log(dy.pick(probs, gold_action_type.value))
      losses.append(loss)
      if train:
        probabilities = probs.npvalue()
      else:
        actions_mask = get_valid_actions_mask(stack, buffer)
        probabilities = probs.npvalue() * np.array(actions_mask)
      predicted_action = probabilities.argmax()
      predicted_actions.append(predicted_action)
      #TODO: Arc history is not needed at this point
      if train:
        apply_action(gold_action_type, stack, buffer, arc_history)
      else:
        # TODO: count incomplete parses / do head accuracy.
        predicted_action_type = ArcStandardAction(predicted_action)
        apply_action(predicted_action_type, stack, buffer, arc_history)
    loss = dy.esum(losses)
    loss_value = loss.value()
    gold_actions = [action_type.value for action_type in action_types]
    accuracy = ArcStandardModel.compute_accuracy(gold_actions,
                                                 predicted_actions)
    if train:
      loss.backward()
      self.trainer.update()
    return predicted_actions, loss_value, accuracy

  @staticmethod
  def compute_accuracy(gold_actions: List[int], predicted_actions: List[int]):
    no_actions = len(gold_actions)
    correct_predictions = 0
    for i in range(no_actions):
      if gold_actions[i] == predicted_actions[i]:
        correct_predictions += 1
    accuracy = correct_predictions/no_actions
    return accuracy
class SentenceEncoder():
  
  def __init__(self, model: dy.Model, vocab_size: int):
    self.input_lookup = model.add_lookup_parameters((vocab_size, EMBEDDINGS_SIZE))
    self.enc_fwd_lstm = dy.LSTMBuilder(
      LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, LSTM_STATE_SIZE, model)
    self.enc_bwd_lstm = dy.LSTMBuilder(
      LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, LSTM_STATE_SIZE, model)
    
  def add_to_cg(self, tokens, train=False):
    """
    Returns:
      a list of BiLSTM representation (a representation for each token in the
      sentence). Each element is a vector of size BILSTM_STATE_SIZE
    """
    embeddings = [self.input_lookup[token] for token in tokens]
    if train:
      self.enc_fwd_lstm.set_dropouts(0.2, 0.2)
      self.enc_bwd_lstm.set_dropouts(0.2, 0.2)
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

def evaluate_model(trained_model: ArcStandardModel, eval_data: DataSet):
  sum_loss = 0
  sum_accuracy = 0
  valid_entries = 0
  for data_entry in eval_data.dataset_entries:
    act_seq = generate_sequence_of_actions(data_entry.tokens,
                                           data_entry.heads,
                                           data_entry.labels)
    # Should we only test on valid entries??
    if act_seq:
      act_types = act_seq[0]
      predicted, loss_value, accuracy = trained_model.create_cg(
        data_entry.tokens,
        act_types,
        train=False)
      sum_loss += loss_value
      sum_accuracy += accuracy
      valid_entries += 1
  avg_loss = sum_loss / valid_entries
  avg_accuracy = sum_accuracy/valid_entries
  return avg_loss, avg_accuracy

def train_model(train_data: DataSet, dev_data: DataSet):
  vocab_size = len(train_data.tokens_vocab.keys())
  arc_standard_model = ArcStandardModel(vocab_size)
  for epoch in range(NO_EPOCHS):
    train_loss, train_acc = train_one_epoch(arc_standard_model, train_data)
    dev_loss, dev_acc = evaluate_model(arc_standard_model, dev_data)
    print('Epoch {0} Train loss {1}, acc {2}; Dev loss {3}, acc {4}'.format(
      epoch, train_loss, train_acc, dev_loss, dev_acc))
  return arc_standard_model

def get_data_paths(dataset_id: str):
  if dataset_id == 'ParTUT':
    train_path = UD_ENGLISH_PATH + ENGLISH_PARTUT.format('train')
    dev_path = UD_ENGLISH_PATH + ENGLISH_PARTUT.format('dev')
  elif dataset_id == 'EWT':
    train_path = UD_ENGLISH_PATH + ENGLISH_EWT.format('train')
    dev_path = UD_ENGLISH_PATH + ENGLISH_EWT.format('dev')
  return train_path, dev_path
    

if __name__ == '__main__':
  train_path, dev_path = get_data_paths('EWT')
  train_data = load_from_file(train_path)
  dev_data = load_from_file(dev_path)
  arc_standard_model = train_model(train_data, dev_data)