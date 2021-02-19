from input_pipeline.data_extraction import DataSet
from input_pipeline.oracle import generate_sequence_of_actions
import tensorflow as tf

class TfDataSet():

  def __init__(self, dataset: DataSet):
    self.tokens_vocab = dataset.tokens_vocab
    self.relations_vocab = dataset.relations_vocab
    self.tf_data = TfDataSet.create_tf_dataset(dataset)

  @staticmethod
  def create_tf_dataset(dataset: DataSet):
    """
    Takes a DataSet object and creates a tf dataset with that data.

    Args:
        dataset: DataSet object
    """
    all_tokens = []
    all_action_types = []
    for dataset_entry in dataset.dataset_entries:
      act_seq = generate_sequence_of_actions(
        dataset_entry.tokens, dataset_entry.heads, dataset_entry.labels,
        partial_result = True)
      action_types, _ = act_seq
      # Go from ArcStandardAction to int (to use in tensor).
      action_types = [a.value for a in action_types]
      all_tokens.append(dataset_entry.tokens)
      all_action_types.append(action_types)
    data = {
      'tokens': tf.ragged.constant(all_tokens),
      'action_types': tf.ragged.constant(all_action_types)
    }
    tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    tf_dataset = tf_dataset.map(lambda x: x) # convert ragged -> uniform
    return tf_dataset
  
  def get_batches(self, batch_size: int, drop_remainder: bool = True):
    padded_shapes = {
      'tokens': [None],
      'action_types': [None]
    }
    return tf_data.padded_batch(
      batch_size, padded_shapes=padded_shapes, drop_remainder=drop_remainder)


if __name__=="__main__":
  # path = 'data/ud-treebanks-v2.7/UD_English-Pronouns/en_pronouns-ud-test.conllu'
  path = 'data/ud-treebanks-v2.7/UD_English-ParTUT/en_partut-ud-train.conllu'
  from input_pipeline.data_extraction import load_from_file
  train_data = load_from_file(path)
  tf_dataset = TfDataSet(train_data)
  tf_data = tf_dataset.tf_data
  i = 0
  for entry in tf_data.as_numpy_iterator():
    if i==10:
      break
    i+=1
    print(entry)
    print()
  batches = tf_dataset.get_batches(batch_size=5)
  first_batch = next(batches.as_numpy_iterator())
  print('First batch')
  print(first_batch)
