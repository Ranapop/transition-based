from os import makedirs, path
from typing import Tuple, List, Dict
from oracle import generate_sequence_of_actions
import pickle as js

class DataSet(object):

  def __init__(self, lines: List[str]):
    self.sentences: List[Sentence] = []
    self.tokens_vocab: Dict[str, int] = {}
    self.relations_vocab: Dict[str, int] = {}

    entry_lines = []
    for line in lines:
      line = line.strip()
      if line:
        entry_lines.append(line)
      elif entry_lines:
        self.sentences.append(Sentence(entry_lines, self))
        entry_lines.clear()
    if entry_lines:
      self.sentences.append(Sentence(entry_lines, self))

class Sentence(object):

  def __init__(self, lines: List[str], dataset: DataSet):
    self.tokens: List[int] = [-1]
    self.heads: List[int] = [-1]
    self.labels: List[int] = [None]

    for line in lines:
      line_fields = line.split('\t')
      if len(line_fields) > 7:
        token, head, relation = str(line_fields[2]), int(line_fields[6]), str(line_fields[7])
        if token not in dataset.tokens_vocab.keys():
          dataset.tokens_vocab.update({token: len(dataset.tokens_vocab.keys())})
        if relation not in dataset.relations_vocab.keys():
          dataset.relations_vocab.update({relation: len(dataset.relations_vocab.keys())})
        self.tokens.append(dataset.tokens_vocab[token])
        self.heads.append(head)
        self.labels.append(dataset.relations_vocab[relation])

def load_from_file(file_path: str, cache=True) -> DataSet:
  """
  Loads data from file or cache.
  Args:
    file_path - the path to the data
    cache - boolean indicating caching
  """
  cached_file_path = file_path + ".cached"

  if cache and path.exists(cached_file_path):
    print("Using cached data from: " + cached_file_path)
    with open(cached_file_path, "rb") as cached_file:
      data = js.load(cached_file)
  else:
    print("Extracting data from: " + file_path)
    with open(file_path, encoding='utf-8') as file:
      lines = file.readlines()
      data = DataSet(lines)

      print(cached_file_path)
      print(path.exists(cached_file_path))
      if cache and not path.exists(cached_file_path):
        print("Caching to file: " + cached_file_path)
        with open(cached_file_path, "wb") as cached_file:
          js.dump(data, cached_file)

  return data


if __name__ == '__main__':
  file_path = "/home/paul/PycharmProjects/transition-based/mock_data/ro_rrt-ud-test.conllu"
  mock_data = load_from_file(file_path)
  for sentence in mock_data.sentences:
    print(sentence.tokens)
    act_seq = generate_sequence_of_actions(sentence.tokens, sentence.heads, sentence.labels)
    if act_seq:
      act_types, act_labels = act_seq
      print(act_types)
    else:
      print("No sequence possible.")