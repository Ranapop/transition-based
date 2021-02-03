from os import path
from typing import List, Dict
import pickle as js

POSITION = 0
TOKEN = 2
HEAD = 6
RELATION = 7
NO_FIELDS = 10

PAD = '<PAD>'
PAD_ID = 0
ROOT = '<ROOT>'
ROOT_ID = 1
class DataSet(object):

  def __init__(self, lines: List[str]):
    self.dataset_entries: List[DataSetEntry] = []
    self.tokens_vocab: Dict[str, int] = {PAD: PAD_ID, ROOT: ROOT_ID}
    self.relations_vocab: Dict[str, int] = {}

    entry_lines = []

    # process lines belonging to a single entry until empty line is found
    for line in lines:
      line = line.strip()
      if line:
        entry_lines.append(line)
      elif entry_lines:
        self.dataset_entries.append(DataSetEntry(entry_lines, self))
        entry_lines.clear()
    if entry_lines:
      self.dataset_entries.append(DataSetEntry(entry_lines, self))


class DataSetEntry(object):

  def __init__(self, lines: List[str], dataset: DataSet):
    self.tokens: List[int] = [ROOT_ID]
    self.heads: List[int] = [-1]
    self.labels: List[int] = [None]

    for line in lines:
      line_fields = line.split('\t')
      if len(line_fields) == NO_FIELDS:
        position_in_sentence = line_fields[POSITION]
        # Skip words tokenized in multiple tokens (eg don't -> do not)
        # that don't have a single int position associated and have missing data. 
        if position_in_sentence.isnumeric():
          token, head, relation = str(line_fields[TOKEN]),\
                                  int(line_fields[HEAD]),\
                                  str(line_fields[RELATION])

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

      if cache and not path.exists(cached_file_path):
        print("Caching to file: " + cached_file_path)
        with open(cached_file_path, "wb") as cached_file:
          js.dump(data, cached_file)

  return data
