from absl.testing import absltest
from input_pipeline.oracle import generate_sequence_of_actions
from input_pipeline.data_extraction import load_from_file
"""
Run with 'python -m tests.input_pipeline.data_extraction_test'
from the project root directory.
"""

class DataExtractionTest(absltest.TestCase):

  def test_load_from_file(self):
    file_path = "transition-based/tests/test_data/example.conllu"
    mock_data = load_from_file(file_path)
    self.assertEqual(len(mock_data), 1)
    for dataset_entry in mock_data.dataset_entries:
      print(dataset_entry.tokens)
      act_seq = generate_sequence_of_actions(dataset_entry.tokens, dataset_entry.heads, dataset_entry.labels)
      if act_seq:
        act_types, act_labels = act_seq
      else:
        print("No sequence possible.")