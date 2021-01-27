from absl.testing import absltest
from input_pipeline.oracle import generate_sequence_of_actions
from input_pipeline.data_extraction import load_from_file
"""
Run with 'python -m tests.input_pipeline.data_extraction_test'
from the project root directory.
"""

class DataExtractionTest(absltest.TestCase):

  def test_load_from_file(self):
    file_path = "tests/test_data/example.conllu"
    mock_data = load_from_file(file_path)
    self.assertEqual(len(mock_data.dataset_entries), 1)
    for dataset_entry in mock_data.dataset_entries:
      act_seq = generate_sequence_of_actions(dataset_entry.tokens,
                                             dataset_entry.heads,
                                             dataset_entry.labels)

if __name__ == '__main__':
  absltest.main()