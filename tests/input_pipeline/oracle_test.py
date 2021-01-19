from absl.testing import absltest
from input_pipeline.oracle import extract_labelled_edges,\
  extract_dependents, generate_sequence_of_actions,\
  ArcStandardAction

"""
Run with 'python -m tests.input_pipeline.oracle_test'
from the project root directory.
"""

class OracleTest(absltest.TestCase):


  def test_extract_labelled_edges(self):
    heads = [-1, 2, 0, 1]
    labels = [None, 10, 20, 30]
    labelled_edges = extract_labelled_edges(heads, labels)
    expected_labelled_edges = {(2,1): 10, (0,2):20, (1,3):30}
    self.assertEqual(labelled_edges, expected_labelled_edges)
  
  def test_extract_dependents(self):
    heads = [-1, 2, 0, 1, 2]
    dependents = extract_dependents(heads)
    expected_dependents = [[2], [3], [1, 4], [], []]
    self.assertEqual(dependents, expected_dependents)

  def est_generate_sequence_of_actions_non_projective(self):
    """
    Tree
    ROOT
      100
        400
          200
        300
          500
    """
    ROOT = 1
    sentence = [ROOT, 100, 200, 300, 400, 500]
    heads = [-1, 0, 4, 1, 1, 3]
    labels = [None, 10, 20, 30, 40, 50]
    act_seq = generate_sequence_of_actions(sentence, heads, labels)
    expected_act_seq = None
    self.assertEqual(act_seq, expected_act_seq)

  def test_generate_sequence_of_actions_projective_right(self):
    """
    Tree
    ROOT
      100
        200
          300
        400
          500
    """
    ROOT = 1
    sentence = [ROOT, 100, 200, 300, 400, 500]
    heads = [-1, 0, 1, 2, 1, 4]
    labels = [None, 10, 20, 30, 40, 50]
    act_seq = generate_sequence_of_actions(sentence, heads, labels)
    act_types, act_labels = act_seq
    expected_act_types = [
      ArcStandardAction.SHIFT,
      ArcStandardAction.SHIFT,
      ArcStandardAction.SHIFT,
      ArcStandardAction.SHIFT,
      ArcStandardAction.RIGHT_ARC,
      ArcStandardAction.RIGHT_ARC,
      ArcStandardAction.SHIFT,
      ArcStandardAction.SHIFT,
      ArcStandardAction.RIGHT_ARC,
      ArcStandardAction.RIGHT_ARC,
      ArcStandardAction.RIGHT_ARC
    ]
    expected_act_labels = [ROOT, 100, 200, 300, 30, 20, 400, 500, 50, 40, 10]
    self.assertEqual(act_types, expected_act_types)
    self.assertEqual(act_labels, expected_act_labels)

  def test_generate_sequence_of_actions_projective(self):
    """
    Tree
    ROOT
      100
        200
        500
          300
          400
    """
    ROOT = 1
    sentence = [ROOT, 100, 200, 300, 400, 500]
    heads = [-1, 0, 1, 5, 5, 1]
    labels = [None, 10, 20, 30, 40, 50]
    act_seq = generate_sequence_of_actions(sentence, heads, labels)
    act_types, act_labels = act_seq
    expected_act_types = [
      ArcStandardAction.SHIFT,
      ArcStandardAction.SHIFT,
      ArcStandardAction.SHIFT,
      ArcStandardAction.RIGHT_ARC,
      ArcStandardAction.SHIFT,
      ArcStandardAction.SHIFT,
      ArcStandardAction.SHIFT,
      ArcStandardAction.LEFT_ARC,
      ArcStandardAction.LEFT_ARC,
      ArcStandardAction.RIGHT_ARC,
      ArcStandardAction.RIGHT_ARC
    ]
    expected_act_labels = [ROOT, 100, 200, 20, 300, 400, 500, 40, 30, 50, 10]
    self.assertEqual(act_types, expected_act_types)
    self.assertEqual(act_labels, expected_act_labels)
    

if __name__ == '__main__':
  absltest.main()