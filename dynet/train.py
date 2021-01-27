
from input_pipeline.data_extraction import load_from_file

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