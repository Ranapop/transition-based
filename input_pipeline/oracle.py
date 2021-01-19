from typing import List, Dict, Tuple
from enum import Enum

class ArcStandardAction(Enum):
  LEFT_ARC = 1
  RIGHT_ARC = 2
  SHIFT = 3


def check_dependents(token_id: int,
                     gold_dependents: List[List[int]],
                     arc_history: List[Tuple[int,int]]
                     ) -> bool:
  """Checks if the input token has all dependents processed. If they all are
  processed (in the arcs history), the method returns true."""
  for dependent in gold_dependents[token_id]:
    if (token_id, dependent) not in arc_history:
      return False
  return True

def shortest_stack_oracle(stack: List[int],
                          buffer: List[int],
                          arc_history: List[Tuple[int, int]],
                          gold_labelled_edges: Dict[Tuple[int, int], int],
                          gold_dependents: Dict[int, List[int]],
                          sentence: List[int]
                          ) -> Tuple[int, int]:
  """
  Shortest stack oracle. Try to apply a left-arc, then a right-arc, and finally
  apply a shift transition. TODO: link to paper/explanation.

  Args:
    stack: transition-system stack. This is where the tree is constructed. The
      stack is represented by a python list, and the elements are tokens
      (positions in the input sentence).
    buffer: initially the input sentence, it's where the non-processed tokens
      are stored. It's represented by a python list of tokens.
    arc_history: List of processed arcs.
    gold_labelled_edges: Output tree -represented as a set of labeled edges,
      more concretely through a dictionary of (token1, token2) -> arc_label_id.
    gold_dependents: Dictionary of dependents for each token, that is
      token id -> list of dependents.
    sentence: Input sentence.
  Returns:
    the gold action for the given configuration or None if no action can be
    generated.
  """
  # make sure there are at least two tokens on the stack.
  if len(stack) >=2:
    stack_top = stack[-1]
    stack_2nd_top = stack[-2]
    # Try Left-Arc, that is link the top of the stack to the second top of the
    # stack and pop the second to top of the stack.
    if (stack_top, stack_2nd_top) in gold_labelled_edges.keys():
      action_label = gold_labelled_edges[(stack_top, stack_2nd_top)]
      return (ArcStandardAction.LEFT_ARC, action_label)
    # Try Right-ARC (check if there is a link from the second to the top of the
    # stack and that the top has no unprocessed dependents).
    if (stack_2nd_top, stack_top) in gold_labelled_edges.keys()\
      and check_dependents(stack_top, gold_dependents, arc_history):
      action_label = gold_labelled_edges[(stack_2nd_top, stack_top)]
      return (ArcStandardAction.RIGHT_ARC, action_label)
  # Try Shift.
  if len(buffer)>0:
    shifted_token = sentence[buffer[0]]
    return (ArcStandardAction.SHIFT, shifted_token)
  # If no action was returned, return None.
  return None

def apply_action(action_type: int,
                 stack: List[int], buffer: List[int],
                 arc_history: List[Tuple[int, int]]):
  """Applies the given action, modifying the input lists (stack and buffer)."""
  if action_type == ArcStandardAction.LEFT_ARC:
    # Store the arc in arc history.
    arc_history.append((stack[-1], stack[-2]))
    # Remove  stack second to top.
    del stack[-2]
  if action_type == ArcStandardAction.RIGHT_ARC:
    # Store the arc in arc history.
    arc_history.append((stack[-2], stack[-1]))
    # Remove stack top.
    del stack[-1]
  if action_type == ArcStandardAction.SHIFT:
    # Shift element from buffer to stack.
    shifted = buffer[0]
    del buffer[0]
    stack.append(shifted)

def extract_labelled_edges(heads: List[int],
                           labels: List[int]) -> Dict[Tuple[int,int], int]:
  """
  Extracts from a list of heads and labels a dictionary of labellled edges.

  Args:
    heads: head for each token (parent vector).
    labels: labels for heads.

  Returns:
    A dictionary (token1,token2) -> label. The tokens are represented by their
    position in the sentence.
  """
  labelled_edges = {}
  n = len(heads)
  for i in range(1, n):
    edge = (heads[i], i)
    labelled_edges[edge] = labels[i]
  return labelled_edges

def extract_dependents(heads: List[int]) -> List[List[int]]:
  """Generates a children list representation of the tree
  from a parent vector representation.
  
  Returns:
    A list of children lists.
  """
  n = len(heads)
  dependents = [ [] for i in range(n)]
  for i in range(len(heads)):
    if heads[i]!=-1:
      dependents[heads[i]].append(i)
  return dependents
    
def generate_sequence_of_actions(sentence: List[int],
                                 heads: List[int],
                                 labels: List[int]
                                 ) -> Tuple[List[int], List[int]]:
  """
  Generate a sequence of actions. Use the static shortest-stack oracle for the
  arc standard transition-system.

  Args:
    sentence: Input sentence (list of token ids). The sentence starts with a
      ROOT token id on the first position.
    heads: Output tree represented by a the parent vector. The first position
      will always be -1 cause that's where the root is.
    labels: A list of arc labels. On the first position is None (corresponding
      to head -1).
  Returns:
    A sequence of actions. Each action has an action type an action label (arc
    label or shifted token). The sequence of actions is represented through two
    arrays: action_types and action_labels, a tuple of which is returned.
  """
  n = len(sentence)
  stack = []
  buffer = list(range(n))
  action_types = []
  action_labels = []
  arc_history = []
  
  gold_dependents = extract_dependents(heads)
  gold_labelled_edges = extract_labelled_edges(heads, labels)
  # The parsing ends when the buffer is empty and the stack only contains
  # the root node.
  while len(buffer)!=0 or len(stack)!=1:
    action = shortest_stack_oracle(stack, buffer, arc_history,
                                   gold_labelled_edges, gold_dependents,
                                   sentence)
    if action is None:
      return None
    # Apply action.
    apply_action( action[0], stack, buffer, arc_history)
    # Save action.
    action_types.append(action[0])
    action_labels.append(action[1])
  
  # Check that the stack contains the root.
  if stack[0]!=0:
    return None
  # Return the sequence of actions.
  return (action_types, action_labels)
