"""
Tests for the single-step completion bug.

The bug: if the LM produces a response with no \\n\\n separators,
parse_response_to_actions returns 1 Action with is_final=True.
Previously, get_truncated_copy always set is_complete=False and
Node.__init__ always set is_complete=False, so these complete nodes
were expanded again — wasting the remaining budget generating text
after an already-finished answer.

The fix: get_truncated_copy sets is_complete based on action.is_final,
and Node.__init__ inherits is_complete from the state.
"""

import sys
sys.path.insert(0, ".")

from core.data_structures import Action, State, Node


# ── helpers ───────────────────────────────────────────────────────────────────

def make_state(prompt="Q:", steps=None):
    s = State(prompt=prompt)
    for step in (steps or []):
        s.append_step(step)
    return s


# ── 1. State.get_truncated_copy ───────────────────────────────────────────────

def test_truncated_copy_marks_complete_when_final_step_included():
    """Truncating TO a final step must produce is_complete=True."""
    final_action = Action(step_text="The answer is 42.", is_final=True)
    s = make_state(steps=[final_action])
    copy = s.get_truncated_copy(1)
    assert copy.is_complete, "Truncated copy that includes a final step must be marked complete"

def test_truncated_copy_not_complete_when_final_step_excluded():
    """Truncating BEFORE a final step must produce is_complete=False."""
    step1 = Action(step_text="Step 1: do maths.\n\n", is_final=False)
    step2 = Action(step_text="The answer is 42.", is_final=True)
    s = make_state(steps=[step1, step2])
    copy = s.get_truncated_copy(1)  # only step1
    assert not copy.is_complete, "Truncated copy that excludes final step must NOT be complete"

def test_truncated_copy_empty():
    """Empty truncation must be not complete."""
    s = make_state(steps=[Action(step_text="x", is_final=True)])
    copy = s.get_truncated_copy(0)
    assert not copy.is_complete


# ── 2. Node inherits is_complete from State ───────────────────────────────────

def test_node_inherits_complete_from_state():
    """Node created from a complete state must be marked complete."""
    final_action = Action(step_text="Answer: 7.", is_final=True)
    s = make_state(steps=[final_action])
    assert s.is_complete
    node = Node(state=s)
    assert node.is_complete, "Node must inherit is_complete=True from its state"

def test_node_inherits_not_complete_from_state():
    """Node created from an incomplete state must not be marked complete."""
    s = make_state(steps=[Action(step_text="Step 1.\n\n", is_final=False)])
    node = Node(state=s)
    assert not node.is_complete


# ── 3. Single-step response: node not returned as leaf to expand ──────────────

def test_single_step_node_excluded_from_leaves():
    """
    A child node whose entire state is a single-step final response
    must NOT appear in get_all_leaves() (and thus not be expanded).
    """
    root = Node(state=make_state())
    # Simulate a single-step complete response from root
    complete_state = make_state(steps=[Action(step_text="Full answer here.", is_final=True)])
    truncated = complete_state.get_truncated_copy(1)  # step 1 = the whole answer
    child = Node(state=truncated, parent=root)
    root.children.append(child)
    root.is_leaf = False

    leaves = root.get_all_leaves()
    assert child not in leaves, (
        "A child node that is already complete must NOT be returned as a leaf to expand"
    )
    assert leaves == [], f"Expected no expandable leaves, got: {leaves}"


# ── 4. Multi-step response: first-step branch IS a valid leaf ─────────────────

def test_multi_step_branch_node_is_leaf():
    """
    A child node whose state is truncated to step 1 (non-final)
    MUST appear in get_all_leaves() for expansion.
    """
    root = Node(state=make_state())
    step1 = Action(step_text="Step 1: reason.\n\n", is_final=False)
    step2 = Action(step_text="Final answer: 5.", is_final=True)
    full_state = make_state(steps=[step1, step2])
    truncated = full_state.get_truncated_copy(1)  # only step1
    child = Node(state=truncated, parent=root)
    root.children.append(child)
    root.is_leaf = False

    leaves = root.get_all_leaves()
    assert child in leaves, "A child with a non-final first step must be available for expansion"


# ── 5. response_to_children integration ──────────────────────────────────────

def test_response_to_children_complete_response_makes_complete_child():
    """
    When roll_out produces a single-step final response, response_to_children
    must create a child that is marked complete and excluded from leaves.
    """
    root = Node(state=make_state())
    # Simulate roll_out_to_leaf storing a 1-step complete state
    complete_resp = make_state(steps=[Action(step_text="Answer: 3.", is_final=True)])
    root.response_list = [complete_resp]
    root.response_to_children()

    assert len(root.children) == 1
    child = root.children[0]
    assert child.is_complete, "Child built from a single-step final response must be complete"
    assert child not in root.get_all_leaves(), "Complete child must not be in leaves"


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_truncated_copy_marks_complete_when_final_step_included,
        test_truncated_copy_not_complete_when_final_step_excluded,
        test_truncated_copy_empty,
        test_node_inherits_complete_from_state,
        test_node_inherits_not_complete_from_state,
        test_single_step_node_excluded_from_leaves,
        test_multi_step_branch_node_is_leaf,
        test_response_to_children_complete_response_makes_complete_child,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
