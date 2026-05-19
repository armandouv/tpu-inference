import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.runner.tpu_runner import _select_next_beams

def test_beam_search_logic():
    """
    Test the beam search sampling logic with concrete arrays.
    """
    beam_width = 3
    vocab_size = 10
    
    # Step 0 logits!
    # Shape (beam_width, vocab_size) = (3, 10)
    logits_step0 = jnp.array([
        [1.0, 2.0, 0.5, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], # Beam 0 candidates
        [0.5, 1.0, 2.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], # Beam 1 candidates
        [0.1, 0.2, 0.3, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Beam 2 candidates
    ])
    
    cum_logprobs = jnp.zeros((beam_width,))
    pad_token_id = -1
    
    # Run beam selection!
    parent_beam_ids, token_ids, top_scores = _select_next_beams(logits_step0, cum_logprobs, beam_width, pad_token_id)
    
    print(f"Step 0 parent_beam_ids: {parent_beam_ids}")
    print(f"Step 0 token_ids: {token_ids}")
    print(f"Step 0 top_scores: {top_scores}")
    
    # Let's calculate expected results manually!
    # total_logprobs will be log_softmax(logits_step0) + 0!
    # Best scores in logits_step0 are:
    # Row 0: 2.0 (index 1).
    # Row 1: 2.0 (index 2).
    # Row 2: 2.0 (index 3).
    # Due to different sum of exponentials, Row 2 has the highest log_softmax score!
    # So top 3 candidates are returned in order: Row 2, Row 0, Row 1!
    # So parent_beam_ids should be [2, 0, 1]!
    # And token_ids should be [3, 1, 2]!
    
    assert np.allclose(parent_beam_ids, np.array([2, 0, 1]))
    assert np.allclose(token_ids, np.array([3, 1, 2]))
    
    # Now Step 1!
    logits_step1 = jnp.array([
        [0.1, 2.0, 0.5, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], # Candidates for (Parent 0, Token 1)
        [0.5, 0.1, 2.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], # Candidates for (Parent 1, Token 2)
        [0.1, 0.2, 0.3, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Candidates for (Parent 2, Token 3)
    ])
    
    # Update cum_logprobs!
    cum_logprobs = top_scores
    
    # Run beam selection again!
    parent_beam_ids, token_ids, top_scores = _select_next_beams(logits_step1, cum_logprobs, beam_width, pad_token_id)
    
    print(f"Step 1 parent_beam_ids: {parent_beam_ids}")
    print(f"Step 1 token_ids: {token_ids}")
    print(f"Step 1 top_scores: {top_scores}")
    
    # Assert results for Step 1!
    assert np.allclose(parent_beam_ids, np.array([2, 0, 1]))
    assert np.allclose(token_ids, np.array([4, 1, 2]))
    
    print("[SUCCESS] Beam search logic test passed with concrete arrays!")

def test_block_table_update():
    """
    Test the block table update logic in the JAX loop.
    """
    beam_width = 3
    max_tokens = 4
    start_block_idx = 1
    
    cur_block_tables = jnp.array([
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 3, 0, 0],
    ]) # Initial block tables
    
    step = 0
    parent_beam_ids = jnp.array([1, 0, 2]) # Candidate 0 follows Parent 1, Candidate 1 follows Parent 0...
    
    # Simulate update in loop!
    # This uses the fix with + 1 to include the block written in this step!
    cur_block_tables = cur_block_tables.at[:beam_width, :start_block_idx + step + 1].set(cur_block_tables[parent_beam_ids, :start_block_idx + step + 1])
    
    print(f"Updated block tables: {cur_block_tables}")
    
    # Expected results!
    # Slicing is up to start_block_idx + step + 1 = 1 + 0 + 1 = 2! (Exclusive!).
    # So columns 0 and 1 are shuffled!
    # Candidate 0 gets Parent 1's columns 0 and 1: [0, 2]!
    # Candidate 1 gets Parent 0's columns 0 and 1: [0, 1]!
    # Candidate 2 gets Parent 2's columns 0 and 1: [0, 3]!
    
    expected = jnp.array([
        [0, 2, 0, 0],
        [0, 1, 0, 0],
        [0, 3, 0, 0],
    ])
    
    assert jnp.all(cur_block_tables == expected)
    print("[SUCCESS] Block table update test passed!")
