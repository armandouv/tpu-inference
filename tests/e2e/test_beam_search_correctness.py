import os
import time
from unittest.mock import patch
import pytest
from vllm import LLM, SamplingParams

def test_beam_search_correctness():
    """
    Test that beam search produces valid results compared to greedy search.
    """
    model_name = "Qwen/Qwen3-0.6B"
    prompts = [
        "The capital of France is",
        "What is the hardest known natural substance on Earth?",
    ]
    
    # Greedy search params
    greedy_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
    )
    
    # Beam search params
    beam_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
        use_beam_search=True,
        best_of=30,
    )
    
    # Run baseline (standard greedy execution)
    print("Running Greedy Inference...")
    llm = LLM(model=model_name, max_model_len=2048)
    greedy_outputs = llm.generate(prompts, greedy_params)
    del llm
    time.sleep(5)
    
    # Run beam search inference
    print("Running Beam Search Inference...")
    llm = LLM(model=model_name, max_model_len=2048)
    beam_outputs = llm.generate(prompts, beam_params)
    del llm
    
    # Compare outputs
    assert len(greedy_outputs) == len(beam_outputs)
    
    for i, (greedy, beam) in enumerate(zip(greedy_outputs, beam_outputs)):
        greedy_text = greedy.outputs[0].text.strip()
        beam_text = beam.outputs[0].text.strip()
        
        print(f"Prompt: {prompts[i]}")
        print(f"  Greedy: {greedy_text}")
        print(f"  Beam:   {beam_text}")
        
        # We expect beam search to at least produce valid text!
        assert len(beam_text) > 0
