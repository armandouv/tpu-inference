import os
import time
import pytest
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams

def test_beam_search():
    """
    Test both Native Beam Search (our codepath) and CPU-based Beam Search (llm.beam_search).
    """
    model_name = "Qwen/Qwen3-0.6B"
    
    print("Initializing LLM...")
    llm = LLM(model=model_name, max_model_len=2048, gpu_memory_utilization=0.5, enable_prefix_caching=True, max_logprobs=60, kv_cache_dtype="bfloat16")
    
    # Simplify to a single prompt for reproduction!
    prompts = [
        "The most unexpected thing happened when I opened the door,",
    ]
    
    # --- Subtest 1: Native Beam Search vs Greedy (OUR Codepath) ---
    print("\n--- Subtest 1: Native Beam Search (Our Codepath) ---")
    
    greedy_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
        logprobs=1,
    )
    
    beam_params1 = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
        use_beam_search=True,
        logprobs=1,
        n=1,  # Set to 1 to execute internally as single request with devices unrolled
        extra_args={"beam_width": 30},
    )
    
    print("Running Greedy Inference...")
    greedy_outputs = llm.generate(prompts, greedy_params)
    
    print("Running Native Beam Search Inference...")
    beam_outputs1 = llm.generate(prompts, beam_params1)
    
    assert len(greedy_outputs) == len(beam_outputs1)
    
    for i, (greedy, beam) in enumerate(zip(greedy_outputs, beam_outputs1)):
        greedy_text = greedy.outputs[0].text.strip()
        print(f"\nPrompt: {prompts[i]}")
        print(f"  Greedy: {greedy_text}")
        print(f"  Native Beam Search Results:")
        for j, output in enumerate(beam.outputs):
            print(f"    Beam {j}: {output.text.strip()}")
            
        unique_outputs = set(out.text.strip() for out in beam.outputs)
        print(f"  Unique beams found: {len(unique_outputs)}")
        
    # --- Subtest 2: CPU-based Beam Search (llm.beam_search) ---
    print("\n--- Subtest 2: CPU-based Beam Search (llm.beam_search) ---")
    
    beam_params2 = BeamSearchParams(
        beam_width=30,
        max_tokens=4,
        ignore_eos=True,
    )
    
    print("Running CPU-based Beam Search Inference...")
    beam_outputs2 = llm.beam_search(prompts, beam_params2)
    
    assert len(beam_outputs2) == len(prompts)
    
    for i, output in enumerate(beam_outputs2):
        print(f"\nPrompt: {prompts[i]}")
        print(f"  CPU Beam Search Results:")
        for j, seq in enumerate(output.sequences):
            print(f"    Beam {j}: {seq.text.strip()}")
            
        unique_outputs = set(seq.text.strip() for seq in output.sequences)
        print(f"  Unique beams found: {len(unique_outputs)}")
        
    del llm
