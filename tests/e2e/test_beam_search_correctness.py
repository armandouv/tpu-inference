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
    
    # --- Subtest 1: Native Beam Search (Our Codepath) ---
    print("\n--- Subtest 1: Native Beam Search (Our Codepath) ---")
    
    beam_params1 = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
        use_beam_search=True,
        n=30,
    )
    
    print("Running Native Beam Search Inference...")
    beam_outputs1 = llm.generate(prompts, beam_params1)
    
    for i, beam in enumerate(beam_outputs1):
        print(f"\nPrompt: {prompts[i]}")
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
        
        # Compare Native vs CPU Beams (sorted, ignoring ordering)
        native_beams = sorted([out.text.strip() for out in beam_outputs1[i].outputs])
        
        # Strip prompt from CPU beams to get only the generated portion
        raw_cpu_beams = [seq.text.strip() for seq in output.sequences]
        stripped_cpu_beams = []
        prompt_prefix = prompts[i].strip()
        for cb in raw_cpu_beams:
            if cb.startswith(prompt_prefix):
                stripped = cb[len(prompt_prefix):].strip()
            else:
                stripped = cb
            stripped_cpu_beams.append(stripped)
        stripped_cpu_beams = sorted(stripped_cpu_beams)
        
        print(f"\n--- PARITY CHECK FOR PROMPT {i} ---")
        print(f"Native beams count: {len(native_beams)}")
        print(f"CPU beams count: {len(stripped_cpu_beams)}")
        
        # Assert same number of beams
        assert len(native_beams) == len(stripped_cpu_beams), f"Mismatch in number of beams: {len(native_beams)} vs {len(stripped_cpu_beams)}"
        
        # Calculate intersection and differences
        native_set = set(native_beams)
        cpu_set = set(stripped_cpu_beams)
        
        intersection = native_set.intersection(cpu_set)
        only_native = native_set - cpu_set
        only_cpu = cpu_set - native_set
        
        print(f"\nDetailed Beam Search Parity Analysis:")
        print(f"  Intersection (identical beams): {len(intersection)} / {len(native_set)}")
        print(f"  Beams only in TPU Native: {len(only_native)}")
        print(f"  Beams only in PyTorch CPU: {len(only_cpu)}")
        
        if only_native:
            print("  Examples of TPU Native only beams:")
            for b in sorted(list(only_native))[:5]:
                print(f"    - '{b}'")
        if only_cpu:
            print("  Examples of PyTorch CPU only beams:")
            for b in sorted(list(only_cpu))[:5]:
                print(f"    - '{b}'")
                
        # We expect extremely high overlap (e.g., >= 90% overlap), but because of floating-point precision differences
        # between Torch CPU and JAX TPU, minor divergences in candidate logprobs at boundary conditions are normal.
        overlap_percentage = (len(intersection) / float(len(native_set))) * 100.0
        print(f"Overlap percentage: {overlap_percentage:.2f}%")
        
        # Let's assert a high threshold of correctness parity!
        assert overlap_percentage == 100.0, f"Correctness overlap {overlap_percentage:.2f}% is below the 90.0% threshold!"
        print("SUCCESS: Overlap exceeds correctness threshold!")
            
    del llm

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    test_beam_search()
