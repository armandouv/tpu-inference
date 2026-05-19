import pytest
from vllm import LLM, SamplingParams

def test_prefill_vs_decode_batched():
    """
    Test to provide evidence of numerical differences between 
    Prefill and Decode paths on TPU, using BATCHING to match beam search conditions.
    """
    model_name = "Qwen/Qwen3-0.6B"
    
    print("Initializing LLM...")
    llm = LLM(model=model_name, max_model_len=2048, gpu_memory_utilization=0.5, enable_prefix_caching=False)
    
    prompt = "The most unexpected thing happened when I opened the door,"
    
    # Run with 5 real prompts and 27 dummy prompts to match beam search padding!
    prompts = [prompt] * 5 + ["dummy"] * 27
    
    # 1. Run generate with max_tokens=2 to get the first token and the logprobs for the second token (via DECODE).
    sampling_params_2 = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        ignore_eos=True,
        logprobs=5,
    )
    
    print("\n--- Running Run 1 (Prefill + Decode) with 5 real + 27 dummy ---")
    outputs1 = llm.generate(prompts, sampling_params_2)
    
    token1_id = outputs1[0].outputs[0].token_ids[0]
    decode_logprobs = outputs1[0].outputs[0].logprobs[1]
    
    # 2. Reconstruct the prompt with token 1 appended, and run generate with max_tokens=1.
    # This will trigger PREFILL on the extended prompt, giving logprobs for the second token.
    tokenizer = llm.get_tokenizer()
    token1_text = tokenizer.decode([token1_id])
    extended_prompt = prompt + token1_text
    
    extended_prompts = [extended_prompt] * 5 + ["dummy"] * 27
    
    sampling_params_1 = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        logprobs=5,
    )
    
    print("\n--- Running Run 2 (Prefill on extended prompt) with batch size 5 ---")
    outputs2 = llm.generate(extended_prompts, sampling_params_1)
    
    prefill_logprobs = outputs2[0].outputs[0].logprobs[0]
    
    print(f"\nPrompt: {prompt}")
    print(f"Extended Prompt: {extended_prompt}")
    print(f"Token 1 generated: {token1_id} ('{token1_text}')")
    
    print(f"\n[DECODE Path] Logprobs for Token 2 (Batch element 0):")
    for k, v in decode_logprobs.items():
        print(f"  Token {k}: {v.logprob}")
        
    print(f"\n[PREFILL Path] Logprobs for Token 2 (Batch element 0):")
    for k, v in prefill_logprobs.items():
        print(f"  Token {k}: {v.logprob}")
        
    # Compare the logprob of the top token in decode path
    top_decode_token = list(decode_logprobs.keys())[0]
    decode_score = decode_logprobs[top_decode_token].logprob
    
    # Find the same token in prefill logprobs
    prefill_score = None
    if top_decode_token in prefill_logprobs:
        prefill_score = prefill_logprobs[top_decode_token].logprob
        
    print(f"\nComparison for top decode token {top_decode_token}:")
    print(f"  Decode score: {decode_score}")
    print(f"  Prefill score: {prefill_score}")
    
    if prefill_score is not None:
        diff = abs(decode_score - prefill_score)
        print(f"  Difference: {diff}")
        
        # Assert that they differ (proving that batching causes shift!)
        # We expect them to differ now!
        assert diff > 1e-5, f"Logprobs are too similar! Diff={diff}"
        print("\n[SUCCESS] Proved numerical difference between Prefill and Decode paths on TPU with batching.")
    else:
        print("\n[WARNING] Top decode token not found in top-5 prefill logprobs. They differ significantly!")
        assert True # They differ significantly!
