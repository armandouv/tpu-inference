import re
with open("tpu_inference/runner/tpu_runner.py", "r") as f:
    content = f.read()

# Fix Reshaping
old_reshape = """            next_tokens = jnp.concatenate(all_tokens, axis=-1)
            
            final_logprobs_token_ids = jnp.concatenate(all_logprobs_token_ids, axis=0)
            final_logprobs_scores = jnp.concatenate(all_logprobs_scores, axis=0)
            final_ranks = jnp.concatenate(all_ranks, axis=0)
            
            token_ids_3d = final_logprobs_token_ids.reshape(max_tokens, beam_width, -1)
            scores_3d = final_logprobs_scores.reshape(max_tokens, beam_width, -1)
            ranks_2d = final_ranks.reshape(max_tokens, beam_width)"""

new_reshape = """            next_tokens = jnp.concatenate(all_tokens, axis=-1)
            
            final_logprobs_token_ids = jnp.concatenate(all_logprobs_token_ids, axis=0)
            final_logprobs_scores = jnp.concatenate(all_logprobs_scores, axis=0)
            final_ranks = jnp.concatenate(all_ranks, axis=0)
            
            actual_steps = len(all_tokens)
            token_ids_3d = final_logprobs_token_ids.reshape(actual_steps, beam_width, -1)
            scores_3d = final_logprobs_scores.reshape(actual_steps, beam_width, -1)
            ranks_2d = final_ranks.reshape(actual_steps, beam_width)"""

content = content.replace(old_reshape, new_reshape)

# Fix EOS early stopping
old_loop = """                next_tokens = token_ids.reshape(beam_width, 1)
                all_tokens.append(next_tokens)
                
                cum_logprobs = top_scores
                
                step_logprobs = self._compute_and_gather_logprobs(logits_step, next_tokens.ravel(), self.model_config.max_logprobs)
                all_logprobs_token_ids.append(step_logprobs.logprob_token_ids)
                all_logprobs_scores.append(step_logprobs.logprobs)
                all_ranks.append(step_logprobs.selected_token_ranks)
                
                self.kv_caches = _shuffle_kv_caches(self.kv_caches, parent_beam_ids, cur_block_tables, beam_width)"""

new_loop = """                eos_token_id = first_req_state.sampling_params.eos_token_id if first_req_state.sampling_params else None
                if eos_token_id is not None:
                    # Inherit parent EOS state
                    parent_active = active_beams[parent_beam_ids]
                    token_ids = jnp.where(parent_active, token_ids, eos_token_id)
                    active_beams = parent_active & (token_ids != eos_token_id)
                
                next_tokens = token_ids.reshape(beam_width, 1)
                all_tokens.append(next_tokens)
                
                cum_logprobs = top_scores
                
                step_logprobs = self._compute_and_gather_logprobs(logits_step, next_tokens.ravel(), self.model_config.max_logprobs)
                all_logprobs_token_ids.append(step_logprobs.logprob_token_ids)
                all_logprobs_scores.append(step_logprobs.logprobs)
                all_ranks.append(step_logprobs.selected_token_ranks)
                
                self.kv_caches = _shuffle_kv_caches(self.kv_caches, parent_beam_ids, cur_block_tables, beam_width)
                
                if eos_token_id is not None and not np.asarray(jax.device_get(jnp.any(active_beams))):
                    break"""

content = content.replace(old_loop, new_loop)

# Add active_beams initialization before the loop
old_init = """            lora_metadata = self.lora_utils.extract_lora_metadata()
            
            for step in range(1, max_tokens):"""

new_init = """            lora_metadata = self.lora_utils.extract_lora_metadata()
            
            active_beams = jnp.ones((beam_width,), dtype=jnp.bool_)
            if first_req_state.sampling_params and first_req_state.sampling_params.eos_token_id is not None:
                active_beams = (next_tokens.ravel() != first_req_state.sampling_params.eos_token_id)
            
            for step in range(1, max_tokens):"""

content = content.replace(old_init, new_init)

# Fix cu_num_generated_tokens
old_cu = """                cu_num_generated_tokens=[i * max_tokens for i in range(beam_width + 1)]"""
new_cu = """                cu_num_generated_tokens=[i * actual_steps for i in range(beam_width + 1)]"""
content = content.replace(old_cu, new_cu)

with open("tpu_inference/runner/tpu_runner.py", "w") as f:
    f.write(content)
