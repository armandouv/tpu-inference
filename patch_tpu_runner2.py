import re
with open("tpu_inference/runner/tpu_runner.py", "r") as f:
    content = f.read()

# Fix Step 0 JAX device_get
old_step_0 = """            # Step 0: Initial branching!
            first_beam_logits = logits_step[0]
            logits_cpu = np.asarray(jax.device_get(first_beam_logits))
            top30_token_ids = np.argsort(logits_cpu)[-beam_width:][::-1]
            top30_logprobs = logits_cpu[top30_token_ids]
            
            next_tokens_cpu = top30_token_ids.reshape(beam_width, 1)
            next_tokens = device_array(self.mesh, next_tokens_cpu, sharding=input_ids.sharding)
            cum_logprobs = device_array(self.mesh, top30_logprobs, sharding=input_ids.sharding)"""

new_step_0 = """            # Step 0: Initial branching!
            first_beam_logits = logits_step[0]
            top30_logprobs, top30_token_ids = jax.lax.top_k(first_beam_logits, beam_width)
            
            next_tokens = top30_token_ids.reshape(beam_width, 1)
            cum_logprobs = top30_logprobs"""

content = content.replace(old_step_0, new_step_0)

# Fix Block Theft
old_block_theft = """                # Scratchpad Block Theft!
                beam_0_block_ids = self.input_batch.block_table[0].get_cpu_tensor()[0]
                
                # Find free block IDs!
                total_blocks = self.cache_config.num_gpu_blocks
                allocated_ids = set()
                for r_idx in range(self.input_batch.num_reqs):
                    ids = self.input_batch.block_table[0].get_cpu_tensor()[r_idx]
                    for b in ids:
                        if b > 0:
                            allocated_ids.add(int(b))
                            
                free_ids = []
                for b in range(1, total_blocks):
                    if b not in allocated_ids:
                        free_ids.append(b)
                        if len(free_ids) == beam_width:
                            break
                            
                assert len(free_ids) == beam_width, f"Could not find {beam_width} free blocks! Found {len(free_ids)}"
                
                # Build cur_block_tables of shape (32, max_blocks)!
                cur_block_tables_np = np.zeros((32, self.max_num_blocks_per_req), dtype=np.int32)
                cur_block_tables_np[0] = beam_0_block_ids
                
                for b in range(1, beam_width):
                    beam_b_blocks = np.copy(beam_0_block_ids)
                    zeros = np.where(beam_b_blocks == 0)[0]
                    if len(zeros) > 0:
                        beam_b_blocks[zeros[0]] = free_ids[b]
                    cur_block_tables_np[b] = beam_b_blocks"""

new_block_theft = """                # Properly allocated blocks from scheduler
                beam_0_block_ids = self.input_batch.block_table[0].get_cpu_tensor()[0]
                nonzero_indices = np.where(beam_0_block_ids > 0)[0]
                
                num_extra = beam_width - 1
                if len(nonzero_indices) >= num_extra:
                    extra_blocks = beam_0_block_ids[nonzero_indices[-num_extra:]]
                    base_blocks = np.copy(beam_0_block_ids)
                    base_blocks[nonzero_indices[-num_extra:]] = 0
                else:
                    extra_blocks = np.zeros(num_extra, dtype=np.int32)
                    base_blocks = np.copy(beam_0_block_ids)
                
                # Build cur_block_tables of shape (32, max_blocks)!
                cur_block_tables_np = np.zeros((32, self.max_num_blocks_per_req), dtype=np.int32)
                cur_block_tables_np[0] = base_blocks
                
                for b in range(1, beam_width):
                    beam_b_blocks = np.copy(base_blocks)
                    zeros = np.where(beam_b_blocks == 0)[0]
                    if len(zeros) > 0 and extra_blocks[b - 1] > 0:
                        beam_b_blocks[zeros[0]] = extra_blocks[b - 1]
                    cur_block_tables_np[b] = beam_b_blocks"""

content = content.replace(old_block_theft, new_block_theft)

with open("tpu_inference/runner/tpu_runner.py", "w") as f:
    f.write(content)
