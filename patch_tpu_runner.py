import functools
import jax

@functools.partial(jax.jit, static_argnums=(3, 4), donate_argnums=(0,))
def _shuffle_kv_caches(kv_caches, parent_beam_ids, block_tables_2d, beam_width, start_block_idx):
    def shuffle_layer(kv_cache):
        for b_idx in range(beam_width):
            p_b_idx = parent_beam_ids[b_idx]
            child_blocks = block_tables_2d[b_idx, start_block_idx:]
            parent_blocks = block_tables_2d[p_b_idx, start_block_idx:]
            kv_cache = kv_cache.at[child_blocks].set(kv_cache[parent_blocks])
        return kv_cache
    
    return [shuffle_layer(c) for c in kv_caches]

