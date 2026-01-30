#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct BlockMeta {
  size_t block_id;
  size_t capacity_floats;  // capacity in float units
  size_t used_floats;      // used floats
  bool swapped_out;
  std::vector<std::pair<int, int>>
      owners;  // list of (sequence_id, layer_id) owners (for bookkeeping)
};

struct TensorDescriptor {
  // describes how to interpret the raw floats inside a block
  size_t block_id;
  size_t offset_floats;  // offset inside block (in floats)
  size_t seq_len;
  size_t num_heads;
  size_t head_dim;
};

class KVCache {
 public:
  // create cache with initial block capacity (floats) and block size (floats
  // per block)
  KVCache(size_t block_size_floats = 16384,
          const std::string& swap_dir = "./kv_swap");

  ~KVCache();

  // allocate storage for a token's KV tensor and register mapping
  // token_id: unique token identifier (user-defined)
  // seq_id, layer_id: owner bookkeeping
  // seq_len, num_heads, head_dim: tensor shape (resulting floats = seq_len *
  // num_heads * head_dim) returns pointer to float storage (nullptr if failed)
  float* allocate_for_token(int64_t token_id, int seq_id, int layer_id,
                            size_t seq_len, size_t num_heads, size_t head_dim);

  // get pointer to token's storage (loads from disk if needed), returns nullptr
  // if not found
  float* get_token_ptr(int64_t token_id);

  // get tensor descriptor metadata (optional)
  std::optional<TensorDescriptor> get_descriptor(int64_t token_id);

  // release one owner reference (seq+layer). When all owners removed, token
  // mapping removed.
  void release_token(int64_t token_id, int seq_id, int layer_id);

  // manually swap out a block (write to disk and free memory)
  bool swap_out_block(size_t block_id);

  // manually swap in a block (read from disk into memory)
  bool swap_in_block(size_t block_id);

  // report status
  std::string debug_status();

  // total memory used in floats
  size_t total_used_floats() const;

  // attempt to free enough memory by swapping least-recently-used blocks
  // (simple heuristic) target_free_floats: how many floats to free; returns
  // freed floats
  size_t try_free_by_swapping(size_t target_free_floats);

 private:
  struct Block {
    size_t block_id;
    std::vector<float> data;  // if swapped_out==true, data is empty
    BlockMeta meta;
    // For simplicity we keep a simple usage counter for LRU-ish behavior
    uint64_t last_access_counter;
  };

  // find a block with enough free space; if none, create new block
  Block* find_or_create_block_for(size_t needed_floats);

  // persistent filename for a block id
  std::string swap_filename_for(size_t block_id) const;

  size_t block_size_floats_;
  std::string swap_dir_;

  std::vector<std::unique_ptr<Block>> blocks_;

  // token_id -> descriptor
  std::unordered_map<int64_t, TensorDescriptor> token_map_;

  // mapping token->owner count is through BlockMeta owners and token_map
  mutable std::mutex mutex_;

  // global access counter
  uint64_t access_counter_;
};

#endif  // KV_CACHE_H
