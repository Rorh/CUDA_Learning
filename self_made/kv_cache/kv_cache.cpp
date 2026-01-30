#include "kv_cache.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

KVCache::KVCache(size_t block_size_floats, const std::string& swap_dir)
    : block_size_floats_(block_size_floats),
      swap_dir_(swap_dir),
      access_counter_(0) {
  if (!fs::exists(swap_dir_)) {
    fs::create_directories(swap_dir_);
  }
}

KVCache::~KVCache() {
  // try to write out remaining blocks (best-effort)
  for (auto& bptr : blocks_) {
    if (bptr && bptr->meta.swapped_out == false) {
      swap_out_block(bptr->block_id);
    }
  }
}

KVCache::Block* KVCache::find_or_create_block_for(size_t needed_floats) {
  // simple first-fit search for a block with enough free floats
  for (auto& bptr : blocks_) {
    if (!bptr) continue;
    if (bptr->meta.swapped_out)
      continue;  // can't allocate into swapped out block
    size_t free_space = bptr->meta.capacity_floats - bptr->meta.used_floats;
    if (free_space >= needed_floats) {
      return bptr.get();
    }
  }
  // create new block
  size_t new_block_id = blocks_.size();
  auto b = std::make_unique<Block>();
  b->block_id = new_block_id;
  b->data.resize(block_size_floats_);  // allocate full block (in floats)
  b->meta.block_id = new_block_id;
  b->meta.capacity_floats = block_size_floats_;
  b->meta.used_floats = 0;
  b->meta.swapped_out = false;
  b->meta.owners.clear();
  b->last_access_counter = ++access_counter_;
  blocks_.push_back(std::move(b));
  return blocks_.back().get();
}

std::string KVCache::swap_filename_for(size_t block_id) const {
  return fs::path(swap_dir_) /
         ("kv_block_" + std::to_string(block_id) + ".bin");
}

float* KVCache::allocate_for_token(int64_t token_id, int seq_id, int layer_id,
                                   size_t seq_len, size_t num_heads,
                                   size_t head_dim) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (token_map_.find(token_id) != token_map_.end()) {
    // already allocated; return pointer (ensure loaded)
    auto desc = token_map_[token_id];
    auto b = blocks_[desc.block_id].get();
    if (b->meta.swapped_out) {
      swap_in_block(desc.block_id);
    }
    b->last_access_counter = ++access_counter_;
    return b->data.data() + desc.offset_floats;
  }

  size_t needed = seq_len * num_heads * head_dim;
  Block* b = find_or_create_block_for(needed);
  if (!b) return nullptr;
  size_t offset = b->meta.used_floats;
  // update meta
  b->meta.used_floats += needed;
  b->meta.owners.push_back({seq_id, layer_id});
  b->last_access_counter = ++access_counter_;
  // create descriptor
  TensorDescriptor desc;
  desc.block_id = b->block_id;
  desc.offset_floats = offset;
  desc.seq_len = seq_len;
  desc.num_heads = num_heads;
  desc.head_dim = head_dim;
  token_map_[token_id] = desc;
  return b->data.data() + offset;
}

float* KVCache::get_token_ptr(int64_t token_id) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = token_map_.find(token_id);
  if (it == token_map_.end()) return nullptr;
  TensorDescriptor desc = it->second;
  if (desc.block_id >= blocks_.size()) return nullptr;
  Block* b = blocks_[desc.block_id].get();
  if (b->meta.swapped_out) {
    // bring back
    if (!swap_in_block(b->block_id)) return nullptr;
  }
  b->last_access_counter = ++access_counter_;
  return b->data.data() + desc.offset_floats;
}

std::optional<TensorDescriptor> KVCache::get_descriptor(int64_t token_id) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = token_map_.find(token_id);
  if (it == token_map_.end()) return std::nullopt;
  return it->second;
}

void KVCache::release_token(int64_t token_id, int seq_id, int layer_id) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = token_map_.find(token_id);
  if (it == token_map_.end()) return;
  TensorDescriptor desc = it->second;
  if (desc.block_id >= blocks_.size()) {
    token_map_.erase(it);
    return;
  }
  Block* b = blocks_[desc.block_id].get();
  // remove owner entry matching (seq_id, layer_id)
  auto& owners = b->meta.owners;
  for (auto o = owners.begin(); o != owners.end(); ++o) {
    if (o->first == seq_id && o->second == layer_id) {
      owners.erase(o);
      break;
    }
  }
  // if no owners remain for this token region, we free token_map entry;
  // note: block used_floats not decreased (simple allocator). In production
  // we'd use free lists inside block.
  token_map_.erase(it);
}

bool KVCache::swap_out_block(size_t block_id) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (block_id >= blocks_.size()) return false;
  Block* b = blocks_[block_id].get();
  if (b->meta.swapped_out) return true;
  // write data to file
  std::string fname = swap_filename_for(block_id);
  std::ofstream ofs(fname, std::ios::binary | std::ios::out);
  if (!ofs) {
    std::cerr << "Failed to open swap file for write: " << fname << "\n";
    return false;
  }
  // write used_floats and then floats
  ofs.write(reinterpret_cast<const char*>(&b->meta.used_floats),
            sizeof(b->meta.used_floats));
  ofs.write(reinterpret_cast<const char*>(b->data.data()),
            sizeof(float) * b->meta.used_floats);
  ofs.close();
  // free memory
  b->data.clear();
  b->data.shrink_to_fit();
  b->meta.swapped_out = true;
  return true;
}

bool KVCache::swap_in_block(size_t block_id) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (block_id >= blocks_.size()) return false;
  Block* b = blocks_[block_id].get();
  if (!b->meta.swapped_out) return true;
  std::string fname = swap_filename_for(block_id);
  std::ifstream ifs(fname, std::ios::binary | std::ios::in);
  if (!ifs) {
    std::cerr << "Failed to open swap file for read: " << fname << "\n";
    return false;
  }
  size_t used = 0;
  ifs.read(reinterpret_cast<char*>(&used), sizeof(used));
  b->data.resize(b->meta.capacity_floats);
  ifs.read(reinterpret_cast<char*>(b->data.data()), sizeof(float) * used);
  ifs.close();
  b->meta.used_floats = used;
  b->meta.swapped_out = false;
  // optionally remove file
  // fs::remove(fname);
  b->last_access_counter = ++access_counter_;
  return true;
}

std::string KVCache::debug_status() {
  std::lock_guard<std::mutex> lk(mutex_);
  std::ostringstream oss;
  oss << "KVCache status:\n";
  oss << " blocks: " << blocks_.size() << "\n";
  size_t i = 0;
  for (auto& bptr : blocks_) {
    if (!bptr) continue;
    oss << "  block " << i << " cap=" << bptr->meta.capacity_floats
        << " used=" << bptr->meta.used_floats
        << " swapped=" << (bptr->meta.swapped_out ? "Y" : "N")
        << " owners=" << bptr->meta.owners.size()
        << " last_access=" << bptr->last_access_counter << "\n";
    ++i;
  }
  oss << " token_map size=" << token_map_.size() << "\n";
  return oss.str();
}

size_t KVCache::total_used_floats() const {
  std::lock_guard<std::mutex> lk(mutex_);
  size_t total = 0;
  for (auto& bptr : blocks_) {
    if (!bptr) continue;
    total += bptr->meta.used_floats;
  }
  return total;
}

size_t KVCache::try_free_by_swapping(size_t target_free_floats) {
  std::lock_guard<std::mutex> lk(mutex_);
  // choose blocks that are not swapped and with smallest last_access (LRU-ish)
  std::vector<Block*> candidates;
  for (auto& bptr : blocks_) {
    if (!bptr) continue;
    if (!bptr->meta.swapped_out && bptr->meta.used_floats > 0) {
      candidates.push_back(bptr.get());
    }
  }
  // sort by least recently used
  std::sort(candidates.begin(), candidates.end(), [](Block* a, Block* b) {
    return a->last_access_counter < b->last_access_counter;
  });
  size_t freed = 0;
  for (Block* b : candidates) {
    if (freed >= target_free_floats) break;
    if (swap_out_block(b->block_id)) {
      freed += b->meta.used_floats;
    }
  }
  return freed;
}
