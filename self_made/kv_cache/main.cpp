#include <cstring>
#include <iomanip>
#include <iostream>

#include "kv_cache.h"

int main() {
  // create a cache: block size = 4096 floats (~16KB if float=4B), swap files in
  // ./kv_swap
  KVCache cache(4096, "./kv_swap");

  // allocate some tokens with multi-head multi-layer shapes
  // token ids: 100, 101, 102
  int64_t t1 = 100;
  int64_t t2 = 101;
  int64_t t3 = 102;

  // shapes: seq_len x heads x head_dim
  size_t seq_len = 4;
  size_t heads = 8;
  size_t head_dim = 64;  // typical

  float* p1 = cache.allocate_for_token(t1, /*seq*/ 1, /*layer*/ 0, seq_len,
                                       heads, head_dim);
  if (!p1) {
    std::cerr << "alloc fail\n";
    return 1;
  }

  // fill p1 with some values
  size_t elems = seq_len * heads * head_dim;
  for (size_t i = 0; i < elems; i++) p1[i] = 1.0f * (i % 10);

  float* p2 = cache.allocate_for_token(t2, /*seq*/ 2, /*layer*/ 0, seq_len,
                                       heads, head_dim);
  float* p3 = cache.allocate_for_token(t3, /*seq*/ 1, /*layer*/ 1, seq_len,
                                       heads, head_dim);

  // write distinct patterns
  for (size_t i = 0; i < elems; i++)
    if (p2) p2[i] = 100.0f + (i % 13);
  for (size_t i = 0; i < elems; i++)
    if (p3) p3[i] = 200.0f + (i % 7);

  std::cout << "After allocations:\n" << cache.debug_status() << "\n";

  // force freeing memory by swapping out some blocks
  size_t want_free =
      elems * 2;  // try to free space equal to two token tensors (in floats)
  size_t freed = cache.try_free_by_swapping(want_free);
  std::cout << "Requested free floats: " << want_free
            << ", actually freed: " << freed << "\n";
  std::cout << cache.debug_status() << "\n";

  // try to access a swapped-out token: should load back automatically
  float* p1_again = cache.get_token_ptr(t1);
  if (p1_again) {
    std::cout << "p1[0]=" << p1_again[0] << " p1[1]=" << p1_again[1] << "\n";
  } else {
    std::cout << "Failed to get token 100 after swap.\n";
  }

  // release a token (owner)
  cache.release_token(t2, /*seq*/ 2, /*layer*/ 0);
  std::cout << "After release token 101:\n" << cache.debug_status() << "\n";

  // Allocate many tokens to force block reuse
  for (int i = 0; i < 20; i++) {
    int64_t tid = 200 + i;
    cache.allocate_for_token(tid, /*seq*/ i % 3, /*layer*/ i % 2, seq_len,
                             heads, head_dim);
  }
  std::cout << "After more allocations:\n" << cache.debug_status() << "\n";

  // cleanup: destructor will attempt to swap out remaining blocks
  std::cout << "Total used floats: " << cache.total_used_floats() << "\n";
  return 0;
}
