#include <cuda_runtime.h>

#include <cassert>

#include "common/cuda_utility.hpp"

#define MAX_NUM_SEQUENCES 32
#define TOKENS_PER_BLOCK 4
#define MAX_NUM_BLOCKS_PER_SEQ 64  // context_size = 64 * 4 = 256
#define MAX_BLOCKS (MAX_NUM_SEQUENCES * MAX_NUM_BLOCKS_PER_SEQ)

// 初始化页表：为 MAX_BLOCKS 个条目分配空间并全部置为 -1
// page_table 作为二级指针传入，以便 cudaMalloc 之后地址能写回调用方
__host__ void init_page_table(unsigned int** page_table) {
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(page_table),
                             MAX_BLOCKS * sizeof(unsigned int)));
  for (int i = 0; i < MAX_BLOCKS; ++i) {
    (*page_table)[i] = static_cast<unsigned int>(-1);
  }
}

// 为 KV cache 的物理块数组分配显存
// blocks: 逻辑上是长度为 MAX_BLOCKS * d_k 的一维数组
// 使用二级指针以便把分配得到的地址写回调用方
template <typename scalar_t>
__host__ void init_blocks(scalar_t** blocks, int d_k) {
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(blocks),
                             MAX_BLOCKS * d_k * sizeof(scalar_t)));
}

// 预留的接口：为某个虚拟地址对应的物理块分配空间
// 当前实现仅作占位，真正的 page_table/blocks 维护逻辑后续可补全
template <typename scalar_t>
__host__ void allocate_block(unsigned int physical_address, scalar_t* block,
                             int d_k, int* page_table, scalar_t* cache_blocks) {
  (void)physical_address;
  (void)page_table;
  (void)cache_blocks;
  checkCudaErrors(
      cudaMalloc(&block, TOKENS_PER_BLOCK * d_k * sizeof(scalar_t)));
}

// 从 (seq_idx, block_idx) 这样的“虚拟地址”查到物理地址
__device__ unsigned int translate_address(unsigned int seq_idx,
                                          unsigned int block_idx,
                                          unsigned int* page_table) {
  unsigned int virtual_address = seq_idx * MAX_NUM_BLOCKS_PER_SEQ + block_idx;
  if (virtual_address >= MAX_BLOCKS || block_idx >= MAX_NUM_BLOCKS_PER_SEQ) {
    printf("Error: the virtual address is out of range.\n");
  }

  unsigned int physical_address = page_table[virtual_address];
  return physical_address;
}

// 根据物理地址在一维 cache_blocks 中取出某个块的起始指针
template <typename scalar_t>
__device__ __inline__ scalar_t* fetch_block(scalar_t* cache_blocks,
                                            int physical_address) {
  return &cache_blocks[physical_address];
}