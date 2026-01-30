add_executable(conv im2col.cu)

target_link_libraries(conv PRIVATE ${CUDA_LIBRARIES} cublas)