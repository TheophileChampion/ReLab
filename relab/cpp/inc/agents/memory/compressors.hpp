// Copyright 2025 Theophile Champion. No Rights Reserved.
/**
 * @file compressors.hpp
 * @brief Declaration of all compression algorithms.
 */

#ifndef RELAB_CPP_INC_AGENTS_MEMORY_COMPRESSORS_HPP_
#define RELAB_CPP_INC_AGENTS_MEMORY_COMPRESSORS_HPP_

#include <memory>
#include <vector>
#include <zlib.h>

#include <torch/extension.h>

namespace relab::agents::memory {

/**
 * Enumeration of all supported compression types.
 */
enum class CompressorType {
  RAW = 0,  // No compression.
  ZLIB = 1  // Compression using the zlib deflate format.
};

/**
 * @brief A class that all compressors must implement.
 */
class Compressor {
  public:
    /**
     * Create the requested compressor.
     * @param height the height of the uncompressed images
     * @param width the width of the uncompressed images
     * @param type the type of compression to use
     * @return the requested compressor
     */
    static std::unique_ptr<Compressor>
    create(int height, int width, CompressorType type = CompressorType::ZLIB);

    /**
     * Ensure the destructor of child classes are called.
     */
    virtual ~Compressor();

    /**
     * Compress the tensor passed as parameters.
     * @param tensor the tensor to compress
     * @return the compressed tensor
     */
    virtual torch::Tensor encode(const torch::Tensor &tensor) = 0;

    /**
     * Decompress the tensor passed as parameters.
     * @param tensor the tensor to decompress
     * @return the decompressed tensor
     */
    virtual torch::Tensor decode(const torch::Tensor &tensor) = 0;

    /**
     * Decompress the tensor passed as parameters.
     * @param input the tensor to decompress
     * @param output the buffer in which to decompress the tensor
     */
    virtual void decode(const torch::Tensor &input, float *output) = 0;

  protected:
    /**
     * Compute the size of the tensor in bytes.
     * @param tensor the tensor whose size must be returned
     * @return the size of the tensor
     */
    int size_of(const torch::Tensor &tensor) const;
};

/**
 * @brief A class that does not compress the tensors.
 */
class NoCompression : public Compressor {
  private:
    int uncompressed_size;

  public:
    /**
     * Create an identity compressor, i.e., no compression of the tensors is
     * actually performed.
     * @param height the height of the uncompressed images
     * @param width the width of the uncompressed images
     */
    NoCompression(int height, int width);

    /**
     * Destroy the identity compressor.
     */
    ~NoCompression();

    /**
     * Compress the tensor passed as parameters.
     * @param tensor the tensor to compress
     * @return the compressed tensor
     */
    torch::Tensor encode(const torch::Tensor &tensor);

    /**
     * Decompress the tensor passed as parameters.
     * @param tensor the tensor to decompress
     * @return the decompressed tensor
     */
    torch::Tensor decode(const torch::Tensor &tensor);

    /**
     * Decompress the tensor passed as parameters.
     * @param input the tensor to decompress
     * @param output the buffer in which to decompress the tensor
     */
    void decode(const torch::Tensor &input, float *output);
};

/**
 * @brief A class using zlib to compress and decompress torch tensors of type
 * float.
 */
class ZCompressor : public Compressor {
  private:
    // The deflate zlib stream.
    z_stream deflate_stream;

    // Precomputed values used to speed up compression.
    int uncompressed_size;
    int n_dims;
    int max_compressed_size;
    std::vector<float> compressed_output;
    std::vector<int64_t> shape;

  public:
    /**
     * Create a zlib compressor.
     * @param height the height of the uncompressed images
     * @param width the width of the uncompressed images
     */
    ZCompressor(int height, int width);

    /**
     * Destroy the compressor.
     */
    ~ZCompressor();

    /**
     * Compress the tensor passed as parameters.
     * @param tensor the tensor to compress
     * @return the compressed tensor
     */
    torch::Tensor encode(const torch::Tensor &tensor);

    /**
     * Decompress the tensor passed as parameters.
     * @param tensor the tensor to decompress
     * @return the decompressed tensor
     */
    torch::Tensor decode(const torch::Tensor &tensor);

    /**
     * Decompress the tensor passed as parameters.
     * @param input the tensor to decompress
     * @param output the buffer in which to decompress the tensor
     */
    void decode(const torch::Tensor &input, float *output);
};
}  // namespace relab::agents::memory

#endif  // RELAB_CPP_INC_AGENTS_MEMORY_COMPRESSORS_HPP_
