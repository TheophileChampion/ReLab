#ifndef COMPRESSORS_HPP
#define COMPRESSORS_HPP

#include <torch/extension.h>
#include <zlib.h>

/**
 * A class using zlib to compress and decompress torch tensors of type float.
 */
class ZCompressor {

private:

    // The deflate zlib stream.
    z_stream deflate_stream;

    // Precomputed values used to speed up compression.
    int uncompressed_size;
    int n_dims;
    int max_compressed_size;
    std::vector<float> compressed_output;
    std::vector<long int> shape;

public:

    /**
     * Create a zlib compressor.
     * @param height the height of the uncompressed images
     * @param width the width of the uncompressed images
     */
    ZCompressor(int height, int width);

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
     * @param tensor the tensor to decompress
     * @param output the buffer in which to decompress the tensor
     */
    void decode(const torch::Tensor &input, float *output);

private:

    /**
     * Compute the size of the tensor in bytes.
     * @param tensor the tensor whose size must be returned
     * @return the size of the tensor
     */
    int size_of(const torch::Tensor &tensor) const;
};

#endif //COMPRESSORS_HPP
