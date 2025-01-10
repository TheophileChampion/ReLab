#ifndef COMPRESSOR_HPP
#define COMPRESSOR_HPP

#include <torch/extension.h>
#include <zlib.h>


/**
 * A class using zlib to compress and decompress torch tensor of type float.
 */
class ZCompressor {

private:

    z_stream deflate_stream;
    z_stream inflate_stream;

public:

    /**
     * Create a zlib compressor.
     */
    ZCompressor();

    /**
     * Compress the tensor passed as parameters.
     * @param tensor the tensor to compress
     * @return the compressed tensor
     */
    torch::Tensor encode(torch::Tensor tensor);

    /**
     * Decompress the tensor passed as parameters.
     * @param tensor the tensor to decompress
     * @return the decompressed tensor
     */
    torch::Tensor decode(torch::Tensor tensor);

    /**
     * Compute the size of the tensor in bytes.
     * @param tensor the tensor whose size must be returned
     * @return the size of the tensor
     */
    int size_of(const torch::Tensor &tensor) const;
};

#endif //COMPRESSOR_HPP
