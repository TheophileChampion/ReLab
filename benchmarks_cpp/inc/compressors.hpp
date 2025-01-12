#ifndef COMPRESSORS_HPP
#define COMPRESSORS_HPP

#include <torch/extension.h>
#include <zlib.h>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp.hpp>

using namespace nvcomp;

/**
 * A class that all compressor must implement.
 */
class Compressor {

public:

    /**
     * Create a compressor for CPU or GPU tensors.
     * @return the requested compressor
     */
    static std::unique_ptr<Compressor> create();

    /**
     * Create a compressor for CPU or GPU tensors.
     * @param device_type the type of device for which the compressor is created
     * @return the requested compressor
     */
    static std::unique_ptr<Compressor> create(torch::DeviceType device_type);

    /**
     * Compress the tensor passed as parameters.
     * @param tensor the tensor to compress
     * @return the compressed tensor
     */
    virtual torch::Tensor encode(torch::Tensor tensor) = 0;

    /**
     * Decompress the tensor passed as parameters.
     * @param tensor the tensor to decompress
     * @return the decompressed tensor
     */
    virtual torch::Tensor decode(torch::Tensor tensor) = 0;

protected:

    /**
     * Compute the size of the tensor in bytes.
     * @param tensor the tensor whose size must be returned
     * @return the size of the tensor
     */
    int size_of(const torch::Tensor &tensor) const;
};

/**
 * A class using nvcomp to compress and decompress torch tensors of type float.
 */
class NvidiaCompressor : public Compressor {

private:

    // The cuda stream and manager used for compression and decompression.
    cudaStream_t stream;
    std::unique_ptr<GdeflateManager> manager;

    // Pre-allocated structures to improve efficiency.
    size_t *comp_size;
    torch::TensorOptions options;
    int *footer;
    std::vector<long int> shape;
    long int decomp_size;
    int input_n_dims;
    int footer_size;

public:

    /**
     * Create a nvidia compressor.
     * @param shape the shape of the tensors to compress and decompress
     */
    NvidiaCompressor(const std::initializer_list<long int> &shape);

    /**
     * Destroy the nvidia compressor.
     */
    ~NvidiaCompressor();

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
};

/**
 * A class using zlib to compress and decompress torch tensors of type float.
 */
class ZCompressor : public Compressor {

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
};

#endif //COMPRESSORS_HPP
