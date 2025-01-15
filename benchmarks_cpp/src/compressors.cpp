#include "compressors.hpp"
#include "replay_buffer.hpp"

ZCompressor::ZCompressor(int height, int width) {

    // Initialize the zlib deflate stream.
    this->deflate_stream.zalloc = Z_NULL;
    this->deflate_stream.zfree = Z_NULL;
    this->deflate_stream.opaque = Z_NULL;

    // Pre-compute uncompressed tensor shape, size and number of dimensions.
    this->shape.push_back(height);
    this->shape.push_back(width);
    this->uncompressed_size = height * width * sizeof(float);
    this->n_dims = 2;

    // Allocate the buffer storing the compressed tensor.
    this->max_compressed_size = this->uncompressed_size + (this->n_dims + 1) * sizeof(int);
    this->compressed_output.resize(this->max_compressed_size / sizeof(float));
}

torch::Tensor ZCompressor::encode(const torch::Tensor &input) {

    // Setup the deflate stream.
    this->deflate_stream.avail_in = (uInt) this->uncompressed_size;
    this->deflate_stream.next_in = (Bytef *) input.data_ptr();
    this->deflate_stream.avail_out = (uInt) this->max_compressed_size;
    this->deflate_stream.next_out = (Bytef *) this->compressed_output.data();

    // Perform the actual compression work.
    deflateInit(&this->deflate_stream, Z_BEST_COMPRESSION);
    deflate(&this->deflate_stream, Z_FINISH);
    deflateEnd(&this->deflate_stream);

    // Return the compressed tensor.
    int compressed_size = ((char *)this->deflate_stream.next_out - (char *)this->compressed_output.data()) / sizeof(int);
    return torch::from_blob(this->compressed_output.data(), {compressed_size}).clone();
}

torch::Tensor ZCompressor::decode(const torch::Tensor &input) {
    torch::Tensor output = torch::zeros(at::IntArrayRef(this->shape));
    this->decode(input, (float *)output.data_ptr());
    return output;
}

void ZCompressor::decode(const torch::Tensor &input, float *output) {

    // Initialize the zlib inflate stream.
    z_stream inflate_stream;
    inflate_stream.zalloc = Z_NULL;
    inflate_stream.zfree = Z_NULL;
    inflate_stream.opaque = Z_NULL;

    // Setup the inflate stream used for decompression.
    inflate_stream.avail_in = (uInt) this->size_of(input);
    inflate_stream.next_in = (Bytef *) input.data_ptr();
    inflate_stream.avail_out = (uInt) this->uncompressed_size;
    inflate_stream.next_out = (Bytef *) output;

    // Perform the actual decompression work.
    inflateInit(&inflate_stream);
    inflate(&inflate_stream, Z_NO_FLUSH);
    inflateEnd(&inflate_stream);
}

int ZCompressor::size_of(const torch::Tensor &tensor) const {
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}
