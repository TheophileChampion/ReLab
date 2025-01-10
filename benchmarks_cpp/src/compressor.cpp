#include "compressor.hpp"

ZCompressor::ZCompressor() {

    // Initialize the zlib deflate stream.
    this->deflate_stream.zalloc = Z_NULL;
    this->deflate_stream.zfree = Z_NULL;
    this->deflate_stream.opaque = Z_NULL;

    // Initialize the zlib inflate stream.
    this->inflate_stream.zalloc = Z_NULL;
    this->inflate_stream.zfree = Z_NULL;
    this->inflate_stream.opaque = Z_NULL;
}

torch::Tensor ZCompressor::encode(torch::Tensor input) {

    // Create a placeholder for the compressed tensor.
    int input_size = this->size_of(input);
    int input_n_dims = input.dim();
    int output_size = input_size + (input_n_dims + 1) * sizeof(int);
    std::vector<float> output(output_size / sizeof(float));

    // Setup the "input" tensor as the input and "output" as the compressed output.
    this->deflate_stream.avail_in = (uInt) input_size;
    this->deflate_stream.next_in = (Bytef *) input.data_ptr();
    this->deflate_stream.avail_out = (uInt) output_size;
    this->deflate_stream.next_out = (Bytef *) output.data();

    // Perform the actual compression work.
    deflateInit(&this->deflate_stream, Z_BEST_COMPRESSION);
    deflate(&this->deflate_stream, Z_FINISH);
    deflateEnd(&this->deflate_stream);

    // Add a footer describing the tensor's shape.
    int true_output_size = ((char *)this->deflate_stream.next_out - (char *)output.data()) / sizeof(int);
    int *footer = ((int *) output.data()) + true_output_size;
    for (auto i = 0; i < input_n_dims; i++) {
        *footer = static_cast<int>(input.size(i));
        ++footer;
    }
    *footer = input_n_dims;

    // Resize the compressed tensor.
    output.resize(true_output_size + input_n_dims + 1);
    return torch::tensor(output);
}

torch::Tensor ZCompressor::decode(torch::Tensor input) {

    // Read the tensor's shape from the footer.
    int input_size = this->size_of(input);
    int *dim_footer = (int *)input.data_ptr() + input_size / sizeof(int) - 1;
    int n_dims = *dim_footer;

    std::vector<long int> shape;
    int *sizes_footer = dim_footer - n_dims;
    int output_size = 1;
    for (auto i = 0; i < n_dims; i++) {
        output_size *= *sizes_footer;
        shape.push_back((long int) *sizes_footer);
        ++sizes_footer;
    }

    // Create a placeholder for the decompressed tensor.
    std::vector<float> output(output_size);

    // setup "b" as the input and "c" as the compressed output
    this->inflate_stream.avail_in = (uInt) input_size;
    this->inflate_stream.next_in = (Bytef *) input.data_ptr();
    this->inflate_stream.avail_out = (uInt) output_size * sizeof(int);
    this->inflate_stream.next_out = (Bytef *) output.data();

    // Perform the actual decompression work.
    inflateInit(&this->inflate_stream);
    inflate(&this->inflate_stream, Z_NO_FLUSH);
    inflateEnd(&this->inflate_stream);

    // Resize the decompressed tensor.
    output.resize(output_size);
    return torch::tensor(output).view(at::IntArrayRef(shape));
}

int ZCompressor::size_of(const torch::Tensor &tensor) const {
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}
