#include "compressors.hpp"

/**
 * Implementation of the Compressor methods.
 */

std::unique_ptr<Compressor> Compressor::create() {
    // TODO bool use_cuda = (torch::cuda::is_available() and torch::cuda::device_count() >= 1);
    // TODO torch::DeviceType device_type = (use_cuda == true) ? torch::kCUDA: torch::kCPU;
    torch::DeviceType device_type = torch::kCPU;
    return Compressor::create(device_type);
}

std::unique_ptr<Compressor> Compressor::create(torch::DeviceType device_type) {
    if (device_type == torch::kCUDA) {
        return std::make_unique<NvidiaCompressor>(std::initializer_list<long int>({1, 84, 84}));
    } else {
        return std::make_unique<ZCompressor>();
    }
}

int Compressor::size_of(const torch::Tensor &tensor) const {
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}

/**
 * Implementation of the NvidiaCompressor methods.
 */

NvidiaCompressor::NvidiaCompressor(const std::initializer_list<long int> &shape) : shape(shape) {

    // Set the chunk size and algorithm type used for compression and decompression.
    const int chunk_size = 1 << 16;
    const nvcompBatchedGdeflateOpts_t algorithm = {1};

    // Create the cuda stream and deflate manager.
    cudaStreamCreate(&this->stream);
    this->manager = std::make_unique<GdeflateManager>(chunk_size, algorithm, this->stream);

    // Allocate the compression size buffer.
    cudaMallocAsync(&this->comp_size, sizeof(*this->comp_size), this->stream);

    // Create the tensor options used for create compressed and decompressed tensors.
    this->options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Create the footer and initialize the decompression size.
    this->decomp_size = 1;
    std::vector<int> footer;
    for (size_t i = 0; i < this->shape.size(); i++) {
        footer.push_back((int)this->shape[i]);
        this->decomp_size *= this->shape[i];
    }
    footer.push_back(shape.size());
    cudaMallocAsync(&this->footer, footer.size() * sizeof(int), this->stream);

    // Keep track of the number of dimensions and the footer's size.
    this->input_n_dims = shape.size();
    this->footer_size = (this->input_n_dims + 1) * sizeof(int);
}

NvidiaCompressor::~NvidiaCompressor() {

    // Free the compression size buffer.
    cudaFreeAsync(this->comp_size, this->stream);

    // Free the footer buffer.
    cudaFreeAsync(this->footer, this->stream);
}

torch::Tensor NvidiaCompressor::encode(torch::Tensor input) {

    // Create the compression configuration.
    int input_size = this->size_of(input);
    CompressionConfig comp_config = this->manager->configure_compression(input_size);

    // Create the buffer in which the compressed data will be stored.
    int output_size = comp_config.max_compressed_buffer_size + this->footer_size;
    uint8_t *comp_buffer;
    cudaMallocAsync(&comp_buffer, output_size, this->stream);

    // The actual compression work.
    this->manager->compress((const uint8_t *)input.data_ptr(), comp_buffer, comp_config, this->comp_size);
    cudaStreamSynchronize(this->stream);

    // Add a footer describing the tensor's shape.
    long int comp_size;
    cudaMemcpy(&comp_size, this->comp_size, sizeof(comp_size), cudaMemcpyDeviceToHost);
    comp_size /= sizeof(float);
    cudaMemcpy((int *)comp_buffer + comp_size, this->footer, this->footer_size, cudaMemcpyDeviceToDevice);

    // Create the output tensor, and free the allocated memory.
    torch::Tensor output = torch::from_blob(comp_buffer, {comp_size + this->input_n_dims + 1}, this->options).clone();
    cudaFreeAsync(comp_buffer, this->stream);
    return output;
}

torch::Tensor NvidiaCompressor::decode(torch::Tensor input) {

    // Create the decompression configuration.
    DecompressionConfig decomp_config = this->manager->configure_decompression((const uint8_t *)input.data_ptr());

    // Create the buffer in which the decompressed data will be stored.
    uint8_t *decomp_buffer;
    cudaMallocAsync(&decomp_buffer, decomp_config.decomp_data_size, this->stream);

    // The actual decompression work.
    this->manager->decompress(decomp_buffer, (const uint8_t *)input.data_ptr(), decomp_config);
    cudaStreamSynchronize(this->stream);

    // Create the output tensor, and free the allocated memory.
    torch::Tensor output = torch::from_blob(decomp_buffer, {this->decomp_size}, this->options).clone();
    cudaFreeAsync(decomp_buffer, this->stream);
    return output.view(at::IntArrayRef(this->shape));
}

/**
 * Implementation of the ZCompressor methods.
 */

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
