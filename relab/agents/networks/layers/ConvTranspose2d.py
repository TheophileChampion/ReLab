import math

from torch import Tensor, nn

from relab.helpers.Typing import DataType, Device, ScalarOrTuple


class ConvTranspose2d(nn.Module):
    """!
    @brief Class implementing an up-convolutional layer with support for padding same.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ScalarOrTuple[int],
        stride: ScalarOrTuple[int] = 1,
        padding: ScalarOrTuple[int] = 0,
        output_padding: ScalarOrTuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: ScalarOrTuple[int] = 1,
        padding_mode: str = "zeros",
        device: Device = None,
        dtype: DataType = None,
    ) -> None:
        """!
        Create a convolutional transpose layer.
        @param in_channels: number of channels in the input image
        @param out_channels: number of channels produced by the convolution
        @param kernel_size: size of the convolution kernel
        @param stride: stride of the convolution
        @param padding: `dilation * (kernel_size - 1) - padding` zero-padding added to both sides of the input.
        Can also be `valid` (no padding) or `same` (padding to ensure `output shape = input shape * stride`)
        @param output_padding: additional size added to one side of each dimension in the output shape
        @param groups: number of blocked connections from input channels to output channels
        @param bias: if True, adds a learnable bias to the output
        @param dilation: spacing between kernel elements
        @param padding_mode: only `zeros` padding mode is supported
        @param device: the device on which the tensors are allocated
        @param dtype: the data type of the tensors
        """

        # Call parent constructor.
        super().__init__()

        # Check that the padding parameter is set to a proper value.
        if isinstance(padding, str) and padding not in ["valid", "same"]:
            raise Exception(
                "In constructor of custom layer: 'ConvTranspose2d', padding type not supported."
            )

        # @var padding
        # Zero-padding added to both sides of the input.
        self.padding = (0, 0) if isinstance(padding, str) else padding

        # @var padding_same
        # Boolean indicating if output shape equals input shape times stride.
        self.padding_same = False
        if isinstance(padding, str) and padding == "same":
            self.padding_same = True

        # Ensure that stride and kernel_size are tuples.
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # @var p_left
        # Amount of padding to remove from the left side of the output.
        self.p_left = math.floor((kernel_size[0] - stride[0]) / 2)

        # @var p_right
        # Amount of padding to remove from the right side of the output.
        self.p_right = kernel_size[0] - stride[0] - self.p_left

        # @var p_top
        # Amount of padding to remove from the top of the output.
        self.p_top = math.floor((kernel_size[1] - stride[1]) / 2)

        # @var p_bottom
        # Amount of padding to remove from the bottom of the output.
        self.p_bottom = kernel_size[1] - stride[1] - self.p_top

        # @var up_conv_layer
        # Transposed convolution layer used for the deconvolution operation.
        self.up_conv_layer = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            self.padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        """!
        Compute the forward pass of the custom ConvTranspose2d.
        @param x: the input of the layer.
        @return the output of the layer.
        """
        out = self.up_conv_layer(x)
        return self.apply_padding_same(out) if self.padding_same else out

    def apply_padding_same(self, tensor: Tensor) -> Tensor:
        """!
        Apply the padding same to the input tensor.
        @param tensor: the input tensor.
        @return the output tensor after applying the padding same.
        """
        return tensor[
            :,
            :,  # Select all batch elements and all channels.
            self.p_left : tensor.shape[2] - self.p_right,
            self.p_top : tensor.shape[3] - self.p_bottom,
        ]
