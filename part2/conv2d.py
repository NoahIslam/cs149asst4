import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    c_out_pmax = c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    # Reshape, transpose, and load in filters/weights to SBUF before running computations
    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))

    W_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    W_transposed = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
    w = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=W.dtype, buffer=nl.sbuf)

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        nisa.dma_copy(src=W[c_out_tile], dst=W_sbuf[c_out_tile])

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    W_transposed[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(W_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype=W.dtype)
                    tmp = nisa.nc_transpose(W_transposed[i, j, c_out_tile, c_in_tile])
                    w[i, j, c_out_tile, c_in_tile] = nl.copy(tmp, dtype=W.dtype)


    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # Row buffer holding filter_height + 1 rows of input image
        x_rows = nl.ndarray((n_tiles_c_in, nl.par_dim(c_in_pmax), filter_height + 1, input_width), dtype=X.dtype, buffer=nl.sbuf)

        for i in nl.affine_range(filter_height + 1):
            for c_in_tile in nl.affine_range(n_tiles_c_in):
                nisa.dma_copy(src=X[b, c_in_tile*c_in_pmax:(c_in_tile+1)*c_in_pmax, i, :], dst=x_rows[c_in_tile, :, i, :])

        bias_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), 1), dtype=bias.dtype, buffer=nl.sbuf)
        for c_out_tile in nl.affine_range(n_tiles_c_out):
            nisa.dma_copy(src=bias[c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax], dst=bias_sbuf[c_out_tile, :, 0])
    


        # Build result two out_rows at a time
        for out_row in nl.affine_range(out_height // 2):
            for c_out_tile in nl.affine_range(n_tiles_c_out):
                
                result1 = nl.zeros((128, out_width), nl.float32, buffer=nl.psum)
                result2 = nl.zeros((128, out_width), nl.float32, buffer=nl.psum)

                for i in nl.affine_range(filter_height):
                    row_idx1 = (out_row * 2 + i) % (filter_height + 1)
                    row_idx2 = (out_row * 2 + i + 1) % (filter_height + 1)
                    for j in nl.affine_range(filter_width):
                        for c_in_tile in nl.affine_range(n_tiles_c_in):
                            result1 += nisa.nc_matmul(w[i, j, c_out_tile, c_in_tile, :, :], x_rows[c_in_tile, :, row_idx1, j:j+out_width])
                            result2 += nisa.nc_matmul(w[i, j, c_out_tile, c_in_tile, :, :], x_rows[c_in_tile, :, row_idx2, j:j+out_width])

                res_sb1 = nl.copy(result1, dtype=X.dtype)
                res_sb2 = nl.copy(result2, dtype=X.dtype)
                row_with_bias1 = nisa.tensor_tensor(res_sb1, bias_sbuf[c_out_tile], op=nl.add)
                row_with_bias2 = nisa.tensor_tensor(res_sb2, bias_sbuf[c_out_tile], op=nl.add)

                if pool_size == 1:
                    nisa.dma_copy(src=row_with_bias1, dst=X_out[b, c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax, out_row * 2, :])
                    nisa.dma_copy(src=row_with_bias2, dst=X_out[b, c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax, out_row * 2 + 1, :])
                else:
                    row_max = nisa.tensor_tensor(row_with_bias1, row_with_bias2, op=nl.maximum)
                    even_cols = row_max[:, 0::2]
                    odd_cols = row_max[:, 1::2]
                    pooled_row = nisa.tensor_tensor(even_cols, odd_cols, op=nl.maximum)
                    nisa.dma_copy(src=pooled_row, dst=X_out[b, c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax, out_row, :])
            

            # Get the next two input rows for the rolling buffer
            next_row1 = out_row * 2 + filter_height + 1
            next_row2 = out_row * 2 + filter_height + 2
            if next_row2 < input_height:
                replace_row1 = (out_row * 2) % (filter_height + 1)
                replace_row2 = (out_row * 2 + 1) % (filter_height + 1)
                for c_in_tile in nl.affine_range(n_tiles_c_in):
                    nisa.dma_copy(src=X[b, c_in_tile*c_in_pmax:(c_in_tile+1)*c_in_pmax, next_row1, :], dst=x_rows[c_in_tile, :, replace_row1, :])
                    nisa.dma_copy(src=X[b, c_in_tile*c_in_pmax:(c_in_tile+1)*c_in_pmax, next_row2, :], dst=x_rows[c_in_tile, :, replace_row2, :])

    return X_out
