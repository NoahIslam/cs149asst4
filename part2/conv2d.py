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

    # Process the images in batches

    # OUT_TILE_DIM = 128
    # IN_TILE_DIM = 128

    # for b in nl.affine_range(batch_size):
    #     for outTile in nl.affine_range(out_channels // OUT_TILE_DIM):
    #         outStart = outTile * OUT_TILE_DIM
            
    #         out_accum = nl.zeros((OUT_TILE_DIM, out_height, out_width), dtype=X.dtype, buffer=nl.psum)
            
    #         for i in nl.affine_range(filter_height):
    #             for j in nl.affine_range(filter_width):
    #                 for inTile in nl.affine_range(in_channels // IN_TILE_DIM):
    #                     inStart = inTile * IN_TILE_DIM
                        
    #                     X_tile = X[b, inStart:inStart+IN_TILE_DIM, i:i+out_height, j:j+out_width]
    #                     X_sbuf = nl.ndarray(X_tile.shape, dtype=X.dtype, buffer=nl.sbuf)
    #                     nisa.dma_copy(src=X_tile, dst=X_sbuf)
                        
    #                     W_tile = nl.ndarray((OUT_TILE_DIM, IN_TILE_DIM), dtype=W.dtype, buffer=nl.sbuf)
    #                     nisa.dma_copy(src=W[outStart:outStart+OUT_TILE_DIM, inStart:inStart+IN_TILE_DIM, i, j], dst=W_tile)

    #                     W_T_psum = nisa.nc_transpose(W_tile)
    #                     W_T_sbuf = nisa.tensor_copy(W_T_psum, dtype=W.dtype)
    #                     out_accum += nisa.nc_matmul(W_T_sbuf, X_sbuf)
            
    #         out_sbuf = nl.copy(out_accum, dtype=X.dtype)
    #         nisa.dma_copy(src=out_sbuf, dst=X_out[b, outStart:outStart+OUT_TILE_DIM, :, :])

    # return X_out


    # OUT_CHANNEL_TILE_DIM = 128
    # IN_CHANNEL_TILE_DIM = 128
    # print(in_channels, out_channels, filter_height, filter_width)
    # print(out_channels, out_height, out_width)

    # for b in nl.affine_range(batch_size):
    #     for outTile in nl.affine_range(out_channels // OUT_CHANNEL_TILE_DIM):
    #         outStart = outTile * OUT_CHANNEL_TILE_DIM

    #         for h in nl.affine_range(((out_height + 7) // 8)):
    #             for w in nl.affine_range(((out_width + 7) // 8)):
    #                 hStart = h * 8
    #                 wStart = w * 8
    #                 hSize = min(8, out_height - hStart)
    #                 wSize = min(8, out_width  - wStart)

    #                 out_accum = nl.zeros((OUT_CHANNEL_TILE_DIM, 8, 8), dtype=X.dtype, buffer=nl.psum)
                    
    #                 for i in nl.affine_range(filter_height):
    #                     for j in nl.affine_range(filter_width):
    #                         for inTile in nl.affine_range(in_channels // IN_CHANNEL_TILE_DIM):
                                

    #                             inStart = inTile * IN_CHANNEL_TILE_DIM
                                
    #                             X_tile = X[b, inStart:inStart+IN_CHANNEL_TILE_DIM, i+hStart:i+hStart+hSize, j+wStart:j+wStart+wSize]
    #                             X_sbuf = nl.ndarray(X_tile.shape, dtype=X.dtype, buffer=nl.sbuf)
    #                             nisa.dma_copy(src=X_tile, dst=X_sbuf)
                                
    #                             W_tile = nl.ndarray((OUT_CHANNEL_TILE_DIM, IN_CHANNEL_TILE_DIM), dtype=W.dtype, buffer=nl.sbuf)
    #                             nisa.dma_copy(src=W[outStart:outStart+OUT_CHANNEL_TILE_DIM, inStart:inStart+IN_CHANNEL_TILE_DIM, i, j], dst=W_tile)

    #                             W_T_psum = nisa.nc_transpose(W_tile)
    #                             W_T_sbuf = nisa.tensor_copy(W_T_psum, dtype=W.dtype)
    #                             out_accum += nisa.nc_matmul(W_T_sbuf, X_sbuf)
                    
    #                 out_sbuf = nl.copy(out_accum, dtype=X.dtype)
    #                 nisa.dma_copy(src=out_sbuf[:, :hSize, :wSize], dst=X_out[b, outStart:outStart+OUT_CHANNEL_TILE_DIM, hStart:hStart+hSize, wStart:wStart+wSize])

    # return X_out



    OUT_CHANNEL_TILE_DIM = 128
    IN_CHANNEL_TILE_DIM = 128
    HEIGHT_TILE = 1

    for b in nl.affine_range(batch_size):
        for outTile in nl.affine_range(out_channels // OUT_CHANNEL_TILE_DIM):
            outStart = outTile * OUT_CHANNEL_TILE_DIM

            for h in nl.affine_range(out_height // HEIGHT_TILE):
                    hStart = h * HEIGHT_TILE

                    out_accum = nl.zeros((OUT_CHANNEL_TILE_DIM, HEIGHT_TILE, out_width), dtype=X.dtype, buffer=nl.psum)
                    
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            for inTile in nl.affine_range(in_channels // IN_CHANNEL_TILE_DIM):
                                

                                inStart = inTile * IN_CHANNEL_TILE_DIM
                                
                                X_tile = X[b, inStart:inStart+IN_CHANNEL_TILE_DIM, i+hStart:i+hStart+HEIGHT_TILE, j:j+out_width]
                                X_sbuf = nl.ndarray(X_tile.shape, dtype=X.dtype, buffer=nl.sbuf)
                                nisa.dma_copy(src=X_tile, dst=X_sbuf)
                                
                                W_tile = nl.ndarray((OUT_CHANNEL_TILE_DIM, IN_CHANNEL_TILE_DIM), dtype=W.dtype, buffer=nl.sbuf)
                                nisa.dma_copy(src=W[outStart:outStart+OUT_CHANNEL_TILE_DIM, inStart:inStart+IN_CHANNEL_TILE_DIM, i, j], dst=W_tile)

                                W_T_psum = nisa.nc_transpose(W_tile)
                                W_T_sbuf = nisa.tensor_copy(W_T_psum, dtype=W.dtype)
                                out_accum += nisa.nc_matmul(W_T_sbuf, X_sbuf)
                    
                    out_sbuf = nl.copy(out_accum, dtype=X.dtype)
                    nisa.dma_copy(src=out_sbuf, dst=X_out[b, outStart:outStart+OUT_CHANNEL_TILE_DIM, hStart:hStart+HEIGHT_TILE, :])

    return X_out