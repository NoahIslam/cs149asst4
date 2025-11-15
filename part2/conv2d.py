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

    # Reshape and PRE-TRANSPOSE weights ONCE (biggest optimization!)
    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))

    weight_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    weight_copy = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
    w = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=W.dtype, buffer=nl.sbuf)

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        nisa.dma_copy(src=W[c_out_tile], dst=weight_sbuf[c_out_tile])

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    weight_copy[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(weight_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype=W.dtype)
                    tmp = nisa.nc_transpose(weight_copy[i, j, c_out_tile, c_in_tile])
                    w[i, j, c_out_tile, c_in_tile] = nl.copy(tmp, dtype=W.dtype)





    # Process the images in batches

<<<<<<< HEAD

    OUT_CHANNEL_TILE_DIM = 128
    IN_CHANNEL_TILE_DIM = 128
    HEIGHT_TILE = 1

    for b in nl.affine_range(batch_size):
        for outTile in nl.affine_range(out_channels // OUT_CHANNEL_TILE_DIM):
            outStart = outTile * OUT_CHANNEL_TILE_DIM
            
            bias_row = nl.ndarray((OUT_CHANNEL_TILE_DIM, 1), dtype=X.dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=bias[outStart:outStart+OUT_CHANNEL_TILE_DIM], dst=bias_row)

            for h in nl.affine_range(out_height // HEIGHT_TILE):
                hStart = h * HEIGHT_TILE
                
                out_accum = nl.zeros((OUT_CHANNEL_TILE_DIM, HEIGHT_TILE, out_width), dtype=nl.float32, buffer=nl.psum)
                
                for inTile in nl.affine_range(in_channels // IN_CHANNEL_TILE_DIM):
                    inStart = inTile * IN_CHANNEL_TILE_DIM
                    
                    X_sbuf = nl.ndarray((IN_CHANNEL_TILE_DIM, HEIGHT_TILE + filter_height - 1, input_width),  dtype=X.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=X[b, inStart:inStart+IN_CHANNEL_TILE_DIM, hStart:hStart+HEIGHT_TILE+filter_height-1, :], dst=X_sbuf)
                    
                    W_sbuf_all = nl.ndarray((OUT_CHANNEL_TILE_DIM, IN_CHANNEL_TILE_DIM, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=W[outStart:outStart+OUT_CHANNEL_TILE_DIM, inStart:inStart+IN_CHANNEL_TILE_DIM, :, :], dst=W_sbuf_all)
    
                    
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            X_tile = X_sbuf[:, i:i+HEIGHT_TILE, j:j+out_width]
                            
                            W_tile = W_sbuf_all[:, :, i, j]
                            
                            W_T_psum = nisa.nc_transpose(W_tile)
                            W_T_sbuf = nisa.tensor_copy(W_T_psum, dtype=W.dtype)
                            out_accum += nisa.nc_matmul(W_T_sbuf, X_tile)
                
                out_sbuf = nl.copy(out_accum, dtype=X.dtype)
                for w_col in nl.affine_range(out_width):
                    out_sbuf[:, 0, w_col] += bias_row[:, 0]
                nisa.dma_copy(src=out_sbuf, dst=X_out[b, outStart:outStart+OUT_CHANNEL_TILE_DIM, hStart:hStart+HEIGHT_TILE, :])
    return X_out



    # c_in_pmax = nl.tile_size.pmax
    # n_tiles_c_in = in_channels // c_in_pmax
    # c_out_pmax = c_in_pmax
    # n_tiles_c_out = out_channels // c_out_pmax


    # # - load in the weights into an SBUF array of shape (n_tiles_out_channels, nl.par_dim(c_out_pmax), n_tiles_in_channels, 128, kernel_height, kernel_width)
    # # - move data around using nl.copy to get an array of shape (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_out_pmax), c_in_pmax)
    # # - transpose that to get an array of shape (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax), call this w

    # W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))

    # weight_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype = W.dtype, buffer = nl.sbuf)
    # weight_copy = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
    # w = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=W.dtype, buffer=nl.sbuf)

    # for c_out_tile in nl.affine_range(n_tiles_c_out):
    #     weight_sbuf[c_out_tile] = nl.ndarray((nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    #     nisa.dma_copy(src=W[c_out_tile], dst=weight_sbuf[c_out_tile])

    # for c_out_tile in nl.affine_range(n_tiles_c_out):
    #     for c_in_tile in nl.affine_range(n_tiles_c_in):
    #         for i in nl.affine_range(filter_height):
    #             for j in nl.affine_range(filter_width):
    #                 weight_copy[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(weight_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype = W.dtype)
    #                 w[i, j, c_out_tile, c_in_tile] = nisa.nc_transpose(weight_copy[i, j, c_out_tile, c_in_tile])

    # # Process the images in batches
    # for b in nl.affine_range(batch_size):
    #     '''TODO:
    #     - x : (n_tiles_c_in, nl.par_dim(c_in_pmax), image_height, image_width)
    #     loop over n_tiles_c_in:
    #         - load corresponding part of input image into x
    #     HINT: First declare an x = nl.ndarray ... 
    #         Then, loop over n_tiles_c_in
    #         Load X into x
    #     '''
    #     for c_out_tile in nl.affine_range(n_tiles_c_out):
    #         '''TODO:
    #         - assign space in SBUF to store output
    #         - space should be of shape : (nl.par_dim(c_out_pmax), out_height, out_width)
    #         HINT: Use nl.ndarray
    #         '''
    #         for out_row in nl.affine_range(out_height):
    #         '''
    #             DONE FOR YOU:
    #             - assign space in PSUM to store output row
    #         '''
    #             result = nl.zeros((128, out_width), nl.float32, buffer = nl.psum)
    #             for i in nl.affine_range(filter_height):
    #                 for j in nl.affine_range(filter_width):
    #                     for c_in_tile in nl.affine_range(n_tiles_c_in):
    #                         '''
    #                         TODO:
    #                         - matmul w[i, j, c_out_tile, c_in_tile, :, :].T with
    #                         x[c_in_tile, :, i, j:j + out_width]

    #                         Hint: You can use nl.matmul to do a matrix multiply with a transpose. It takes
    #                         in another parameter to specify whether the left-hand matrix should be transposed
    #                         or not. See https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.matmul.html
    #                         - matmul w[i, j, c_out_tile, c_in_tile, :, :].T with
    #                         x[c_in_tile, :, i, j:j + out_width]
    #                         '''
    #                         - matmul w[i, j, c_out_tile, c_in_tile, :, :].T with
    #                         x[c_in_tile, :, i, j:j + out_width]
    #                         - matmul w[i, j, c_out_tile, c_in_tile, :, :].T with
    #                         x[c_in_tile, :, i, j:j + out_width]
    #                         - matmul w[i, j, c_out_tile, c_in_tile, :, :].T with
    #             ''' TODO:
    #             - copy stuff from PSUM back to SBUF
    #             HINT: Use nl.copy
    #             '''
    #         ''' TODO:
    #         - copy stuff from SBUF back to HBM
    #         HINT: Use nl.store
    #         ''';
=======
    for b in nl.affine_range(batch_size):
        # Rolling input row buffer: keep only filter_height rows
        x_rows = nl.ndarray((n_tiles_c_in, nl.par_dim(c_in_pmax), filter_height, input_width), dtype=X.dtype, buffer=nl.sbuf)

        # Prime the ring buffer with the first filter_height rows
        for i in nl.affine_range(filter_height):
            for c_in_tile in nl.affine_range(n_tiles_c_in):
                nisa.dma_copy(src=X[b, c_in_tile*c_in_pmax:(c_in_tile+1)*c_in_pmax, i, :],
                            dst=x_rows[c_in_tile, :, i, :])

        # Stage bias for all output tiles once
        bias_sbuf_all = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), 1), dtype=bias.dtype, buffer=nl.sbuf)
        for c_out_tile in nl.affine_range(n_tiles_c_out):
            nisa.dma_copy(src=bias[c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax],
                        dst=bias_sbuf_all[c_out_tile, :, 0])
        
    

        even_row = nl.zeros((128, out_width), nl.float32, buffer=nl.psum)
        max_pool_row = nl.zeros((128, out_pool_width), nl.float32, buffer=nl.psum)
        # Process row-by-row using rolling buffer
        for out_row in nl.affine_range(out_height):

            for c_out_tile in nl.affine_range(n_tiles_c_out):
                # Accumulate in PSUM for this output row
                result = nl.zeros((128, out_width), nl.float32, buffer=nl.psum)

                for i in nl.affine_range(filter_height):
                    row_slot = (out_row + i) % filter_height  # Ring buffer indexing
                    for j in nl.affine_range(filter_width):
                        for c_in_tile in nl.affine_range(n_tiles_c_in):
                            result += nisa.nc_matmul(w[i, j, c_out_tile, c_in_tile, :, :],
                                                    x_rows[c_in_tile, :, row_slot, j:j+out_width])

                # Convert from PSUM and add bias
                res_sb = nl.copy(result, dtype=X.dtype)
                row_with_bias = nisa.tensor_tensor(res_sb, bias_sbuf_all[c_out_tile], op=np.add)

                # Write row directly to HBM
                if pool_size == 1:
                    nisa.dma_copy(src=row_with_bias,
                                dst=X_out[b, c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax, out_row, :])
                else:
                    if out_row % pool_size == 0:
                        even_row = row_with_bias
                    else:
                        row_max = nisa.tensor_tensor(even_row, row_with_bias, op=np.maximum)
                        pooled_row = nl.ndarray((128, out_pool_width), dtype=row_max.dtype, buffer=nl.sbuf)
                        for pw in nl.affine_range(out_pool_width):
                            c0 = 2 * pw
                            c1 = c0 + 1
                            colmax = nisa.tensor_tensor(row_max[:, c0:c0+1], row_max[:, c1:c1+1], op=np.maximum)
                            nisa.tensor_copy(src=colmax, dst=pooled_sb[:, pw:pw+1])
                        nisa.dma_copy(src=pooled_row,
                                    dst=X_out[b, c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax, out_row//pool_size, :])
                    

            # Prefetch the next needed input row into rolling buffer
            next_input_row = out_row + filter_height
            if next_input_row < input_height:
                overwrite_slot = out_row % filter_height
                for c_in_tile in nl.affine_range(n_tiles_c_in):
                    nisa.dma_copy(src=X[b, c_in_tile*c_in_pmax:(c_in_tile+1)*c_in_pmax, next_input_row, :],
                                dst=x_rows[c_in_tile, :, overwrite_slot, :])

    return X_out
>>>>>>> c3ff8f8 (passes some stuff for no maxpool)
