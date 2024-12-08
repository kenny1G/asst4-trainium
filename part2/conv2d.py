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

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    # pool_size is how big finale will be after the pca-esque max pooling is done
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Various tiling dimensions
    c_in_pmax = nl.tile_size.pmax
    c_out_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    """
    concpetually, we need to be multiplying our flattened image by 128 x 128 tilings
    of the Cin x Cout matrices made from each spatial position across the filters,
    (kernel_height, kernel_width,          n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax)
    spatial position across the filters,       number of tiles,                        128 x 128 tilings
    """

    """
    pseudocode on ED reformating filters (weights) into tiles:
    load in the weights into an SBUF array of shape  (n_tiles_out_channels, nl.par_dim(c_out_pmax), n_tiles_in_channels, 128, kernel_height, kernel_width)
    conceptually: take weights from #filters number of (fh x fw x cin) 3d blocks to 128 size batches of (fh x fw x 128) 3d blocks

    move data around using nl.copy to get an array of shape
    (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_out_pmax), c_in_pmax)
    transpose that to get an array of shape
    (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax), call this w
    conceptually: move things around such that each 3d block represents one spatial position across all filters
    """

    # reshape weights step 1: conceptually: convert weights from num_filters number of (fh x fw x cin)
    #  3d blocks to 128 size batches of (fh x fw x 128) 3d blocks
    w_tiled= nl.ndarray(
        shape=(n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, 128, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    # load in the weights into an SBUF array of shape
    # (n_tiles_out_channels, nl.par_dim(c_out_pmax), n_tiles_in_channels, 128, kernel_height, kernel_width)
    for cout in nl.affine_range(n_tiles_c_out):
        for cin in nl.affine_range(n_tiles_c_in):
            w_tiled[cout, :, cin, :, :, :] = nl.load(W[cout * 128: cout * 128 + 128, cin * 128 : cin * 128 + 128, :, :])


   # reshape weights step 2: onceptually: move things around such that each
   # 3d block represents one spatial position across all filters

   # move data around using nl.copy to get an array of shape
   # (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_out_pmax), c_in_pmax)
    w_moved = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

   # transpose that to get an array of shape
   # (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax),
   #  call this w
    w = nl.ndarray(
       shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax),
       dtype=W.dtype,
       buffer=nl.sbuf
    )

    for fh in nl.affine_range(filter_height):
        for fw in nl.affine_range(filter_width):
            for block in nl.affine_range(n_tiles_c_out):
                for depth in nl.affine_range(n_tiles_c_in):
                    w_moved[fh, fw, block, depth] = nl.copy(w_tiled[block, :, depth, :, fh, fw])
                    w[fh,fw, block, depth] = nisa.nc_transpose(w_moved[fh, fw, block, depth])


    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1
    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    n_rows_out_tile = 2
    input_tile_height = n_rows_out_tile + filter_height - 1
    n_tiles_out = out_height // n_rows_out_tile

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )


   # Process the images in batches
    for img in nl.affine_range(batch_size):
        for out_tile in nl.affine_range(n_tiles_out):
            #assign space in SBUF to store entire image, call it x
            x = nl.ndarray(shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_tile_height, input_width),
                dtype=X[img].dtype,
                buffer=nl.sbuf)

            tile_start = out_tile * n_rows_out_tile
            tile_end = tile_start + n_rows_out_tile
            for cin in nl.affine_range(n_tiles_c_in):
                # load part of input image corresponding to filter tiles
                x[cin, :, :, :] = nl.load(X[img, cin * 128: cin * 128 + 128, tile_start: tile_start + input_tile_height, :])

            # apply one filter at a time
            for block in nl.affine_range(n_tiles_c_out):
                #asign space in SBUF to store output
                output = nl.ndarray(shape=(nl.par_dim(c_out_pmax), n_rows_out_tile, out_width),
                    dtype=X_out[img].dtype,
                    buffer=nl.sbuf)

                # loop over output rows
                for out_row in nl.affine_range(n_rows_out_tile):
                    # assign space in PSUM to store output row
                    row_out = nl.zeros((nl.par_dim(c_out_pmax), out_width), nl.float32, buffer=nl.psum)

                    for fh in nl.affine_range(filter_height):
                        for fw in nl.affine_range(filter_width):
                            # for each spatial position in the filter
                            for cin in nl.affine_range(n_tiles_c_in):
                                # Perform matrix multiplication and accumulate in PSUM
                                row_out += nl.matmul(
                                    w[fh, fw, block, cin, :, :],
                                    x[cin, :, out_row + fh, fw : fw + out_width],
                                    transpose_x=True
                                )

                    #temp = nl.copy(temp, dtype=output[:, out_row, :].dtype)
                    # copy stuff from PSUM back to SBUF
                    output[:, out_row, :] = nl.add(row_out, nl.load(bias[block * 128 : block * 128 + 128]))

                # copy stuff from SBUF back to HBM
                nl.store(X_out[img, block * 128 : block * 128 + 128, tile_start:tile_end, :], value=output)

    return X_out
