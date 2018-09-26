"""
This module includes some methods frequently used in the AlexNet activation
maximization, such as regularization methods and a data saving method.

Requires am_constants in order to work.
Antonia Hain, 2018
"""
import datetime
import cv2
import am_constants
import os
import numpy as np
import tensorflow as tf

def get_section(im_big, distances, xdim):
    """
    Crops a section out of an image for upscaling or jitter.

    Args:
        im_big: full image
        distances: distance image placeholder
        x_dim: image dimensions

    Returns:
        cropped image part, corresponding distance image part, indices of the
        pixels included in the cropped part and the positions from which the part
        was cropped.
    """

    im_big_dims = im_big.shape

    if am_constants.LARGER_IMAGE:
        rand_x = np.random.randint(-xdim[0], im_big_dims[1])
        rand_y = np.random.randint(-xdim[0], im_big_dims[2])
    if am_constants.JITTER:
        rand_x = np.random.randint(-am_constants.JITTER_STRENGTH, am_constants.JITTER_STRENGTH)
        rand_y = np.random.randint(-am_constants.JITTER_STRENGTH, am_constants.JITTER_STRENGTH)

    indx = [i % im_big_dims[1] for i in range(rand_x,rand_x+xdim[0])]
    indy = [i % im_big_dims[2] for i in range(rand_y,rand_y+xdim[1])]

    tmpsection = im_big[:,indx]
    im = tmpsection[:,:,indy]

    tmp_distances_crop = distances[indx]
    distances_crop = tmp_distances_crop[:,indy]

    return im, distances_crop, indx, indy, rand_x, rand_y

def plug_in_section(im_big, im, indx, indy, rand_x, rand_y):
    """
    Better description needed. Plugs section back into the image

    """

    # in case pixel indices beyond the border are supposed to be filled in on
    # the other side of the original image
    if am_constants.WRAP_AROUND:
        # get temporary image section of right composition (this might be
        # two sections of im_big stitched together if the selected section
        # crossed the image border)
        tmpsection = im_big[:,indx]
        tmpsection[:,:,indy] = im # put updated section in temporary image
        im_big[:,indx] = tmpsection # plug into big image

    # no wrap-around
    # pixels across borders were optimized, but are not to be included in big image again
    else:

        sectionindy = indy
        sectionindx = indx

        # if section crossed image border
        if 0 in indy:
            # get index of 0 to determine where the border was crossed
            wrap_index_y = indy.index(0)
            # to avoid cases where the subimage just randomly started at 0
            if wrap_index_y > 0:
                # cases where the subimage starts somewhere in the image and
                # crosses the right border
                if rand_y > 0:
                    # slice indices to be kept (start to right border)
                    indy = indy[:wrap_index_y]
                    # get corresponding indices for the subimage
                    # (necessary when the result image is upscaled)
                    sectionindy = range(len(indy))
                # cases where the subimage starts from beyond the left big
                # image border
                else:
                    # slice indices to be kept (starting from left border)
                    indy = indy[wrap_index_y:]
                    # get corresponding indices for the subimage
                    sectionindy = range(im.shape[2]-len(indy),im.shape[2])

            # update temporary image section with the pixels to be kept
            tmpsection = im[:,:,sectionindy]
        else:
            # no need to worry about boundaries if the border was not crossed
            tmpsection = im

        # similar to y dimension
        if 0 in indx:
            wrap_index_x = indx.index(0)
            if wrap_index_x > 0:
                if rand_x > 0:
                    indx = indx[:wrap_index_x]
                    sectionindx = range(len(indx))
                else:
                    indx = indx[wrap_index_x:]
                    sectionindx = range(im.shape[1]-len(indx),im.shape[1])
            tmpsection2 = tmpsection[:,sectionindx]
        else:
            tmpsection2 = tmpsection

        # plug updated, cut section into big image
        im_big[:,indx[0]:indx[0]+tmpsection2.shape[1],indy[0]:indy[0]+tmpsection2.shape[2]] = tmpsection2

    return im_big

def reg_blur(im):
    """
    Applies Gaussian blur on the image according to given constants in
    am_constants.

    Args:
        im: The image. It shouldn't matter if the image itself is 2- or 3d
            but it should be 1x(image_dimensions) as this is the network
            placeholder shape
    Returns:
        Blurred image
    """

    im[0] = cv2.GaussianBlur(im[0],am_constants.BLUR_KERNEL,am_constants.BLUR_SIGMA)
    return im

def reg_clip_norm(im):
    """
    Clips pixels with small norm in image (= sets them to 0) according to given
    constant in am_constants.

    Args:
        im: The image. It shouldn't matter if the image itself is 2- or 3d
            but it should be 1x(image_dimensions) as this is the network
            placeholder shape
    Returns:
        Image with small norm pixels clipped.
    """

    # get pixel norms
    norms = np.linalg.norm(im[0],axis=2)

    # clip all below percentile value in norms as computed by np percentile
    norm_threshold = np.percentile(norms, am_constants.NORM_PERCENTILE)

    # create mask and clip
    mask = norms < norm_threshold
    im[0][mask] = 0
    return im

def reg_clip_contrib(im, grad):
    """
    Clips pixels with small contribution in image (= sets them to 0) according
    to given constant in am_constants.

    Args:
        im: The image. It shouldn't matter if the image itself is 2- or 3d
            but it should be 1x(image_dimensions) as this is the network
            placeholder shape
        grad: Gradient computed by the network
    Returns:
        Image with small contribution pixels clipped.
    """

    contribs = np.sum(im * np.asarray(grad[0]),axis=3)
    # I tried using the absolute values but it didn't work well
    #contribs = np.sum(np.abs(im) * np.asarray(grad[0]),axis=3)

    # clip all below percentile value in contribs as computed by np percentile
    contrib_threshold = np.percentile(contribs, am_constants.CONTRIBUTION_PERCENTILE)

    # create mask and clip
    mask = contribs < contrib_threshold
    im[mask] = 0
    return im

def reg_border(im_placeholder, center_distance, loss):
    """
    Adds border region punishment by adding the product of the current image and
    the center distance image to the loss term

    Args:
        im_placeholder: placeholder for optimized image
        center_distance: placeholder for distance image (/distance image crop
            when upscaling is used)
        loss: loss tensor
    Returns:
        updated loss tensor
    """
    return loss + tf.reduce_sum(tf.multiply(tf.abs(im_placeholder),center_distance))

def save_am_data(outputim, idx, finish, steps, comptime, avg_steptime, top5, impath, paramfilename):
    """
    Saves the activation maximization result data (image and parameter values).
    The image is normalized or not depending on NORMALIZE_OUTPUT in am_constants.
    Image and data will be saved with a timestamp in order to be able to match
    the data to one another more easily.
    If the parameter file was not found, a new csv file including headline will
    be created at the specified location.

    Args:
        outputim: the image to save
        finish: indicating whether maximization process was finished or aborted
                due to too many steps being taken
        top5: top 5 object classes of outputim as determined by network
        impath: path where to save the image (not including filename)
        paramfilename: reference to file where parameters are saved

    """

    # get current time and condense to small id
    # ex. 24th of July, 19:28:15 will be 2407_192815
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")

    # normalize image if desired
    if am_constants.NORMALIZE_OUTPUT:
        cv2.normalize(outputim,  outputim, 0, 255, cv2.NORM_MINMAX)


    filename = f"{idx}_{timestamp}.png"

    cv2.imwrite(impath+filename, outputim)
    # write parameters to file with or without headline depending on if file
    # was found
    if not os.path.isfile(paramfilename):
        with open(paramfilename,"a") as param_file:
            print("Specified parameter output file not found. Preparing file including headline.")

            param_file.write("timestamp,ETA,RANDOMIZE_INPUT,BLUR_ACTIVATED,BLUR_KERNEL,BLUR_SIGMA,BLUR_FREQUENCY,L2_ACTIVATED,L2_LAMBDA,NORM_CLIPPING_ACTIVATED,NORM_CLIPPING_FREQUENCY,NORM_PERCENTILE,CONTRIBUTION_CLIPPING_ACTIVATED,CONTRIBUTION_CLIPPING_FREQUENCY,CONTRIBUTION_PERCENTILE,BORDER_REG_ACTIVATED,BORDER_FACTOR,BORDER_EXP, JITTER_ACTIVATED,JITTER_STRENGTH,LARGER_IMAGE,LOSS_GOAL,MAX_STEPS,finish,steps,comptime, average time per step\n")

            param_file.write(f"{timestamp},{am_constants.ETA},{am_constants.RANDOMIZE_INPUT},{am_constants.BLUR_ACTIVATED},{am_constants.BLUR_KERNEL[0]}x{am_constants.BLUR_KERNEL[1]},{am_constants.BLUR_SIGMA},{am_constants.BLUR_FREQUENCY},{am_constants.L2_ACTIVATED},{am_constants.L2_LAMBDA},{am_constants.NORM_CLIPPING_ACTIVATED},{am_constants.NORM_CLIPPING_FREQUENCY},{am_constants.NORM_PERCENTILE},{am_constants.CONTRIBUTION_CLIPPING_ACTIVATED},{am_constants.CONTRIBUTION_CLIPPING_FREQUENCY},{am_constants.CONTRIBUTION_PERCENTILE},{am_constants.BORDER_REG_ACTIVATED},{am_constants.BORDER_FACTOR},{am_constants.BORDER_EXP},{am_constants.JITTER},{am_constants.JITTER_STRENGTH},{am_constants.LARGER_IMAGE},{am_constants.LOSS_GOAL},{am_constants.MAX_STEPS},{finish},{steps},{comptime},{avg_steptime}\n")
    else:
        with open(paramfilename,"a") as param_file:
                param_file.write(f"{timestamp},{am_constants.ETA},{am_constants.RANDOMIZE_INPUT},{am_constants.BLUR_ACTIVATED},{am_constants.BLUR_KERNEL[0]}x{am_constants.BLUR_KERNEL[1]},{am_constants.BLUR_SIGMA},{am_constants.BLUR_FREQUENCY},{am_constants.L2_ACTIVATED},{am_constants.L2_LAMBDA},{am_constants.NORM_CLIPPING_ACTIVATED},{am_constants.NORM_CLIPPING_FREQUENCY},{am_constants.NORM_PERCENTILE},{am_constants.CONTRIBUTION_CLIPPING_ACTIVATED},{am_constants.CONTRIBUTION_CLIPPING_FREQUENCY},{am_constants.CONTRIBUTION_PERCENTILE},{am_constants.BORDER_REG_ACTIVATED},{am_constants.BORDER_FACTOR},{am_constants.BORDER_EXP},{am_constants.JITTER},{am_constants.JITTER_STRENGTH},{am_constants.LARGER_IMAGE},{am_constants.LOSS_GOAL},{am_constants.MAX_STEPS},{finish},{steps},{comptime},{avg_steptime}\n")
