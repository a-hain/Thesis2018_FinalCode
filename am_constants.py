"""
This module offers a hub to select all important constants used in the AlexNet
activation maximization module.
Antonia Hain, 2018
"""
import numpy as np

# in which layer the unit is found. needs to be a key in layer_dict in am_alexnet
LAYER_KEY = 'fc8'

# whether a unit from the chosen layer to perform activation maximization on
# should be chosen at random
RANDOM_UNIT = True

# this is only used if RANDOM_UNIT is set to False
# manually chosen unit to run activation maximization on
UNIT_INDEX = 849

# learning rate
ETA = 2500

# whether maximization is initialized with random input or flat colored image
RANDOMIZE_INPUT = True

# gaussian blur parameters
BLUR_ACTIVATED = True
BLUR_KERNEL = (3,3)
BLUR_SIGMA = 0.5
BLUR_FREQUENCY = 5 # how many steps between two blurs. paper used 4.

# l2 decay parameters
L2_ACTIVATED = True
L2_LAMBDA = 0.000001 # totally arbitrarily chosen

# low norm pixel clipping parameters
NORM_CLIPPING_ACTIVATED = False
NORM_CLIPPING_FREQUENCY = 25 # how many steps between pixel clippings
NORM_PERCENTILE = 50 # how many of the pixels are clipped

# low contribution pixel clipping parameters
# see norm clipping for explanation
CONTRIBUTION_CLIPPING_ACTIVATED = True
CONTRIBUTION_CLIPPING_FREQUENCY = 10
CONTRIBUTION_PERCENTILE = 30

# border regularizer - punishes pixel values the higher their distance to
# the image center
BORDER_REG_ACTIVATED = True

BORDER_FACTOR = 0.00000005 # modulates overall strength of effect
BORDER_EXP = 1.5 # the higher this factor, the stronger the center from distance is punished

LARGER_IMAGE = False # now called upscaling in thesis
IMAGE_SIZE_X = 350
IMAGE_SIZE_Y = 350

JITTER = False
JITTER_STRENGTH = 5

# whether pixels beyond the image borders in upscaling/jitter are updated via
# wrap-around
WRAP_AROUND = True

# convergence parameters
# relative(!) difference between loss and last LOSS_COUNT losses to converge to
LOSS_GOAL = 0.001
LOSS_COUNT = 100
MAX_STEPS = 2000 # how many steps to maximally take when optimizing an image

# to easily switch on and off the logging of the image and loss
TENSORBOARD_ACTIVATED = False

# whether to save output image normalized
NORMALIZE_OUTPUT = True

SAVE_PATH = "" # defaults to cwd
