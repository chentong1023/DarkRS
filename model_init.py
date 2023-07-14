# register the newly defined datasets

from dataset import fastec_rs
from dataset import bsrsc
from dataset import darkrs

# register the defined losses
from loss import DSUNL1Loss, VariationLoss

# register adarsc model
from model import rsc_arch
from model import vi_arch
from model import dark_rsc_arch

from model import ifrnet
from model import ifrnet_L
from model import qvinet
# from model import rscnet
