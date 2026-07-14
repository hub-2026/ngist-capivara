import logging
import os

import numpy as np
from scipy import ndimage
from astropy.io import fits
from printStatus import printStatus
from scipy.spatial import cKDTree

"""
PURPOSE:
  

"""


def generateSpatialBins(config, cube):
    """
    This function applies the Voronoi-binning algorithm of Cappellari & Copin
    2003 (ui.adsabs.harvard.edu/?#abs/2003MNRAS.342..345C) to the data. It can
    be accounted for spatial correlations in the noise (see function sn_func()).
    A BIN_ID is assigned to every spaxel. Spaxels which were masked are excluded
    from the Voronoi-binning, but are assigned a negative BIN_ID, with the
    absolute value of the BIN_ID corresponding to the nearest Voronoi-bin that
    satisfies the minimum SNR threshold.  All results are saved in a dedicated
    table to provide easy means of matching spaxels and bins.
    """

    # Generate the Voronoi bins
    printStatus.running("Defining the segments with Capivara")
    logging.info("Defining the segments with Capivara")

    # Read maskfile
    maskfile = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_mask.fits"
    )
    
    mask = fits.open(maskfile)[1].data.MASK
    mask = mask.ravel()
    idxUnmasked = np.where(mask == 0)[0]
    idxMasked = np.where(mask == 1)[0]
    
    # Read segmentfile
    segmentfile = (
        os.path.join(os.path.dirname(config["GENERAL"]["INPUT"]), 
                     config["SPATIAL_BINNING"]["SEGMENTS"])
        + ".fits"
    )
    with fits.open(segmentfile) as hdul:
        segments = hdul['SEGMENTS'].data

    try:
        maskMap = mask.reshape(segments.shape).copy()
        valid = ~maskMap.astype(bool)
        binMap = np.full(segments.shape, np.nan)
        _, inv = np.unique(segments[valid], return_inverse=True)
        binMap[valid] = inv
        
        binNum = binMap[~np.isnan(binMap)]
        binNum = np.array(binNum, dtype=int)
        
        ny, nx = binMap.shape #binNum.shape
        x = np.arange(nx) * cube["pixelsize"]
        y = np.arange(ny) * cube["pixelsize"]
        xv, yv = np.meshgrid(x, y, indexing='xy')
        good = np.unique(binNum)
        xNode = ndimage.mean(xv, labels=binMap, index=good)
        yNode = ndimage.mean(yv, labels=binMap, index=good)
        
        nPixels = np.bincount(binNum)
        sn = np.full_like(nPixels, fill_value='99') # To improve
        
        printStatus.updateDone("Defining the segments")
        print("             " + str(np.max(binNum) + 1) + " segments generated!")
        logging.info(str(np.max(binNum) + 1) + " Binning with segments generated!") 
    except ValueError as e:
        # Any uncaught exceptions, causing the galaxy to be skipped
        printStatus.updateFailed("Defining the segments")
        print(
            "The segment routine returned the following error: \n"
            + str(e)
        )
        logging.error(
            "Defining the segments failed. The segment routine"
            + "returned the following error: \n"
            + str(e)
        )
        return "SKIP"

    # Find the nearest Voronoi bin for the pixels outside the Voronoi region
    binNum_outside = find_nearest_voronoibin(
        cube["x"], cube["y"], idxMasked, xNode, yNode
    )

    # Generate extended binNum-list:
    #   Positive binNum (including zero) indicate the Voronoi bin of the spaxel (for unmasked spaxels)
    #   Negative binNum indicate the nearest Voronoi bin of the spaxel (for masked spaxels)
    ubins = np.unique(binNum)
    # nbins = len(ubins)
    binNum_long = np.full(len(cube["x"]), np.nan)
    binNum_long[idxUnmasked] = binNum
    binNum_long[idxMasked] = -1 * binNum_outside

    # Save bintable: data for *ALL* spectra inside and outside of the Voronoi region!
    save_table(
        config,
        cube["x"],
        cube["y"],
        cube["signal"],
        cube["snr"],
        binNum_long,
        ubins,
        xNode,
        yNode,
        sn,
        nPixels,
        cube["pixelsize"],
        cube["wcshdr"],
    )

    return None


def find_nearest_voronoibin(x, y, idx_outside, xNode, yNode):
    """
    Find the nearest Voronoi-bin for each spaxel that does not satisfy the minimum SNR threshold.
    
    Args:
    - x (array): x-coordinates of all spaxels
    - y (array): y-coordinates of all spaxels
    - idx_outside (array): indices of spaxels which do not satisfy the minimum SNR threshold
    - xNode (array): x-coordinates of the Voronoi bins
    - yNode (array): y-coordinates of the Voronoi bins
    
    Returns:
    - closest (array): array of indices representing the nearest Voronoi-bin for each spaxel
    """
    # Create an array of pixel coordinates
    pix_coords = np.column_stack((x[idx_outside], y[idx_outside]))
    # Create an array of bin coordinates
    bin_coords = np.column_stack((xNode, yNode))

    # Build a KDTree from the bin coordinates
    tree = cKDTree(bin_coords)
    # Query the nearest bin for each pixel
    closest = tree.query(pix_coords, k=1, workers=-1)[1]

    return closest


def save_table(
    config,
    x,
    y,
    signal,
    snr,
    binNum_new,
    ubins,
    xNode,
    yNode,
    sn,
    nPixels,
    pixelsize,
    wcshdr,
):
    """
    Save all relevant information about the Voronoi binning to disk. In
    particular, this allows to later match spaxels and their corresponding bins.

    Args:
        config (dict): Configuration settings.
        x (ndarray): X-coordinates of the spaxels.
        y (ndarray): Y-coordinates of the spaxels.
        signal (ndarray): Flux values of the spaxels.
        snr (ndarray): Signal-to-noise ratio values of the spaxels.
        binNum_new (ndarray): Array of bin IDs for each spaxel.
        ubins (ndarray): Unique bin IDs.
        xNode (ndarray): X-coordinates of the bin nodes.
        yNode (ndarray): Y-coordinates of the bin nodes.
        sn (ndarray): Signal-to-noise ratio values of the bins.
        nPixels (ndarray): Number of spaxels in each bin.
        pixelsize (float): Size of each pixel.
        wcshdr (Header): WCS header information.

    Returns:
        None
    """
    outfits_table = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_table.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_table.fits")
    xshape = x.shape
    yshape = y.shape
    printStatus.running("size of the input array=" + str(xshape) + str(yshape))
    # Expand data to spaxel level
    xNode_new = np.zeros(len(x))
    yNode_new = np.zeros(len(x))
    sn_new = np.zeros(len(x))
    nPixels_new = np.zeros(len(x))
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(binNum_new))[0]
        xNode_new[idx] = xNode[i]
        yNode_new[idx] = yNode[i]
        sn_new[idx] = sn[i]
        nPixels_new[idx] = nPixels[i]

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    # Table HDU with output data
    cols = []
    cols.append(fits.Column(name="ID", format="J", array=np.arange(len(x))))
    cols.append(fits.Column(name="BIN_ID", format="J", array=binNum_new))
    cols.append(fits.Column(name="X", format="D", array=x))
    cols.append(fits.Column(name="Y", format="D", array=y))
    cols.append(fits.Column(name="FLUX", format="D", array=signal))
    cols.append(fits.Column(name="SNR", format="D", array=snr))
    cols.append(fits.Column(name="XBIN", format="D", array=xNode_new))
    cols.append(fits.Column(name="YBIN", format="D", array=yNode_new))
    cols.append(fits.Column(name="SNRBIN", format="D", array=sn_new))
    cols.append(fits.Column(name="NSPAX", format="J", array=nPixels_new))

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "TABLE"

    # create empty imageHDU with wcs header info
    imghdu = fits.ImageHDU(data=None, header=wcshdr)

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, tbhdu, imghdu])
    HDUList.writeto(outfits_table, overwrite=True)
    fits.setval(outfits_table, "PIXSIZE", value=pixelsize)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_table.fits")
    logging.info("Wrote Voronoi table: " + outfits_table)
