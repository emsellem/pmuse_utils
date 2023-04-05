# -*- coding: utf-8 -*-

# -------------- General initialisation of muse quantities ---------
# This also serves to provide some useful structure for maps and dat
# EE - 2020. 
# v1.0 - Dec 2020
# ------------------------------------------------------------------

# -- Importing the usual packages. 
# Numpy, sys, copy, os
import numpy as np
import sys
from os.path import join as joinpath, isfile
import copy
import importlib
from astropy.io import fits as pyfits

# Importing the config parameters
from .phangs_muse_config import *

class Phangs_Muse_Info(object):
    """Main phangs class to gather some useful functions and info

    DR (str): data release number (e.g., 2.1)
    BB (str): WFI release number (e.g., 1.0)
    base_folder (str): folder top of tree
    """
    def __init__(self, **kwargs_phangs):

        # ============= FOLDERS =======================================
        # DR2 data folder
        print(f"INFO ------ we will be using data from DR {DR} --------")
        self.dict_phangs_keywords = define_phangs_keywords()
        for keyword in self.dict_phangs_keywords:
            setattr(self, keyword, kwargs_phangs.pop(keyword, self.dict_phangs_keywords[keyword]))

        self.dict_obs = {'WFI': ['WFI_BB', self.wfi_folder], 
                         'DUPONT': ['DUPONT_R', self.dp_folder]}
        
        # import sample and dictionaries
        pm_dict =  importlib.import_module(f'.dict.{dict_config_files[DR][0]}', package="pmuse_utils")
        psf_dict =  importlib.import_module(f'.dict.{dict_config_files[DR][1]}', package="pmuse_utils")

        for key in psf_dict.dict_locals:
            if key[0] != "_" and key != "dict_locals":
                setattr(self, key, getattr(psf_dict, key))
        for key in pm_dict.dict_locals:
            if key[0] != "_" and key != "dict_locals":
                setattr(self, key, getattr(pm_dict, key))

        # Phangs table
        self._version_phangs_table = "dummy"

    @property
    def nphangs(self):
        """Number of phangs targets
        """
        return len(self.phangs_sample)

    @property
    def nmuse(self):
        """Number of muse targets
        """
        return len(self.phangs_muse_sample)

    def _check_muse_targetname(self, targetname):
        """Check if target is in the muse sample
        """
        if targetname not in self.phangs_muse_sample:
            print(f"ERROR: {targetname} not in PHANGS MUSE sample [DR = {self.DR}]")
            return False
        return True

    def get_bbname(self, targetname, cut=False, size=5):
        """Get the reference image for a given galaxy
    
        Input
        -----
        targetname (str): name of the galaxy
        """
        if not self._check_muse_targetname(targetname): return ""

        if cut:
            add = f"_{size}x{size}"
        else:
            add = ""
        obs = self.dict_filter_for_alignment[targetname]
        return f"{self.dict_obs[obs][1]}{targetname}_Rc_flux_nosky{add}.fits"

    @property
    def phangs_table(self):
        version = self.version_phangs_table
        if self._version_phangs_table != version:
            phangs_tablename = joinpath(self.phangs_data_folder, 
                                        f"phangs_sample_table_{version}.fits")
            if isfile(phangs_tablename):
                self.phangs_tablename = phangs_tablename
                from astropy.table import Table
                self.version_phangs_table = copy.copy(self._version_phangs_table)
                self._phangs_table = Table.read(phangs_tablename)
            else:
                print(f"WARNING: file {phangs_tablename} does not exist - setting None")
                self._phangs_table = None
        return self._phangs_table

    @property
    def phangs_sample(self):
        return list(self.phangs_table['name'].data.astype(str))

    @property
    def muse_table(self):
        return self.phangs_table[self.phangs_table['survey_muse_status'] == 'released']

    def get_muse_pc_arcsec(self, targetname):
        dist = self.get_muse_param_1gal(targetname, 'dist')
        return dist * np.pi / 0.648

    def get_muse_param_1gal(self, targetname, param_name='dist'):
        """Get one param from the muse table and the value
        """
        if not self._check_muse_targetname(targetname): return None
        ind = np.argwhere(self.muse_table['name'].data.astype('str') == targetname.lower())[0][0]
        return self.muse_table[param_name][ind]

    def get_muse_param(self, param_name='dist'):
        """Get one param from the muse table and return dictionary
        """
        dict_param = {}
        for i in range(self.nmuse):
            dict_param[self.muse_table['name'][i].upper()] = self.muse_table[param_name][i]
        return dict_param

    def get_muse_fov(self, spaxel=0.2, unit="asec"):
        """Get the field of view from the dictionary of number
        of spaxels. Transform in kpc and asec for further usage.
        """
        if not hasattr(self, 'dict_muse_npix'):
            self._get_all_muse_npix()
        dist = self.get_muse_param('dist')
        dict_fov = {}
        ngal = len(dist)
        grid_asec = np.zeros((2, ngal))
        grid_kpc = np.zeros_like(grid_asec)
        for i, key in enumerate(dist):
            d = np.float(dist[key])
            fackpc = np.pi * d / 0.648 / 1000.
            nf = np.array(self.dict_muse_npix[key])
            asec = nf * spaxel
            kpc = asec * fackpc
            grid_asec[:,i] = [asec[0], asec[1]]
            grid_kpc[:,i] = [kpc[0], kpc[1]]
            dict_fov[key] = [asec, kpc]
    
        if unit == "asec" : return grid_asec
        elif unit == "kpc" : return grid_kpc
        else: return grid_asec

    def get_summary_table(surveys=['HST', 'HI', 'ASTROSAT', 'MUSE', 'ALMA']): 
        list_ok = ['released', 'observed_not_released']
    
        # select sample
        name = self.phangs_table['name']
        ra = self.phangs_table['orient_ra']
        dec = self.phangs_table['orient_dec']
        log_m = self.phangs_table['props_mstar']
        log_sfr = self.phangs_table['props_sfr']
    
        dict_surveys = {}
        for survey in surveys:
            test_ok = [self.phangs_table['survey_{survey}_status'] == ok for ok in list_ok]
            dict_surveys[survey] = np.argwhere(np.logical_or.reduce(test_ok))
    
        return name, ra, dec, log_m, log_sfr, dict_surveys

    def get_sdss(self, z_limit=0.05, folder=None, name='GSWLC-X2.dat.gz'):
        from astropy.io import ascii
        if folder is None: folder = self.phangs_data_folder
        tbl = ascii.read(joinpath(folder, name))
        log_m, log_sfr = tbl['col10'], tbl['col12']
        if z_limit is not None:
            z = tbl['col8']
            ind = np.where(z < z_limit)
        else:
            ind = np.where(log_sfr > -np.inf)
        return log_m[ind], log_sfr[ind], z_limit

    def get_muse_footprint(self, targetname):
        if not self._check_muse_targetname(targetname): return
        return muse_footprint(targetname, self.mask_folder)

    def get_map_suffix(self, targetname, jwst="anchored", version="native"):
        if version == "native":
            return "", "_MAPS", jwst
        elif version == "copt":
            psf = self.dict_psf_copt[targetname]
            print(f"Copt version. For galaxy {targetname}, PSF is {psf}asec")
            return f"_copt_{psf}asec", f"-{psf}asec_MAPS", jwst

    def get_muse_npix(self, targetname, ref_images_dict=default_white, 
                     version="native", verbose=False, **kwargs):
        """Just getting the shape of the reference images
        """
        suffix, mapsuffix, _ = self.get_map_suffix(targetname, version)

        images_folder = self.dict_phangs_keywords[f"{version}_folders"][0]
        ref_imaname = (f"{images_folder}{targetname}_"
                     f"IMAGE_FOV_{ref_images_dict['white']}_WCS_Pall_mad{suffix}.fits")
        if not isfile(ref_imaname):
            if verbose:
                print(f"WARNING: ref image {ref_imaname} not found for {targetname}")
            return np.empty((0,0))

        return {targetname: np.array(pyfits.getdata(ref_imaname).shape)[::-1]}

    def _get_all_muse_npix(self, ref_images_dict=default_white, 
                           version="native", verbose=False):
        """Just getting the shapes of all reference images
        """
        self.dict_muse_npix = {}
        for targetname in self.phangs_muse_sample:
            self.dict_muse_npix.update(self.get_muse_npix(targetname, ref_images_dict, 
                                 version, verbose))


# Footprints
class Muse_Footprint(object):
    """Class to use the footprint of MUSE pointings
    It will be using the white_footprint.txt files for each pointing, which were
    derived using footprintfinder.py -d -t 1 {ima}
    """
    def __init__(self, targetname, mask_folder=""):
        self.target = targetname
        self.mask_folder = mask_folder

    def filename(self, pointing=1):
        if pointing not in self.pointings:
            print(f"File {name} not found in existing list. Please check folder")
            return None
        name = f"{self.mask_folder}{self.target}_mask_IMAGE_FOV_P{pointing:02d}_white_footprint.txt"
        return name

    @property
    def pointings(self):
        import re
        pointings = []
        for name in self._filenames:
            numpoint = np.int(re.findall(r'P\d\d', name)[0][1:])
            pointings.append(numpoint)
        return sorted(pointings)

    @property
    def _filenames(self, verbose=False):
        import glob
        nf = sorted(glob.glob(f"{self.mask_folder}{self.target}_mask_"
                            f"IMAGE_FOV_P??_white_footprint.txt"))
        if verbose:
            print(f"Found {len(nf)} pointings for {self.target}")
        return nf

    def get_footprint(self, pointing=1):
        # Opening the file
        poly = open(self.filename(pointing=pointing))
        # Filtering the first lines
        lines = poly.readlines()
        clines = [l for l in lines if l[0] != '#']

        # Creating the structures
        poly_pix = np.zeros((len(clines), 2))
        poly_wcs = np.zeros_like(poly_pix)
        k = 0
        for i in range(len(clines)):
            sl = clines[i].split()
            if len(sl) > 4:
                continue
            poly_pix[k, 0] = np.float(sl[0])
            poly_pix[k, 1] = np.float(sl[1])
            poly_wcs[k, 0] = np.float(sl[2])
            poly_wcs[k, 1] = np.float(sl[3])
            k += 1
        self.poly_pix = poly_pix[:k]
        self.poly_wcs = poly_wcs[:k]

        self.center_pix = [np.float(clines[1].split()[6]), np.float(clines[1].split()[7])]
        self.center_wcs = [np.float(clines[1].split()[8]), np.float(clines[1].split()[9])]

