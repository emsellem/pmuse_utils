# -*- coding: utf-8 -*-

# ============== USER DEFINED PARAM =========================
# Version you wish to use for the input Data/Table
BB = "1.0"
JW = "v0p7p2"
DR = "2.1"
version_phangs_table = "v1p6"

# Where the init py script is on your computer
base_folder = "/home/science/PHANGS/"
muse_subfolder =  "MUSE/"
jwst_subfolder =  "JWST/"
bb_subfolder = "BB_images/"
# Where is the "Info" folder
pyscript_folder = f"{base_folder}{muse_subfolder}pymuse_scripts/"
dr_folder = f"{base_folder}{muse_subfolder}data/DR{DR}/"

# Function setting up the folder dictionary - you can change
# the values in the dictionary according to your structure 
# but pls do not change the function itself
def define_phangs_keywords(base_folder=base_folder,
                           DR=DR, BB=BB, JW=JW,
                           version_phangs_table=version_phangs_table):
    """Define the keywords needed for phangs
    """

    dict_phangs_keywords = {
            # From base folder, here is the set of folders used in this script
            "BB": BB,
            "DR": DR,
            "version_phangs_table": version_phangs_table,
            "phangs_data_folder" : f"{base_folder}PhangsTables/",
            "bb_folder" : f"{base_folder}{bb_subfolder}/DR{BB}/",
            "wfi_folder" : f"{base_folder}{bb_subfolder}/DR{BB}/WFI_images/",
            "dp_folder" : f"{base_folder}{bb_subfolder}/DR{BB}/DuPont_images/",
            "muse_folder": f"{base_folder}{muse_subfolder}",
            "jwst_folder": f"{base_folder}{jwst_subfolder}/{JW}/",
            "dr_folder" : dr_folder,
            "mask_folder" : f"{base_folder}{muse_subfolder}data/DR{DR}/masks/",
            "muse_info_folder" : f"{pyscript_folder}Info/DR{DR}/",
            "native_folders": [f"{dr_folder}native_IMAGES/", f"{dr_folder}native_MAPS/"],
            "copt_folders": [f"{dr_folder}copt_IMAGES/", f"{dr_folder}copt_MAPS/"]
            }
    return dict_phangs_keywords

# ============= MAPS  =======================================
# For colours
default_rgb_bands = {'R': "SDSS_i", 'G': "SDSS_r", 'B': "SDSS_g"}
jwst_lambda = [300, 335, 360, 770, 1000, 1130, 2100]
default_jwst_bands = {}
for l in jwst_lambda:
    default_jwst_bands[l] = f"F{l}W"
default_white = {'white': "white"}

default_colormap = "viridis"
def_perc = [2, 99]

# Rule for maps (cmap and stretch)
# Colors - please install it, cmocean is very nice!
# Just install it via:
#     pip install cmocean
import cmocean

# Using labels - first is stretch, second is color map, last is centred cuts or not
maps_rules_dict = {"V_STARS": [cmocean.cm.balance, 'linear', False, def_perc, True],
                   "SIGMA_STARS": [cmocean.cm.thermal, 'linear', False, [5, 97], False],
                   "AGE_LW": ['Greys_r', 'linear', False, [1, 99], False],
                   "AGE_MW": ['Greys_r', 'linear', False, def_perc, False],
                   "Z_LW": ['Greys_r', 'linear', False, def_perc, False],
                   "Z_MW": ['Greys_r', 'linear', False, def_perc, False]}

# Also keywords work - beware, this is not robust and it needs to be exclusive
maps_keyw_dict = {"_VEL": [cmocean.cm.balance, 'linear', False, def_perc, True],
                  "_SIGMA": [cmocean.cm.thermal, 'linear', False, [5, 99], False],
                  "_FLUX": [default_colormap, 'sqrt', False, [5, 99], False],
                  "WFI": [default_colormap, 'asinh', False, [2, 99.5], False]
                  }

dict_config_files = {'2.1': ["phangsdictionaries_v19", "psf_dictionary_v6"]}
# ============== END USER DEFINED PARAM =====================
