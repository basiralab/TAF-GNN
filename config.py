import os

CURRENT_DIR = os.getcwd()

REAL_DATA_PATH          = os.path.join(CURRENT_DIR, "data/LMCI_AD_dataset")
SIMULATED_DATA_PATH     = os.path.join(CURRENT_DIR, "data/simulated_data")

ACC_RESULTS_DIR         = os.path.join(CURRENT_DIR, "results/")

# CBT
CBTS_DIR                = os.path.join(CURRENT_DIR, "cbts")
CBTS_DIR_SIMULATED_DATA = os.path.join(CBTS_DIR, "simulated")
CBTS_DIR_REAL_DATA      = os.path.join(CBTS_DIR, "real")

