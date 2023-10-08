import os

CURRENT_DIR = os.getcwd()

REAL_DATA_PATH          = os.path.join(CURRENT_DIR, "data/LMCI_AD/RH_dataset")
SIMULATED_DATA_PATH     = os.path.join(CURRENT_DIR, "data/simulated_data")

ACC_RESULTS_DIR         = os.path.join(CURRENT_DIR, "results/evaluation_acc")
LOSS_RESULTS_DIR        = os.path.join(CURRENT_DIR, "results/training_loss")
PLOT_DIR                = os.path.join(CURRENT_DIR, "results/plot")
WEIGHTS_DIR             = os.path.join(CURRENT_DIR, "results/weights")

# CBT
CBT_OBTAINED_WAY = "MGN-Net"
CBTS_DIR                = os.path.join(CURRENT_DIR, "cbts", CBT_OBTAINED_WAY)
CBTS_DIR_SIMULATED_DATA = os.path.join(CBTS_DIR, "simulated")
CBTS_DIR_REAL_DATA      = os.path.join(CBTS_DIR, "real")

# Aligner
ALIGNER_WEIGHTS_DIR                  = os.path.join(CURRENT_DIR, "single_aligner", "alignment_weights", CBT_OBTAINED_WAY)
ALIGNER_WEIGHTS_DIR_SIMULATED_DATA   = os.path.join(ALIGNER_WEIGHTS_DIR, "simulated")
ALIGNER_WEIGHTS_DIR_REAL_DATA        = os.path.join(ALIGNER_WEIGHTS_DIR, "real")

# Temporary aligner weights
TMP_ALIGNER_WEIGHTS_DIR                 = os.path.join(CURRENT_DIR, "temp", "single_aligner_results")
TMP_ALIGNER_WEIGHTS_DIR_SIMULATED_DATA  = os.path.join(TMP_ALIGNER_WEIGHTS_DIR, "simulated")
TMP_ALIGNER_WEIGHTS_DIR_REAL_DATA       = os.path.join(TMP_ALIGNER_WEIGHTS_DIR, "real")

