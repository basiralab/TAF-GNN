import os
import numpy as np
from uuid import uuid4
from datetime import datetime

import config

class FEDDebugger():
    def __init__(self, args):
        self.args = args
        
        if args.path == "":
            self.unique_run_id = str(uuid4())
            self.unique_path = os.path.join(config.ACC_RESULTS_DIR, self.unique_run_id)
            os.mkdir(self.unique_path)
            args.path = self.unique_path

            os.mkdir(os.path.join(args.path, "models"))               
            os.mkdir(os.path.join(args.path, "losses"))    
            os.mkdir(os.path.join(args.path, f"{args.alignment}_alignment"))
            for f in range(1, args.num_folds + 1):
                os.mkdir(os.path.join(args.path, f"{args.alignment}_alignment", f"fold{f}"))    

            
        self.FILENAME_FOR_RESULTS = f"{self.args.simulated_data}SIM_({self.args.alignment})PA_EXP{str(self.args.exp)}_{self.args.eval_mode.upper()}_{self.args.mode}.txt"
        self.PATH_TO_SAVE_RESULTS = os.path.join(args.path, self.FILENAME_FOR_RESULTS)

    def save_headers(self):
        with open(self.PATH_TO_SAVE_RESULTS, 'a') as txt_file:
            txt_file.write(f'\t\t{self.args.eval_mode.upper()} evaluation at {datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\n')
            txt_file.write("\t\tParameters: " + str(self.args) + "\n\n")

    def save_loss_array(self, all_mae_losses):
        with open(self.PATH_TO_SAVE_RESULTS, 'a') as txt_file:
            txt_file.write('\n\n\t\tResults:\n')
            np.savetxt(txt_file, all_mae_losses, fmt='%1.4f')
            txt_file.write("\n\n\t\tAverage: \n")
            np.savetxt(txt_file, np.mean(all_mae_losses, axis=1), fmt='%1.4f')

    def save_eval_results(self, h_i, t, f, loss):
        with open(self.PATH_TO_SAVE_RESULTS, 'a') as txt_file:
            txt_file.write(f'[Val]: MAE Loss Model - Hospital {h_i}, timepoint {t}, fold {f}' + f': {loss:.5f}' + '\n')



class AlignerDebugger():
    def __init__(self, args, fold):
        file_name = f"fold{fold}_{args.alignment}_aligner_log.txt"

        self.log_file_path = os.path.join(args.path, f"{args.alignment}_alignment", file_name)

        notes = ""
        with open(self.log_file_path, 'a') as txt_file:
            txt_file.write(f'\t\tDate: {datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\n')
            txt_file.write("\t\tParameters: " + str(args) + "\n\n")
            txt_file.write("\t\t" + notes + "\n")

    def write_text(self, text):
        with open(self.log_file_path, 'a') as txt_file:
            txt_file.write(text)