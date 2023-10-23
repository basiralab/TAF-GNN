import torch
import numpy as np
import random
import argparse
from sklearn.model_selection import KFold
import copy
import timeit
from data_utils import timer
import os

from dataset import *
from plot import *
from model_rbgm import GNN_1
from debugger import FEDDebugger
import config
from losses import *
from alignment import *

#############################################################################################################

# Setting the running device

manualSeed = 1

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on GPU')
    # if you are using GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

else:
    device = torch.device("cpu")
    print('running on CPU')

#############################################################################################################

# Global variables

MAE_LOSS_CALCULATE = None
debugger = None


#############################################################################################################

def get_args():
    ''' Command-line argument getter
    '''

    parser = argparse.ArgumentParser(description='Args for graph prediction')

    # changeable parameters
    parser.add_argument('-mode',            type=str, default="4D-FED-GNN",     help='training technique')
    parser.add_argument('-num_epochs',      type=int, default=25,               help='number of epochs')
    parser.add_argument('-num_folds',       type=int, default=3,                help='cv number')
    parser.add_argument('-eval_mode',       type=str, default="intra-domain",   help="inter-domain or intra-domain evaluation")
    parser.add_argument('-mix_amount',      type=int, default=15,               help="taking mix_amount data from each hospital for inter-domain evaluation")
    parser.add_argument('-exp',             type=int, default=1,                help='Which experiment are you running')
    parser.add_argument('-C',               type=int, default=5,                help='number of round before averaging')
    parser.add_argument('-D',               type=int, default=4,                help='number of rounds before daisy chain')
    parser.add_argument('-simulated_data',  type=int, default=0,                help='use simulated data or real data')
    parser.add_argument('-alignment',       type=str, default="single",         help="alignment type") # "single", "prior", "statistical"

    # single aligner parameters
    parser.add_argument('-single_aligner_num_epochs', type=int,   default=275,  help='number of epochs')

    # 
    parser.add_argument('-path',            type=str, default="",               help='path to save data')

    # hyperparameters and other parameters
    parser.add_argument('--num_regions', type=int, default=35, help='Number of regions')
    parser.add_argument('--num_timepoints', type=int, default=2, help='Number of timepoints')
    parser.add_argument('--lr_g', type=float, default=0.01, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate')
    parser.add_argument('--decay', type=float, default=0.0, help='Weight Decay')
    parser.add_argument('--batch_num', type=int, default=1, help='batch number')
    parser.add_argument('--tp_coeff', type=float, default=0.0, help='Coefficient of topology loss')
    parser.add_argument('--g_coeff', type=float, default=2.0, help='Coefficient of adversarial loss')
    parser.add_argument('--i_coeff', type=float, default=2.0, help='Coefficient of identity loss')
    parser.add_argument('--kl_coeff', type=float, default=0.001, help='Coefficient of KL loss')
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--tp_coef', type=float, default=10, help="KL Loss Coefficient")
    
    args, _ = parser.parse_known_args()
    return args


def get_folds(length, num_folds):
    """
    Arguments:
        length: number of subjects
        num_folds: number of folds

    This function returns a list of subjects for each fold (list of lists)
    """
    indexes = list(range(length))
    random.shuffle(indexes)
    n = length // num_folds

    folds = []
    for fold in range(num_folds):
        if fold == num_folds - 1:
            folds.append(indexes[fold * n:])
        else:
            folds.append(indexes[fold * n: (fold * n) + n])

    return folds

#############################################################################################################

global_model = GNN_1().to(device)

class Hospital():
    def __init__(self, args, dataset):
        """
        Hospital object contains a GNN and an optimizer for each timepoint
        Hospital object can update GNN-layer wise weights of its GNNs
        """
        self.dataset = dataset
        self.folds = None

        # if args.eval_mode == 'intra-domain':
        #     # if the mode is intra-domain, we are using all data in the dataset
        #     self.folds = get_folds(dataset.shape[0], args.num_folds)
        # elif args.eval_mode == 'inter-domain':
        #     # if the mode is inter-domain, we are using not all, but (dataset.shape[0]-args.mix_amount) subjects
        #     # if dataset has 100 subjects, and mix_amount is 30, then we are just using 70 subjects to train on
        #     self.folds = get_folds(dataset.shape[0] - args.mix_amount, args.num_folds)

        self.folds = get_folds(dataset.shape[0], args.num_folds)

        self.models = []
        self.optimizers = []
        for i in range(args.num_timepoints - 1):
            self.models.append(GNN_1().to(device))
            self.optimizers.append(torch.optim.Adam(self.models[i].parameters(), lr=args.lr))

    def update_hospital(self, main_model):
        for i in range(len(self.models)):
            self.models[i].load_state_dict(main_model.models[i].state_dict())

    def get_data_from_folds(self, fold_array):
        data = []
        for f in fold_array:
            indices = self.folds[f]
            for index in indices:
                data.append(self.dataset[index])

        data = torch.stack(data)
        
        return data.clone()

#############################################################################################################

def train(args, dataset, num_of_hospitals):
    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=1901)
    f = 0
    indexes = range(args.num_folds)
    for train, test in kfold.split(indexes):       
        if f == 1:
            break

        print(f'------------------------------------Fold [{f + 1}/{args.num_folds}]-----------------------------------------')
        tic0 = timeit.default_timer()        

        # initialize error lists
        mae_list, tp_list, tot_list = list(), list(), list()

        # initialize hospitals and train data for each hospital
        hospitals = []
        train_data_for_each_hospital = [] # 4 tane x,2,35,35
        test_data_for_each_hospital = []  # 4 tane x,2,35,35
        for h in range(num_of_hospitals):
            hospitals.append(Hospital(args, dataset[h]))
            train_data_for_each_hospital.append(hospitals[h].get_data_from_folds(train))
            test_data_for_each_hospital.append(hospitals[h].get_data_from_folds(test))

        train_data_for_each_hospital, test_data_for_each_hospital = align(args, train_data_for_each_hospital, test_data_for_each_hospital, f+1)

        # start training
        for t in range(1, args.num_timepoints):
            print(f"---------------------------------- timepoint {t} ------------------------------------------")
            for epoch in range(args.num_epochs):
                print(f'\n\tEpoch [{epoch + 1}/{args.num_epochs}]')
                
                tot_mae, tot, tp = 0.0, 0.0, 0.0 # error initiliaze

                for h_i in range(len(hospitals)):
                    h = hospitals[h_i]

                    train_data = train_data_for_each_hospital[h_i]

                    hospitals[h_i], tot_l, tp_l, mae_l = train_one_epoch(args, h, train_data, f, t, h_i)

                    tot_mae += mae_l
                    tot += tot_l
                    tp += tp_l

                    print(f'Hospital [{h_i + 1}/{len(hospitals)}]')
                    print(f'[Train] Loss T' + str(t) + f': {mae_l:.5f}',
                          f'[Train] TP Loss T' + str(t) + f': {tp_l:.5f} ',
                          f'[Train] Total Loss T' + str(t) + f': {tot_l:.5f} ')

                # updating the models
                if epoch != args.num_epochs - 1 or epoch != 0:
                    if epoch % args.C == 0 and args.mode != "4D-GNN":
                        # for 4D-FED-GNN and 4D-FED-GNN+
                        if args.eval_mode == "inter-domain":
                            save_global_model(hospitals, t, epoch, f)
                        hospitals = update_main_by_average(hospitals, t)
                    elif epoch % args.D == 0 and args.mode == "4D-FED-GNN+":
                        hospitals = exchange_models(hospitals, t)

                mae_list.append(tot_mae)
                tot_list.append(tot)
                tp_list.append(tp)

            # when epochs are ended
            print("\n")
            plot_title = f"model{t}_fold{str(f)}"
            plot(args, "TotalLoss",  plot_title, tot_list)
            plot(args, "MAE",        plot_title, mae_list)
            plot(args, "TP",         plot_title, tp_list)

            mae_list.clear()
            tot_list.clear()
            tp_list.clear()


        # when all models are trained, then comes to evaluation part
        if args.eval_mode == "intra-domain":
            for h_i in range(len(hospitals)):
                h = hospitals[h_i]
                # test_data = h.get_data_from_folds(test)
                test_data = test_data_for_each_hospital[h_i]

                validate(args, hospitals[h_i], test_data, h_i, f)

        elif args.eval_mode == "inter-domain":
            # clipped all is an array of datasets taken from each hospital
            # clipped_data = [] 
            # for d in dataset:
            #     clipped_data.append(d[-args.mix_amount:])
            clipped_data = test_data_for_each_hospital

            # firstly concatenating all taken datasets into 1-dimension array and then shufffling randomly
            global_mixed_test_data = torch.cat(clipped_data, dim=0)
            # global_mixed_test_data = global_mixed_test_data[torch.randperm(global_mixed_test_data.size(0))]
            for h_i in range(len(hospitals)):
                h = hospitals[h_i]
                validate(args, hospitals[h_i], global_mixed_test_data, h_i, f)
            
            validate_on_global_model(args, global_mixed_test_data, f)

        # saving all models after each epoch, just for the first fold
        if f == 0:
            for h_i, hospital in enumerate(hospitals):
                for m_i, model in enumerate(hospital.models):
                    torch.save(model.state_dict(), os.path.join(args.path, "models", f"EXP{str(args.exp)}_{args.eval_mode.upper()}_{args.mode}___hospital{str(h_i+1)}_model{m_i+1}_fold{str(f+1)}.model"))
                    if args.eval_mode == "inter-domain":
                        torch.save(global_model.state_dict(), os.path.join(args.path, "models", f"EXP{str(args.exp)}_{args.eval_mode.upper()}_{args.mode}___globalmodel_fold{str(f+1)}.model"))

        tic1 = timeit.default_timer()
        timer(tic0,tic1)
        f += 1

def train_one_epoch(args, hospital, train_data, fold, t, h_i):
    """
    Arguments:
        hospital: the currently training hospital
        train_data: local data of the hospital
        index: [hospital_id, timepoint]

    Returns:
        hospital, total loss, topological loss, mae loss
    """
    mael = torch.nn.L1Loss().to(device)
    tp = torch.nn.MSELoss().to(device)

    total_step = len(train_data)
    train_loss = 0.0
    tp_loss, tr_loss = 0.0, 0.0

    cur_id = t - 1  # id of the model that will be trained

    # training one epoch
    hospital.models[cur_id].train()

    for i, data in enumerate(train_data):
        data = data.to(device)
        hospital.optimizers[cur_id].zero_grad()
        out = hospital.models[cur_id](data[cur_id])

        # Topological Loss
        tp_l = tp(out.sum(dim=-1), data[cur_id + 1].sum(dim=-1))
        tp_loss += tp_l.item()
        # MAE Loss
        loss = mael(out, data[cur_id + 1])
        train_loss += loss.item()
        # Topological Loss
        self_tp_l = tp(out.sum(dim=-1), data[cur_id].sum(dim=-1))
        tp_loss += self_tp_l.item()
        # MAE Loss
        self_loss = mael(out, data[cur_id])
        train_loss += self_loss.item()
        # total loss
        total_loss = (loss + self_loss + args.tp_coef * tp_l + args.tp_coef * self_tp_l) / 2
        tr_loss += total_loss.item()

        total_loss.backward()
        hospital.optimizers[cur_id].step()

    tot = tr_loss / total_step
    tp_l = tp_loss / total_step
    mae = train_loss / total_step

    return hospital, tot, tp_l, mae


#############################################################################################################

def validate(args, hospital, test_data, h_i, f):
    """
        Output:
            plotting of each predicted testing brain graph, also saved as a numpy file
            average MAE of predicted brain graphs
    """
    mael = torch.nn.L1Loss().to(device)

    val_hos = len(test_data)

    hloss = []
    for k in range(len(hospital.models)):
        hospital.models[k].eval()
        hloss.append(0)

    outs = torch.zeros((100, 1, 35, 35))

    with torch.no_grad():
        for i, data in enumerate(test_data):
            data = data.to(device)
            out_1 = data[0]
            for k, model in enumerate(hospital.models):
                temp = model.rnn[0].hidden_state
                out_1 = model(out_1)
                model.rnn[0].hidden_state = temp

                outs[i, 0, :, :] = out_1
                cur_loss = torch.mean(torch.abs(out_1 - data[k + 1]))
        
                print(f"{cur_loss}", end=',')
                hloss[k] += cur_loss

    for k in range(1, args.num_timepoints):
        loss = hloss[k - 1] / val_hos
        debugger.save_eval_results(h_i+1, k, f, loss)
        print('[Val]: MAE Loss Model' + str(k) + f': {loss:.5f}', sep=' ', end='\n', flush=True)
        MAE_LOSS_CALCULATE[(args.num_timepoints - 1) * h_i + (k - 1)][f] = loss

    print(" ")


def validate_on_global_model(args, test_data, f):
    mael = torch.nn.L1Loss().to(device)

    val_hos = len(test_data)

    hloss = 0
    global_model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data):
            data = data.to(device)
            out_1 = data[0]

            temp = global_model.rnn[0].hidden_state
            out_1 = global_model(out_1)
            global_model.rnn[0].hidden_state = temp

            cur_loss = mael(out_1, data[1])
            print(f"{cur_loss} loss on {i}")
            hloss += cur_loss
    
        loss = hloss / val_hos
        debugger.save_eval_results("GLOBAL", -1, f, loss)
        print('[Val]: MAE Loss Global Model' + f': {loss:.5f}', sep=' ', end='\n', flush=True)

    print(" ")

#############################################################################################################

def exchange_models(hospitals, t):
    """
        This function exchanges GNNs of hospitals at timepoint t with each other
    """
    pre_model = None
    for i, hospital in enumerate(hospitals):
        next_model = copy.deepcopy(hospitals[i].models[t - 1].state_dict())

        if not pre_model is None:
            hospitals[i].models[t - 1].load_state_dict(pre_model)

        pre_model = copy.deepcopy(next_model)

        if i == 0:
            hospitals[i].models[t - 1].load_state_dict(copy.deepcopy(hospitals[-1].models[t - 1].state_dict()))

    return hospitals

def update_main_by_average(hospitals, t):
    """
        This function takes the GNN-layer weights of the GNN at timepoint t and computes the global model by averaging,
        then broadcats the weights to the hospitals (updates each GNN with the global model)
    """
    for i, hospital in enumerate(hospitals):
        target_state_dict = copy.deepcopy(hospital.models[t - 1].state_dict())
        model_list = []
        for k, h in enumerate(hospitals):
            if k != i:
                model_list.append(h.models[t - 1])

        mux = 1 / (len(model_list) + 1)
        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].data = target_state_dict[key].data.clone() * mux
                for model in model_list:
                    state_dict = copy.deepcopy(model.state_dict())
                    target_state_dict[key].data += mux * state_dict[key].data.clone()

        hospitals[i].models[t - 1].load_state_dict(target_state_dict)

    return hospitals


def save_global_model(hospitals, t, epoch, fold):
    target_state_dict = copy.deepcopy(hospitals[0].models[t - 1].state_dict())
    model_list = []
    for k, h in enumerate(hospitals):
        if k != 0:
            model_list.append(h.models[t - 1])

    mux = 1 / (len(model_list) + 1)
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data = target_state_dict[key].data.clone() * mux
            for model in model_list:
                state_dict = copy.deepcopy(model.state_dict())
                target_state_dict[key].data += mux * state_dict[key].data.clone()

    global_model.load_state_dict(target_state_dict)
    


##############################################################################################################


if __name__ == "__main__":
    # pre-works
    args = get_args()
    if args.alignment not in ["", "single", "prior", "statistical"]:
        raise Exception(f"Aligner type {args.alignment} is not correct.")

    print(args)
    debugger = FEDDebugger(args)
    print("Results will be saved to: ", debugger.PATH_TO_SAVE_RESULTS)
    debugger.save_headers()


    # getting data
    if args.simulated_data == 0:
        all_views = dataset_builder(config.REAL_DATA_PATH) # [67, 2, 35, 35, 4]
    elif args.simulated_data == 1:
        all_views = torch.from_numpy(np.load(os.path.join(config.SIMULATED_DATA_PATH, "example_sim_data.npy"))) # [200, 4, 35, 35, 4]
        all_views = all_views.to(dtype=torch.float32)

    num_of_hospitals = all_views.shape[4] # number of views equals to number of hospitals

    dataset = []
    for i in range(num_of_hospitals):
        dataset.append(all_views[torch.randperm(all_views.shape[0]), :, :, :, i])
        print(f"Dataset {i+1} size: ", dataset[i].size())    


    MAE_LOSS_CALCULATE = np.zeros((((args.num_timepoints - 1) * num_of_hospitals), args.num_folds))

    train(args, dataset, num_of_hospitals)

    debugger.save_loss_array(MAE_LOSS_CALCULATE)
    print("MAE Loss Array:\n", MAE_LOSS_CALCULATE)
    print("Means for each hospital and timepoint:\n", np.mean(MAE_LOSS_CALCULATE, axis=1))
    print("Results are saved to: ", debugger.PATH_TO_SAVE_RESULTS)