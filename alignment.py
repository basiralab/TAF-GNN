import torch
import os
from model_aligners import *
from losses import *
from preprocess import *
import config
from uuid import uuid4
from datetime import datetime
from debugger import *
from plot import *

import warnings
warnings.filterwarnings("ignore")


#########################################################################################

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


#########################################################################################


def align(args, train_data, test_data, fold):
    if args.alignment != "":
        if args.alignment == "single":
            aligned_train_data, aligned_test_data = single_aligner_alignment(args, train_data, test_data, fold)
        elif args.alignment == "statistical":
            aligned_train_data, aligned_test_data = statistical_alignment(args, train_data, test_data)
        elif args.alignment == "prior":
            aligned_train_data, aligned_test_data = prior_alignment(args, train_data, test_data)

        return aligned_train_data, aligned_test_data
    
    return train_data, test_data

#########################################################################################


def single_aligner_alignment(args, dataset, test_data, fold):

    debugger = AlignerDebugger(args, fold)

    train_datasets = convert_matrices_to_with_features(dataset)
    test_datasets = convert_matrices_to_with_features(test_data)

    if args.simulated_data == 0:
        cbt_path = config.CBTS_DIR_REAL_DATA
    elif args.simulated_data == 1:
        cbt_path = config.CBTS_DIR_SIMULATED_DATA

    # dataset properties
    N_views = len(dataset)
    N_timepoints = dataset[0].shape[1]

    # getting saved cbts
    all_cbts = []
    for t in range(1, N_timepoints+1):
        cbt = np.load(os.path.join(cbt_path, f"t{t}_cbt_1.npy"))
        all_cbts.append(cbt)
        # plot_matrix(cbt, f"CBT of timepoint {t}")

    indices_i, indices_j = np.triu_indices(35, 1)
    counter = 0
    train_aligned = []
    test_aligned = []
    for i in range(N_views):
        for t in range(N_timepoints):
            torch.cuda.empty_cache()

            print(f"view {i+1}, timepoint {t+1}")
            target = np.expand_dims(all_cbts[t][indices_i, indices_j], axis=0) # [1, 595]

            train_cur_data = train_datasets[counter]

            single_aligner = Aligner()
            single_aligner = single_aligner.to(device)
            optimizer = torch.optim.AdamW(single_aligner.parameters(), lr=0.025, betas=(0.5, 0.999))

            train_single_aligner(args, single_aligner, optimizer, train_cur_data, target, i, t, fold, debugger)



            # use saved models and aligning training and testing data
            print(f"Aligning view {i+1}, timepoint {t+1}")

            single_aligner_filepath = os.path.join(args.path, "single_alignment", f"fold{fold}", f"fold{fold}_view{i+1}_t{t+1}_single_aligner.model")
            single_aligner.load_state_dict(torch.load(single_aligner_filepath))

            # align training data
            tmp_train_aligned = []
            X_casted_training_cur_data = cast_data_vector_RH(train_cur_data)
            for data_source in X_casted_training_cur_data:
                data_source = data_source.to(device)
                A_output = single_aligner(data_source)

                tmp_train_aligned.append(A_output)

            tmp_train_aligned = torch.stack(tmp_train_aligned)
            train_aligned.append(tmp_train_aligned.detach().cpu())

            # align testing data
            test_cur_data = test_datasets[counter]

            tmp_test_aligned = []
            X_casted_test_cur_data = cast_data_vector_RH(test_cur_data)
            for data_source in X_casted_test_cur_data:
                data_source = data_source.to(device)
                A_output = single_aligner(data_source)

                tmp_test_aligned.append(A_output)

            tmp_test_aligned = torch.stack(tmp_test_aligned)
            test_aligned.append(tmp_test_aligned.detach().cpu())


            counter += 1


    train_res = []
    test_res = []
    for i in range(N_views):
        tmp_train = torch.zeros((train_aligned[0].shape[0], N_timepoints, train_aligned[0].shape[1], train_aligned[0].shape[2]))
        tmp_test = torch.zeros((test_aligned[0].shape[0], N_timepoints, test_aligned[0].shape[1], test_aligned[0].shape[2]))
        for t in range(N_timepoints):
            tmp_train[:, t, :, :] = train_aligned[i*N_timepoints+t]
            tmp_test[:, t, :, :] = test_aligned[i*N_timepoints+t]
            
        train_res.append(tmp_train) 
        test_res.append(tmp_test) 

    return train_res, test_res


def train_single_aligner(args, single_aligner, optimizer, X_train_source, X_train_target, view_num, timepoint_num, fold, debugger):
    X_casted_train_source = cast_data_vector_RH(X_train_source)
    X_casted_train_target = cast_data_vector_FC(X_train_target)

    target = X_casted_train_target[0].edge_attr.view(N_SOURCE_NODES, N_SOURCE_NODES)
    target = target.to(device)

    single_aligner.train()

    aligner_losses = []

    for epoch in range(args.single_aligner_num_epochs):
        aligner_loss = []
        for data_source in X_casted_train_source:
            data_source = data_source.to(device)
            
            A_output = single_aligner(data_source)

            kl_loss = alignment_loss(target, A_output)
            aligner_loss.append(kl_loss)

        optimizer.zero_grad()
        aligner_loss = torch.mean(torch.stack(aligner_loss))
        aligner_loss.backward(retain_graph=True)
        optimizer.step()

        aligner_losses.append(aligner_loss.cpu().detach())

        loss_string = "[Epoch: %d]| [Al loss: %f]|" % (epoch, aligner_loss)
        print(loss_string)
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            debugger.write_text(loss_string + "\n")

    plot_path = os.path.join(args.path, "single_alignment", f"fold{fold}")
    plot_aligner_loss(aligner_losses, fold, view_num, timepoint_num, plot_path)

    single_aligner_filepath = os.path.join(args.path, "single_alignment", f"fold{fold}", f"fold{fold}_view{view_num+1}_t{timepoint_num+1}_single_aligner.model")
    torch.save(single_aligner.state_dict(), single_aligner_filepath)




###################################################################################


def statistical_alignment(args, dataset_to_align, test_data):
    '''
        dataset_to_align: list of length of N_dataset where each of them has shape
                [N_subjects, N_timepoints, N_roi, N_roi]. Also, N_dataset equals to N_views.
    '''
    N_views = len(dataset_to_align)
    N_timepoints = dataset_to_align[0].shape[1]

    # getting saved cbts
    if args.simulated_data == 0:
        cbt_path = config.CBTS_DIR_REAL_DATA
    elif args.simulated_data == 1:
        cbt_path = config.CBTS_DIR_SIMULATED_DATA

    all_cbts = []
    for t in range(1, N_timepoints+1):
        cbt = np.load(os.path.join(cbt_path, f"t{t}_cbt_1.npy"))
        all_cbts.append(cbt)
        

    cbts_mean = []
    cbts_std = []
    for cbt in all_cbts:
        cbts_mean.append(np.mean(cbt))
        cbts_std.append(np.std(cbt))

    # alignment
    for i in range(N_views):
        for t in range(N_timepoints):
            cur_data = dataset_to_align[i][:, t, :, :]
            cur_mean, cur_std = torch.mean(cur_data), torch.std(cur_data)
            dataset_to_align[i][:, t, :, :] = (((cur_data - cur_mean) / cur_std) * cbts_std[t]) + cbts_mean[t]

            test_data[i][:, t, :, :] = (((test_data[i][:, t, :, :] - cur_mean) / cur_std) * cbts_std[t]) + cbts_mean[t]


    print("Statistical alignment is done!")

    return dataset_to_align, test_data


###################################################################################


def prior_alignment(args, dataset_to_align, test_data):
    '''
        dataset_to_align: list of length of N_dataset where each of them has shape
                [N_subjects, N_timepoints, N_roi, N_roi]. Also, N_dataset equals to N_views.
    '''
    N_views = len(dataset_to_align)
    N_timepoints = dataset_to_align[0].shape[1]

    prior_mean = []
    prior_std = []
    for i in range(N_timepoints):
        prior_mean.append(0.5 + 0.3*i)
        prior_std.append(0.5)

    # alignment
    for i in range(N_views):
        for t in range(N_timepoints):
            cur_data = dataset_to_align[i][:, t, :, :]
            cur_mean, cur_std = torch.mean(cur_data), torch.std(cur_data)
            dataset_to_align[i][:, t, :, :] = (((cur_data - cur_mean) / cur_std) * prior_std[t]) + prior_mean[t]

            test_data[i][:, t, :, :] = (((test_data[i][:, t, :, :] - cur_mean) / cur_std) * prior_std[t]) + prior_mean[t]

    print("Prior alignment is done!")

    return dataset_to_align, test_data