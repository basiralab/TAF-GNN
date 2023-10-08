from scipy import io
import torch
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# dataset_builder(args, "./LMCI_AD/LMCI_AD/LH_dataset")

def dataset_builder(path):
    # get filenames
    file_view1_t1 = os.path.join(path, "67subjectsRH_view1_t1.mat")
    file_view1_t2 = os.path.join(path, "67subjectsRH_view1_t2.mat")
    file_view2_t1 = os.path.join(path, "67subjectsRH_view2_t1.mat")
    file_view2_t2 = os.path.join(path, "67subjectsRH_view2_t2.mat")
    file_view3_t1 = os.path.join(path, "67subjectsRH_view3_t1.mat")
    file_view3_t2 = os.path.join(path, "67subjectsRH_view3_t2.mat")
    file_view4_t1 = os.path.join(path, "67subjectsRH_view4_t1.mat")
    file_view4_t2 = os.path.join(path, "67subjectsRH_view4_t2.mat")

    # open files and load data to numpy, then to torch
    # each of them is [67, 595]
    view1_t1 = torch.from_numpy(np.array(io.loadmat(file_view1_t1)["view1"]))
    view1_t2 = torch.from_numpy(np.array(io.loadmat(file_view1_t2)["view1"]))
    view2_t1 = torch.from_numpy(np.array(io.loadmat(file_view2_t1)["view2"]))
    view2_t2 = torch.from_numpy(np.array(io.loadmat(file_view2_t2)["view2"]))
    view3_t1 = torch.from_numpy(np.array(io.loadmat(file_view3_t1)["view3"]))
    view3_t2 = torch.from_numpy(np.array(io.loadmat(file_view3_t2)["view3"]))
    view4_t1 = torch.from_numpy(np.array(io.loadmat(file_view4_t1)["view4"]))
    view4_t2 = torch.from_numpy(np.array(io.loadmat(file_view4_t2)["view4"]))

    # take some features
    # TODO: take these from args
    N_subjects = view1_t1.size()[0]
    N_timepoints = 2
    N_regions = 35
    N_views = 4

    # resulting dataset is [N_subjects, N_timepoints, N_regions, N_regions, N_views], or [67, 2, 35, 35, 4]
    dataset = torch.zeros((N_subjects, N_timepoints, N_regions, N_regions, N_views), dtype=torch.float64)
    
    # making the dataset symmetric with corresponding features
    triu_indices_x, triu_indices_y = torch.triu_indices(N_regions, N_regions, 1)
    
    dataset[:, 0, triu_indices_x, triu_indices_y, 0] = view1_t1
    dataset[:, 1, triu_indices_x, triu_indices_y, 0] = view1_t2
    dataset[:, 0, triu_indices_x, triu_indices_y, 1] = view2_t1
    dataset[:, 1, triu_indices_x, triu_indices_y, 1] = view2_t2
    dataset[:, 0, triu_indices_x, triu_indices_y, 2] = view3_t1
    dataset[:, 1, triu_indices_x, triu_indices_y, 2] = view3_t2
    dataset[:, 0, triu_indices_x, triu_indices_y, 3] = view4_t1
    dataset[:, 1, triu_indices_x, triu_indices_y, 3] = view4_t2

    dataset[:, :, :, :, :] = torch.add(dataset[:, :, :, :, :], torch.transpose(dataset[:, :, :, :, :], 2, 3))
    
    dataset = dataset.to(dtype=torch.float32)

    return dataset



def diverse_simulated_data(data):
    ''' data: (200, 4, 35, 35, 4)
    '''
    N_timepoints = data.shape[1]
    N_views = data.shape[4]
    
    devs = torch.mean(data, dim=[0,2,3,4])

    shift_amounts = np.random.uniform(0.02, 0.08, size=(N_timepoints*N_views))
    scale_amounts = np.random.uniform(0.85, 1.16, size=(N_timepoints*N_views))

    # scale_amounts[8] -= 0.1
    # scale_amounts[11] += 0.04
    # shift_amounts[11] += 0.01

    for i in range(N_timepoints):
        shift_amounts[i*N_timepoints:(i+1)*N_timepoints] = np.sort(shift_amounts[i*N_timepoints:(i+1)*N_timepoints])
        scale_amounts[i*N_timepoints:(i+1)*N_timepoints] = np.sort(scale_amounts[i*N_timepoints:(i+1)*N_timepoints])

    scale_amounts[9] += 0.05
    scale_amounts[10] -= 0.1
    shift_amounts[11] += 0.01


    # shift_amounts = [np.random.uniform(0, 0.2)]
    # scale_amounts = [np.random.uniform(0.8, 1.2)]
    # for i in range(1, N_timepoints*N_views):
    #     shift_amounts.append(shift_amounts[i - 1] + np.random.uniform(0, 0.2))
    #     scale_amounts.append(np.random.uniform(0.9, 1.2))

    # shift_amounts = np.array(shift_amounts)
    # scale_amounts = np.array(scale_amounts)


    c = 0
    for i in range(N_views):
        for t in range(N_timepoints):
            data[:, t, :, :, i] = (data[:, t, :, :, i]) * scale_amounts[c]  + shift_amounts[c]
            c += 1

    return data