import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import torch
import config
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from preprocess import convert_matrices_to_with_features


def plot_distribution(args, datasets):
    min_vals, max_vals = [], []
    for dataset in datasets:
        min_vals.append(dataset.min())
        max_vals.append(dataset.max())
    min_val, max_val = np.min(min_vals), np.max(max_vals)

    _, axes = plt.subplots(len(datasets),)
    for i, fmri_data in enumerate(datasets):
        timepoints = fmri_data.shape[1]
        for t in range(timepoints):
            fmri_data_t = fmri_data[:, t, :, :]

            triu_indices_x, triu_indices_y = torch.triu_indices(fmri_data.shape[2], fmri_data.shape[2], 1)
            fmri_data_1d = fmri_data_t[:, triu_indices_x, triu_indices_y].flatten()
        
            sns.kdeplot(data=fmri_data_1d, ax=axes[i], fill=True, alpha=0.2, linewidth=0.4, clip=(min_val, max_val))

        labels = ["t"+str(i) for i in range(1, timepoints+1)]
        axes[i].set(title="all t's for domain "+str(i+1))
        axes[i].legend(labels=labels, loc='upper right')
        axes[i].set_xlim(min_val - 0.5, max_val + 0.5)

    plt.tight_layout()
    # plt.savefig(os.path.join(config.PLOT_DIR, f"{args.simulated_data}SIM_{args.alignment}PA_EXP{args.exp}_distribution.png"), bbox_inches='tight')
    plt.show()

def plot_tsne(args, datasets):
    N_views = len(datasets)
    N_subjects = datasets[0].shape[0]
    N_timepoints = datasets[0].shape[1]

    all_view_and_timepoints = convert_matrices_to_with_features(datasets)

    tsne = TSNE(n_components=2)

    _, axes = plt.subplots(2,2)
    for i in range(N_views):
        
        data = all_view_and_timepoints[N_timepoints*i : N_timepoints*i + N_timepoints]
        data = torch.stack(data)
        data = torch.reshape(data, shape=(-1, data.size(2)))
        
        tsne_result = tsne.fit_transform(data)

        for t in range(N_timepoints):
            tsne_for_t = tsne_result[t*N_subjects:N_subjects*(t+1)]
            sns.scatterplot(x=tsne_for_t[:, 0], y=tsne_for_t[:, 1], ax=axes[i//2][i % 2])
        
        labels = ["t"+str(i) for i in range(1, N_timepoints+1)]
        axes[i//2][i % 2].set_title("all t's for domain "+str(i+1))
        axes[i//2][i % 2].legend(labels=labels, loc='upper right')

    plt.tight_layout()
    # plt.savefig(os.path.join(config.PLOT_DIR, f"{args.simulated_data}SIM_{args.alignment}PA_EXP{args.exp}_distribution.png"), bbox_inches='tight')
    plt.show()
    

def plot_distribution_all_in_one(datasets, N_views, N_timepoints):
    min_vals, max_vals = [], []
    for d in datasets:
        for dataset in d:
            min_vals.append(dataset.min())
            max_vals.append(dataset.max())
        min_val, max_val = np.min(min_vals), np.max(max_vals)

    _, axes = plt.subplots(N_views, N_timepoints)
    for i, dataset in enumerate(datasets):
        for fmri_data in dataset:
            fmri_data_t = fmri_data[:, :, :]

            triu_indices_x, triu_indices_y = torch.triu_indices(fmri_data.shape[2], fmri_data.shape[2], 1)
            fmri_data_1d = fmri_data_t[:, triu_indices_x, triu_indices_y].flatten()

            sns.kdeplot(data=fmri_data_1d, ax=axes[int(i//N_timepoints)][i % N_timepoints], fill=True, alpha=0.2, linewidth=0.4, clip=(fmri_data.min(), fmri_data.max()))

            labels = ["original", "target CBT", "aligned"]
            axes[int(i//N_timepoints)][i % N_timepoints].set(title="dataset "+str(i//N_timepoints + 1)+" t="+str(i % N_timepoints + 1))
            axes[int(i//N_timepoints)][i % N_timepoints].legend(labels=labels, loc='upper right')
            axes[int(i//N_timepoints)][i % N_timepoints].set_xlim(min_val - 0.5, max_val + 0.5)
        
    plt.tight_layout()
    # plt.savefig(os.path.join(config.PLOT_DIR, f"ALIGNMENTRESULTS_{args.simulated_data}SIM_{args.alignment}PA_EXP{args.exp}_distribution.png"), bbox_inches='tight')
    plt.show()


def plot_aligner_loss(losses, fold, view, timepoint, path):
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(path, f"fold{fold}_view{view+1}_t{timepoint+1}_all_losses.png"))
    plt.close()


def plot(args, loss, title, losses):
    plt.plot(losses)
    plt.xlabel("# epoch")
    plt.ylabel(loss)
    plt.title(title)
    plt.savefig(os.path.join(args.path, "losses", f"EXP{args.exp}_{args.eval_mode.upper()}_{args.mode}_{loss}___{title}.png"))
    plt.close()


def plot_matrix(out, title):
    plt.pcolor(abs(out))
    plt.colorbar()
    plt.imshow(out)
    # title = "RBGM Output,  Strategy = " + strategy
    plt.title(title)
    # plt.savefig('./plots/' + str(strategy) + '.png')
    plt.show()
    