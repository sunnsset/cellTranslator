import torch
from torch.utils.data import DataLoader
import time
import os
import numpy as np
# import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import pandas as pd


def save_model_and_results(model, eval_adata, eval_loader, results, hyperparameters, timestamp, args):
    # saving model
    SAVE_MODEL_PATH = os.path.join(args.outdir, f'epoch_{timestamp}')
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters}
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/model.pth')
    
    
#     # saving UMAP latent result on eval dataset
#     newtest_adata = eval_adata.copy()
#     # breakpoint()
#     # 定位到这句话：：
#     test = model.encodeBatch(eval_loader, device=args.device, out='z_ouput')
#     newtest_adata.obsm['latent'] = test.numpy()
#     # 定位到上一句话up
    
#     proj = umap.UMAP(random_state=args.seed).fit_transform(newtest_adata.obsm['latent'])
#     fig = plt.figure()
#     fig.set_figwidth(15)
#     fig.set_figheight(15)
#     df = {'component_1':proj[:, 0],\
#           'component_2':proj[:, 1], \
#           'label':newtest_adata.obs['cell_type']}
#     df = pd.DataFrame(df)
#     ax = sns.scatterplot(x="component_1", y="component_2", hue="label", palette = 'Dark2', s=5, linewidth = 0.05, data=df)
#     ax.legend()

#     plt.savefig(SAVE_MODEL_PATH + '/eval_latent.png', bbox_inches='tight')