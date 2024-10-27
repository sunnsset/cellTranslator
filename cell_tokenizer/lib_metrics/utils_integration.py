import numpy as np
import torch
import os
import random
import pandas as pd
import scanpy as sc
from glob import glob
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,homogeneity_score,normalized_mutual_info_score

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True 
    
class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, outdir='./'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = os.path.join(outdir, 'model.pt')

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss
        
# load data
def read_mtx(path):
    """
    Read .mtx format file
    """
    for filename in glob(path+'/*'):
        if ('count' in filename or 'matrix' in filename or 'data' in filename) and ('mtx' in filename):
            adata = sc.read_mtx(filename).T
    for filename in glob(path+'/*'):
        if 'barcode' in filename:
            barcode = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            adata.obs = pd.DataFrame(index=barcode)
        if 'gene' in filename or 'peaks' in filename or 'bins' in filename:
            gene = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            adata.var = pd.DataFrame(index=gene)
    return adata

def read_csv(path):
    """
    Read .csv/.txt/.h5ad format file
    """
    adata=str(1)
    for filename in glob(path+'/*'):
        postfix=filename.split('/')[-1]
        if ('count' in postfix or 'data' in postfix) and ('.csv' in postfix or '.csv.gz' in postfix):
            adata = sc.read_csv(filename).T
        elif ('count' in postfix or 'data' in postfix) and ('.txt' in postfix or '.txt.gz' in postfix or '.tsv' in postfix or '.tsv.gz' in postfix):
            df = pd.read_csv(filename, sep='\t', index_col=0).T
            adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))
        elif postfix.endswith('.h5ad'):
            adata = sc.read_h5ad(filename)
        else:
            print('{} no read'.format(filename))
    if not isinstance(adata,str):
        return adata
    else:
        raise ValueError("File {} not exists".format(path))
def load_file(path): 
    """
    Load adata
    """
    if len(glob(path+'/*')) < 3:
        adata = read_csv(path)
    elif len(glob(path+'/*')) >= 3:
        adata = read_mtx(path)

    if type(adata.X) == np.ndarray:
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata

def run_louvain(adata,n_cluster,range_min=0,range_max=3,max_steps=15):  # 之前 max_step 是 30
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
#         print('step ' + str(this_step))
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.louvain(adata,resolution=this_resolution)
        this_clusters = adata.obs['louvain'].nunique()

#         print('got ' + str(this_clusters) + ' at resolution ' + str(this_resolution))

        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
#             return(this_resolution, adata)
            ari = adjusted_rand_score(adata.obs['label'], adata.obs['louvain'])
            ami = adjusted_mutual_info_score(adata.obs['label'], adata.obs['louvain'])
            homo = homogeneity_score(adata.obs['label'], adata.obs['louvain'])
            nmi = normalized_mutual_info_score(adata.obs['label'], adata.obs['louvain'])
            # print('ARI: %.3f, AMI: %.3f, Homo: %.3f, NMI: %.3f' % (ari,ami,homo,nmi))
            return (adata.obs['label'], adata.obs['louvain'], ari,ami,homo,nmi)
        this_step += 1

    print('Cannot find the number of clusters')
#     print('Clustering solution from last iteration is used:' + str(this_clusters) + ' at resolution ' + str(this_resolution))

    ari = adjusted_rand_score(adata.obs['label'], adata.obs['louvain'])
    ami = adjusted_mutual_info_score(adata.obs['label'], adata.obs['louvain'])
    homo = homogeneity_score(adata.obs['label'], adata.obs['louvain'])
    nmi = normalized_mutual_info_score(adata.obs['label'], adata.obs['louvain'])
    # print('ARI: %.3f, AMI: %.3f, Homo: %.3f, NMI: %.3f' % (ari,ami,homo,nmi))
    return (adata.obs['label'], adata.obs['louvain'], ari,ami,homo,nmi)