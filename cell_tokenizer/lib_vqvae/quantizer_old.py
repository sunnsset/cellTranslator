import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n : the number of codebooks an embedding is split into
    - n_e : the number of codebooks
    - e_dim : the dim of each codebook
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n, n_e, e_dim, beta, gamma):
        super(VectorQuantizer, self).__init__()
        self.n = n
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.gamma = gamma
    
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.embedding.weight.data.normal_()


    def forward(self, z, device):
        """
        the size of z is  (batch_size, n*e_dim)
        """
        # breakpoint()
        z = z.view(z.size(0), self.n, self.e_dim)
        # flatten z
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # d[i,j] is the square of the distance between z_flattened[i] and the j-th embedding
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        # the min indice of each row
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # one-hot, if z_flattened[i] is the closest embedding to the j-th embedding, then min_encodings[i,j] is 1 and the other positions are 0
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        # matrix multiplication of min_encodings with all embedding vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        # loss1 = torch.mean((z_q.detach()-z)**2)  # 字典学习算法里的经典算法Vector Quantisation(VQ)，也就是VQ-VAE里的那个VQ，它用于优化嵌入空间
        # loss2 = torch.mean((z_q - z.detach()) ** 2)  # 专注误差，它用于约束编码器的输出，不让它跑到离嵌入空间里的向量太远的地方
        loss1 = F.mse_loss(z_q.detach(), z) # e_latent_loss
        loss2 = F.mse_loss(z_q, z.detach()) # q_latent_loss
        loss = self.gamma * loss1 + self.beta * loss2

        # preserve gradients
        # z_q size (batch_size, n, e_dim)
        z_q = z + (z_q - z).detach()

        # perplexity
        # obtained by the entropy of the cluster distribution obtained by min_encodings, and its function is to measure the degree of dispersion of the model
        e_mean = torch.mean(min_encodings, dim=0)
        # print(e_mean)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        # reshape for decoder
        z_output = z_q.view(z_q.size(0), -1)

        return loss, loss1, loss2, z_q, z_output, perplexity, min_encodings, min_encoding_indices