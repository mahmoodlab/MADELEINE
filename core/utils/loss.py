import torch
import torch.nn.functional as F
from torch import nn

import pdb 

__all__ = ['InfoNCE', 'info_nce', 'GOT', 'init_intra_wsi_loss_function']

# Global loss implementation
class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None, symmetric=False):
        return self.info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        symmetric=symmetric)

    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired', symmetric=False):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors (normalize by euclidean distance)
        query, positive_key, negative_keys = self.normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ self.transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ self.transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ self.transpose(positive_key)

            # Positive keys are the entries on the diagonal (class indices for each row)
            labels = torch.arange(len(query), device=query.device)
            
            # symmetric contrastive loss 
            if symmetric:
                logits2 = positive_key @ self.transpose(query)
                loss = 0.5*F.cross_entropy(logits / temperature, labels, reduction=reduction) + 0.5*F.cross_entropy(logits2 / temperature, labels, reduction=reduction)
            else:
                loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
                
            return loss

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]
    
################################################################################################################################

# Intra loss implementation
def init_intra_wsi_loss_function(config):
    """
    Initializes the intra-modality loss function based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        loss_fn: The initialized loss function.

    Raises:
        None

    """
    if config["intra_modality_mode_wsi"] == "reconstruct_avg_emb" or config["intra_modality_mode_wsi"] == "reconstruct_masked_emb":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = InfoNCE(temperature=config["temperature"])
    return loss_fn

################################################################################################################################

# Local loss implementation
# From: https://github.com/ShramanPramanick/VoLTA/blob/master/Pre-training/main.py 
def cost_matrix_batch_torch(x, y):
	"Returns the cosine distance batchwise"
	# x is the image feature: bs * d * m * m
	# y is the audio feature: bs * d * nF
	# return: bs * n * m
	# print(x.size())
	bs = list(x.size())[0]
	D = x.size(1)
	assert(x.size(1)==y.size(1))
	x = x.contiguous().view(bs, D, -1) # bs * d * m^2
	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
	cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)#.transpose(1,2)
	cos_dis = 1 - cos_dis # to minimize this value
	return cos_dis.transpose(2,1)


def IPOT_torch_batch_uniform(C, bs, n, m, beta=0.5, iteration=50):
    # C is the distance matrix
    # c: bs by n by m
    sigma = torch.ones(bs, int(m), 1).cuda()/float(m)
    T = torch.ones(bs, n, m).cuda()
    A = torch.exp(-C/beta).float().cuda()
    for t in range(iteration):
        Q = A * T # bs * n * m
        for k in range(1):
            delta = 1 / (n * torch.bmm(Q, sigma))
            a = torch.bmm(torch.transpose(Q,1,2), delta)
            sigma = 1 / (float(m) * a)
        T = delta * Q * sigma.transpose(2,1)

    return T


def batch_trace(input_matrix, n, bs):
	a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
	b = a * input_matrix
	return torch.sum(torch.sum(b,-1),-1).unsqueeze(1)


def IPOT_distance_torch_batch_uniform(C, bs, n, m, iteration=50):
	C = C.float().cuda()
	T = IPOT_torch_batch_uniform(C, bs, n, m, iteration=iteration)
	temp = torch.bmm(torch.transpose(C,1,2), T)
	distance = batch_trace(temp, m, bs)
	return -distance


def cos_batch_torch(x, y):
	"Returns the cosine distance batchwise"
	# x is the image feature: bs * d * m * m
	# y is the audio feature: bs * d * nF
	# return: bs * n * m
	# print(x.size())
	bs = x.size(0)
	D = x.size(1)
	assert(x.size(1)==y.size(1))
	x = x.contiguous().view(bs, D, -1) # bs * d * m^2
	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
	cos_dis = torch.bmm(torch.transpose(x,1,2), y)#.transpose(1,2)
	cos_dis = 1 - cos_dis # to minimize this value
	# return cos_dis.transpose(2,1)
	# TODO:
	beta = 0.1
	min_score = cos_dis.min()
	max_score = cos_dis.max()
	threshold = min_score + beta * (max_score - min_score)
	res = cos_dis - threshold
	# res = torch.nn.ReLU()

	return torch.nn.functional.relu(res.transpose(2,1))


def GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
	one_m = torch.ones(bs, m, 1).float().cuda()
	one_n = torch.ones(bs, n, 1).float().cuda()

	Cst = torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_m, 1, 2)) + \
	      torch.bmm(one_n, torch.bmm(torch.transpose(q,1,2), torch.transpose(Ct**2, 1, 2))) # bs by n by m
	gamma = torch.bmm(p, q.transpose(2,1)) # outer product, init
	# gamma = torch.einsum('bi,bj->bij', (torch.squeeze(p), torch.squeeze(q))) # outer product, initialization
	for i in range(iteration):
		C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
		gamma = IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
	Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
	return gamma.detach(), Cgamma


def GW_distance(X, Y, p, q, lamda=0.5, iteration=5, OT_iteration=20):
	'''
	:param X, Y: Source and target embeddings , batchsize by embed_dim by n
	:param p, q: probability vectors
	:param lamda: regularization
	:return: GW distance
	'''
	Cs = cos_batch_torch(X, X).float().cuda()
	Ct = cos_batch_torch(Y, Y).float().cuda()
	bs = Cs.size(0)
	m = Ct.size(2)
	n = Cs.size(2)
	T, Cst = GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
	temp = torch.bmm(torch.transpose(Cst,1,2), T)
	distance = batch_trace(temp, m, bs)
	return distance


def GW_distance_uniform(X, Y, lamda=1e-1, iteration=5, OT_iteration=20):
	m = X.size(2)
	n = Y.size(2)
	bs = X.size(0)
	p = (torch.ones(bs, m, 1)/m).cuda()
	q = (torch.ones(bs, n, 1)/n).cuda()
	return GW_distance(X, Y, p, q, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)


def GOT(v_, q_, subsample=None):

    # randomly subsample of subset of the tokens to keep memory ok
    if subsample is not None:
        patch_indices = torch.randperm(v_.shape[0])[:subsample]
        v_ = v_[:, patch_indices, :]
        q_ = q_[:, patch_indices, :]

    cos_distance = cost_matrix_batch_torch(v_.transpose(2, 1), q_.transpose(2, 1))
    cos_distance = cos_distance.transpose(1,2)
    beta = 0.1
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)

    wd = -IPOT_distance_torch_batch_uniform(cos_dist, v_.size(0), v_.size(1), q_.size(1), 30)
    # torch.distributed.all_reduce(wd)
    wd = torch.sum(wd)
    gwd = GW_distance_uniform(v_.transpose(2,1), q_.transpose(2,1))
    # torch.distributed.all_reduce(gwd)
    gwd = torch.sum(gwd)
    twd = torch.mean(gwd) + torch.mean(wd)

    return twd