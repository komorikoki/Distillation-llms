import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
torch.cuda.empty_cache()
device = "cuda"

num_vectors = 1
vector_len = 1

big_matrix = torch.randn(num_vectors, vector_len, device=device)
big_matrix /= big_matrix.norm(p=2, dim=1, keepdim=True)
big_matrix.requires_grad_(True)

optimizer = torch.optim.Adam([big_matrix], lr=0.01)
num_steps = 250

losses= []
dot_diff_cutoff = 0.01
big_id = torch.eye(num_vectors, num_vectors, device=device)

batch_size = 1
for step_num in tqdm(range(num_vectors // 1)):
    optimizer.zero_grad()
    dot_products = big_matrix @ big_matrix.T

    diff = dot_products - big_id
    loss = (diff.abs() - dot_diff_cutoff).relu().sum()
    loss += num_vectors * diff.diag().pow(2).sum()

    loss.backward()
    optimizer.step()
    losses.append(loss.item())