import pickle
import torch

with open('grid_sample_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())

fmap2 = torch.Tensor(data['fmap2'])
print(fmap2.shape)

grid_slice = torch.Tensor(data['grid_slice'])
print(grid_slice.shape)

grid_sampler_output = torch.Tensor(data['fmapw_mini (output of F.grid_sample)'])
print(grid_sampler_output.shape)

