import torch

class MLAInstanceNorm2d(torch.nn.Module):
    def __init__(self, 
                 num_features, 
                 eps = 1e-05, 
                 momentum = 0.1,
                 affine = False,
                 track_running_stats = False,
                 device = None,
                 dtype = None):

                 super(MLAInstanceNorm2d, self).__init__()
                 self.num_features = num_features
                 self.eps = eps
                 self.momentum = momentum
                 self.affine = affine
                 self.track_running_stats = track_running_stats
                 self.device = device
                 self.dtype = dtype

                 if self.affine:
                    self.weight = torch.nn.Parameter(torch.ones(1, num_features, 1, 1))
                    self.bias = torch.nn.Parameter(torch.ones(1, num_features, 1, 1))
                 else:
                      self.weight = None
                      self.bias = None

    
    def forward(self, x):
        # numerator
        x_mean = torch.mean(x, dim = (2, 3), keepdim = True)
        #numerator = torch.sub(x, x_mean)
        numerator = x - x_mean

        # denominator
        #eps_mod = torch.full(x.shape, self.eps).to('cuda')

        x_var = torch.var(x, dim = (2, 3), keepdim = True)
        denominator = torch.sqrt(x_var)

        #x_norm = torch.divide(numerator, denominator)
        x_norm = numerator / denominator

        if self.affine:
            #x_norm = x_norm * self.weight + self.bias
            x_norm = torch.add(torch.multiply(x_norm, self.weight))

        return x_norm