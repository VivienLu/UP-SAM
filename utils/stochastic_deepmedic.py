import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F
from utils.distributions import ReshapedDistribution


def to_one_hot_bg_reverse(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    # print(tensor.max())
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    one_hot[:, 0, ...] = 1 - one_hot[:, 0, ...] 
    return one_hot

# FEATURE_MAPS = (30, 30, 40, 40, 40, 40, 50, 50)
# f_out is the number of channels for the input
f_out = 16

class StochasticDeepMedic(nn.Module):
    def __init__(self,
                 num_classes,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False,
                 dim=3):
        super().__init__()
        self.dim = dim
        conv_fn = nn.Conv3d if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.mean_l = conv_fn(f_out, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(f_out, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(f_out, num_classes * rank, kernel_size=(1, ) * self.dim)

    def forward(self, input, sampling_mask):
        logits = F.relu(input)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        #mask = kwargs['sampling_mask']

        # 有一种想法是改进这里的sample mask，也就是不同的分类对应不同的channel对应不同的mask，要在sam认可的/VNet认可的区域进行采样
        # 然后对应不同的mu, D, P卷积 

        mask = sampling_mask # 这里需要mask是one_hot形式
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)

        cov_factor = cov_factor * mask.unsqueeze(-1)
        cov_diag = cov_diag * mask + self.epsilon

        # print('Logits shape: {}, batch_size: {}, event shape: {}, mean shape: {}, cov_diag shape: {}, cov_factor shape: {}'.format(
        #     logits.shape, batch_size, event_shape, mean.shape, cov_diag.shape, cov_factor.shape,
        # )) 
        # # Logits shape: torch.Size([2, 16, 128, 128, 128]), batch_size: 2, event shape: (2, 128, 128, 128), 
        # # mean shape: torch.Size([2, 4194304]), cov_diag shape: torch.Size([2, 4194304]), cov_factor shape: torch.Size([2, 4194304, 10])
        # print('sampling_mask shape: {}, middel shape: {}\t{}, Mask shape: {}, diff mask: {}'.format(
        #     sampling_mask.shape, (batch_size, self.num_classes) + sampling_mask.shape[1:], 
        #     sampling_mask.unsqueeze(1).expand((batch_size, self.num_classes) + sampling_mask.shape[1:]).shape, mask.shape,
        #     torch.mean(mask_-mask)))
        # # sampling_mask shape: torch.Size([2, 128, 128, 128]), Mask shape: torch.Size([2, 4194304])

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = cov_factor.transpose(2, 1).view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict
