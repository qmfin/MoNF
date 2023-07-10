import math
import numpy as np
import torch
import torch.nn as nn
from survae.flows import InverseFlow
from survae.distributions import Distribution
from survae.transforms.bijections import Bijection


def m_func(x):
    b = torch.tensor(2.0).log()
    pos = nn.functional.softplus(x)-b
    neg = -nn.functional.softplus(-x)+b
    return torch.where(x > 0, pos, neg)


def m_inv(x):
    b = torch.tensor(2.0).log()
    pos = x+b+torch.log1p(-torch.exp(-x-b))
    neg = x-b-torch.log1p(-torch.exp(x-b))
    return torch.where(x > 0, pos, neg)


def m_grad_log(x):
    pos = nn.functional.logsigmoid(x)
    neg = nn.functional.logsigmoid(-x)
    return torch.where(x > 0, pos, neg)


class M(Bijection):
    def __init__(self):
        super(M, self).__init__()

    def forward(self, x):
        z = m_func(x)
        ldj = m_grad_log(x)
        return z, ldj

    def inverse(self, z):
        x = m_inv(z)
        return x


class StandardNormal(Distribution):

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        log_base = - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * x ** 2
        return log_base + log_inner

    def sample(self, num_samples):
        return torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)


class ScalarAffine(Bijection):
    def __init__(self, shift=None, scale=None):
        super(ScalarAffine, self).__init__()
        self._shift = shift
        self._scale = scale

    def forward(self, x):
        z = x * self._scale + self._shift
        ldj = torch.log(torch.abs(self._scale))
        ldj = ldj.expand(z.shape)
        return z, ldj

    def inverse(self, z):
        x = (z - self._shift) / self._scale
        return x


class InverseFlowExt(InverseFlow):
    def log_prob(self, x):
        log_prob = torch.zeros(x.shape, device=x.device)
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
            log_prob -= transform(x)[-1]
        log_prob += self.base_dist.log_prob(x)
        return log_prob


ae_func = torch.nn.L1Loss(reduction='none')
se_func = torch.nn.MSELoss(reduction='none')


def compute_fitting_loss(pred_price, ground_truth_price):
    absolute_error = ae_func(pred_price, ground_truth_price)
    absolute_percentage_error = absolute_error / ground_truth_price
    squared_error = se_func(pred_price, ground_truth_price)

    return absolute_error, absolute_percentage_error, squared_error


class RND(nn.Module):

    def __init__(self, n_models, n_layers, device):
        super(RND, self).__init__()
        self.n_models = n_models
        self.weights = nn.Parameter(torch.randn(n_models)*0.1)
        self.shifts = []
        self.scales = []

        for _ in range(n_layers):
            self.shifts.append(nn.Parameter(torch.zeros(n_models)))
            self.scales.append(nn.Parameter(1.0 + torch.randn(n_models) * 0.1))

        self.samples_buff = None
        self.model = self.init_model(n_models, n_layers, device)

    def init_model(self, n_models, n_layers, device):
        transforms = []
        for i in range(n_layers):
            transforms.extend(
                [ScalarAffine(self.shifts[i], self.scales[i]), M()])

        return InverseFlowExt(base_dist=StandardNormal((n_models,)), transforms=transforms).to(device)

    def sample(self, n_samples):
        self.samples_buff = self.model.sample(n_samples)
        return self.samples_buff

    def _sample(self, num_samples):
        nf_idx = torch.multinomial(
            self.get_weights(), num_samples, replacement=True)
        idx, count = torch.unique(nf_idx, return_counts=True)
        idx = idx.detach().cpu().numpy()
        count = count.detach().cpu().numpy()
        d = dict(zip(idx, count))
        count_max = count.max()
        samples = self.model.sample(count_max).detach().cpu().numpy()
        samples_final = []
        for i in range(self.n_models):
            if i in d:
                samples_final.append(np.random.choice(
                    samples[:, i], d[i], replace=False))
        return torch.tensor(np.concatenate(samples_final))

    def get_weights(self):
        return nn.Softmax(dim=-1)(self.weights)

    def rsample(self, num_samples, num=5000, min_mul=1.25):
        x = self._sample(num_samples)
        prob = self.prob(x).detach()

        start = x.min()*min_mul
        end = x.max()
        delta_x = (end-start)/num
        s = torch.linspace(start, end, num)

        p = self.prob(s)
        cdf = (((x.view(-1, 1)-s.view(1, -1)) >= 0).detach()*p).sum(dim=1)*delta_x

        surrogate_x = -cdf/prob

        return x + (surrogate_x - surrogate_x.detach())

    def prob(self, x):
        log_prob = self.model.log_prob(
            x.view(-1, 1).expand([len(x), self.n_models]))
        prob = log_prob.exp()
        prob = torch.where(torch.isnan(prob), torch.zeros_like(prob), prob)
        prob = torch.where(torch.isinf(prob), torch.zeros_like(prob), prob)
        prob = (prob*self.get_weights()).sum(dim=1)
        return prob

    def mc_pricing(self, n_samples, strike_price_normalized, is_call, rescaler=1.0):
        log_moneyness_samples = self.sample(n_samples)
        moneyness_samples = log_moneyness_samples.exp()*rescaler
        price_norm = torch.relu(
            (moneyness_samples[:, :, None] - strike_price_normalized) * is_call).mean(dim=0)
        price_norm = (self.get_weights()[:, None]*price_norm).sum(dim=0)
        return price_norm

    def ni_pricing(self, k_min, k_max, num, strike_price_normalized, is_call, rescaler=1.0):
        s = torch.linspace(k_min, k_max, num)
        d = (k_max - k_min) / num
        prob = self.prob(s)
        inner = torch.relu((s.view(-1, 1).exp()*rescaler -
                           strike_price_normalized) * is_call) * prob.view(-1, 1)
        price_norm = inner.sum(axis=0) * d
        return price_norm
