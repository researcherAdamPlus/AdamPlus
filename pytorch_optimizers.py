import math
import torch
from torch.optim import Optimizer
import numpy as np


class AdamPlus(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-10, weight_decay=0.0,
                 is_mask=True, is_lamb=False, weight_decouple=False,
                 db_threshold=-3, db_noise=-80, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, weight_decouple=weight_decouple,
                        amsgrad=amsgrad)
        super(AdamPlus, self).__init__(params, defaults)
        self.iteration = 0
        self.is_mask = is_mask
        self.is_lamb = is_lamb
        self.db_noise = db_noise
        self.db_threshold = db_threshold
        self.noise_std = 10 ** (db_noise / 20)

    @torch.no_grad()
    def step(self):
        self.iteration += 1

        for group in self.param_groups:
            beta1, beta2, eps, weight_decay = group['beta1'], group['beta2'], group['eps'], group['weight_decay']
            lr = group['lr']
            weight_decouple = group.get('weight_decouple', False)
            amsgrad = group.get('amsgrad', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    if weight_decouple:
                        # Apply weight decay directly to parameters
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['snr'] = torch.zeros_like(p.data)
                    state['acc'] = torch.zeros_like(p.data)
                    state['cnt'] = torch.zeros_like(p.data)
                    state['mask'] = torch.zeros_like(p.data, dtype=torch.bool)
                    if beta1 != beta2:
                        state['w'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']
                acc, cnt, snr, mask = state['acc'], state['cnt'], state['snr'], state['mask']
                if beta1 != beta2:
                    w = state['w']
                if amsgrad:
                    max_v = state['max_exp_avg_sq']

                noisy_grad = grad
                if self.db_noise > -100:
                    noisy_grad += torch.normal(
                        mean=0.0, std=self.noise_std, size=grad.shape, device=grad.device, dtype=grad.dtype
                    )
                # m.mul_(beta1).add_(noisy_grad, alpha=1 - beta1)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                if beta1 != beta2:
                    w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                    v.mul_(beta2).addcmul_((noisy_grad - w), (noisy_grad - w), value=1 - beta2)
                else:
                    v.mul_(beta2).addcmul_((noisy_grad - m), (noisy_grad - m), value=1 - beta2)

                if amsgrad:
                    torch.maximum(max_v, v, out=max_v)
                    denom = max_v.sqrt().add_(eps)
                else:
                    denom = v.sqrt().add_(eps)

                power_sig = m.pow(2)
                power_noise = v
                snr.copy_(10.0 * torch.log10(power_sig / (power_noise + 1e-10)))

                if self.iteration > 10:
                    beta1_t = beta1 ** self.iteration
                    beta2_t = beta2 ** self.iteration
                    alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)

                    update = m / denom
                    if not self.is_lamb:
                        trust_ratio = 1.0
                    else:
                        m_norm = update.norm()
                        w_norm = p.data.norm()
                        # trust_ratio = w_norm / m_norm if m_norm > 0 and w_norm > 0 else 1.0
                        trust_ratio = w_norm / m_norm

                    if not self.is_mask:
                        p.data.sub_(alpha * trust_ratio* update)
                    else:
                        acc.add_(update)
                        cnt.add_(1.0)
                        snr_minus_threshold = snr - self.db_threshold
                        clipped_exponent = torch.clamp(-snr_minus_threshold, min=-20.0, max=30.0)
                        cnt_threshold = 10.0 ** (clipped_exponent / 10.0)
                        mask.copy_(cnt > cnt_threshold)

                        acc_applied = acc * mask.float()
                        p.data.sub_(alpha * trust_ratio * acc_applied)
                        acc.sub_(acc_applied)
                        cnt.sub_(mask.float())

        return None

    def snr_layer(self):
        snr = None
        if self.iteration > 10:
            snr = np.nanmean(self.state[self.param_groups[0]['params'][-1]]['snr'].cpu().detach().numpy())
        return snr
    
    def v_layer(self):
        v = None
        if self.iteration > 10:
            v = self.state[self.param_groups[0]['params'][-1]]['exp_avg_sq'].cpu().detach().numpy()
            v = 10 * np.log10(v)
            v = np.nanmean(v)
        return v



class Adam5(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-10, weight_decay=0.0,
                 is_mask=False, is_lamb=False, weight_decouple=False, option=0,
                 db_threshold=-3, db_noise=-80, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, weight_decouple=weight_decouple,
                        amsgrad=amsgrad)
        super(Adam5, self).__init__(params, defaults)
        self.iteration = 0
        self.is_mask = is_mask
        self.is_lamb = is_lamb
        self.db_noise = db_noise
        self.db_threshold = db_threshold
        self.noise_std = 10 ** (db_noise / 20)
        self.option = option

    @torch.no_grad()
    def step(self):
        self.iteration += 1

        for group in self.param_groups:
            beta1, beta2, eps, weight_decay = group['beta1'], group['beta2'], group['eps'], group['weight_decay']
            lr = group['lr']
            weight_decouple = group.get('weight_decouple', False)
            amsgrad = group.get('amsgrad', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    if weight_decouple:
                        # Apply weight decay directly to parameters
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['snr'] = torch.zeros_like(p.data)
                    state['acc'] = torch.zeros_like(p.data)
                    state['cnt'] = torch.zeros_like(p.data)
                    state['mask'] = torch.zeros_like(p.data, dtype=torch.bool)
                    if beta1 != beta2:
                        state['w'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']
                acc, cnt, snr, mask = state['acc'], state['cnt'], state['snr'], state['mask']
                if beta1 != beta2:
                    w = state['w']
                if amsgrad:
                    max_v = state['max_exp_avg_sq']

                noisy_grad = grad
                if self.db_noise > -100:
                    noisy_grad += torch.normal(
                        mean=0.0, std=self.noise_std, size=grad.shape, device=grad.device, dtype=grad.dtype
                    )
                w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                noise = noisy_grad - w
                v.mul_(beta2).addcmul_(noise, noise, value=1 - beta2)

                if amsgrad:
                    torch.maximum(max_v, v, out=max_v)
                    denom = max_v.sqrt().add_(eps)
                else:
                    denom = v.sqrt().add_(eps)

                # power_sig = m.pow(2)
                power_sig = w.pow(2)
                power_noise = v
                snr_linear = power_sig / (power_noise + 1e-10)
                snr.copy_(10.0 * torch.log10(snr_linear))

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                if self.iteration > 10:
                    beta1_t = beta1 ** self.iteration
                    beta2_t = beta2 ** self.iteration
                    alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)

                    # update = m * torch.log2(1+snr_linear.sqrt()) / denom
                    if self.option == 0:
                        update = m * torch.log2(1+snr_linear.sqrt()) / denom
                    elif self.option == 1:
                        update = m * snr_linear.sqrt() / denom
                    elif self.option == 2:
                        snr_clipped_sqrt = snr_linear.sqrt().clamp(min=0.1,max=10.0)
                        update = m * snr_clipped_sqrt / denom
                    elif self.option == 3:
                        snr_norm = snr - snr.mean()
                        snr_sqrt_linear = torch.pow(10, snr_norm / 20)
                        update = m * snr_sqrt_linear / denom
                    elif self.option == 4:
                        update = m * snr_linear
                        
                    if not self.is_lamb:
                        trust_ratio = 1.0
                    else:
                        m_norm = update.norm()
                        w_norm = p.data.norm()
                        # trust_ratio = w_norm / m_norm if m_norm > 0 and w_norm > 0 else 1.0
                        trust_ratio = w_norm / m_norm

                    if not self.is_mask:
                        p.data.sub_(alpha * trust_ratio* update)
                    else:
                        acc.add_(update)
                        cnt.add_(1.0)
                        snr_minus_threshold = snr - self.db_threshold
                        clipped_exponent = torch.clamp(-snr_minus_threshold, min=-20.0, max=30.0)
                        cnt_threshold = 10.0 ** (clipped_exponent / 10.0)
                        mask.copy_(cnt > cnt_threshold)

                        acc_applied = acc * mask.float()
                        p.data.sub_(alpha * trust_ratio * acc_applied)
                        acc.sub_(acc_applied)
                        cnt.sub_(mask.float())

        return None

    def snr_layer(self):
        snr = None
        if self.iteration > 10:
            snr = np.nanmean(self.state[self.param_groups[0]['params'][-1]]['snr'].cpu().detach().numpy())
        return snr

    def v_layer(self):
        v = None
        if self.iteration > 10:
            v = self.state[self.param_groups[0]['params'][-1]]['exp_avg_sq'].cpu().detach().numpy()
            v = 10 * np.log10(v)
            v = np.nanmean(v)
        return v
AdamPlusSNRlr = Adam5

class Adam2(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-10, weight_decay=0.0,
                 is_lamb=True, weight_decouple=False, option=0, db_noise=-160):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, db_noise=db_noise,
            weight_decouple=weight_decouple
            )
        super(Adam2, self).__init__(params, defaults)
        self.iteration = 0
        self.is_lamb = is_lamb
        self.option = option
        self.db_noise = db_noise
        self.noise_std = 10 ** (db_noise / 20)

    @torch.no_grad()
    def step(self):
        self.iteration += 1

        for group in self.param_groups:
            beta1, beta2, eps, weight_decay = group['beta1'], group['beta2'], group['eps'], group['weight_decay']
            lr = group['lr']
            weight_decouple = group.get('weight_decouple', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    if weight_decouple:
                        # Apply weight decay directly to parameters
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['emv'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['snr'] = torch.zeros_like(p.data)
                    state['mask'] = torch.zeros_like(p.data, dtype=torch.bool)
                    state['w'] = torch.zeros_like(p.data)

                m, v = state['emv'], state['exp_avg_sq']
                snr, mask = state['snr'], state['mask']
                w = state['w']

                noisy_grad = grad 
                if self.db_noise < -100:
                    noisy_grad += torch.normal(
                        mean=0.0, std=self.noise_std, size=grad.shape, device=grad.device, dtype=grad.dtype
                    )
                w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                v.mul_(beta2).addcmul_((noisy_grad - w), (noisy_grad - w), value=1 - beta2)
                grad_adjusted = noisy_grad / (v.sqrt() + eps)
                if self.option == 2:
                    grad_adjusted = torch.sign(grad_adjusted) * torch.log2(1 + torch.abs(grad_adjusted)) 
                elif self.option == 3:
                    grad_adjusted = torch.sign(grad_adjusted) * (1 + torch.log2(1 + torch.abs(grad_adjusted))) 
                else:
                    pass

                m.mul_(beta1).add_(grad_adjusted, alpha=1 - beta1)
                # u_norm = grad_adjusted.norm()
                # g_norm = noisy_grad.norm()
                # grad_scale = g_norm / u_norm if g_norm > 0 and u_norm > 0 else 1.0

                power_sig = w.pow(2)
                power_noise = v
                snr.copy_(10.0 * torch.log10(power_sig / (power_noise + 1e-10)))

                if self.iteration > 10:
                    # grad_scale = noisy_grad.norm()/grad_adjusted.norm()
                    # grad_project = grad_adjusted * grad_scale
                    # m.mul_(beta1).add_(grad_adjusted, alpha=1 - beta1)

                    beta1_t = beta1 ** self.iteration
                    beta2_t = beta2 ** self.iteration
                    alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)
                    
                    update = m

                    if self.option == 0:
                        # update = m / (v.sqrt() + eps)
                        update = m
                    elif self.option == 1:
                        update = torch.sign(m) * torch.log2(1 + torch.abs(m)) 
                    # elif self.option == 2:
                    #     update = torch.sign(m) * (1 + torch.log2(1 + torch.abs(m))) 
                    elif self.option > 3:
                        raise ValueError("unsupported option {}".format(self.option))

                    if not self.is_lamb:
                        trust_ratio = 1.0
                    else:
                        m_norm = update.norm()
                        w_norm = p.data.norm()
                        # trust_ratio = w_norm / m_norm if m_norm > 0 and w_norm > 0 else 1.0
                        trust_ratio = w_norm / m_norm

                    p.data.sub_(alpha * trust_ratio * update)


        return None

    def snr_layer(self):
        snr = None
        if self.iteration > 10:
            snr = np.nanmean(self.state[self.param_groups[0]['params'][-1]]['snr'].cpu().detach().numpy())
        return snr
ADOPTPlus = Adam2


class Adam3(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-10, weight_decay=0.0,
                 is_sign=True, weight_decouple=False, db_threshold=0, db_noise=-140):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, db_noise=db_noise,
            weight_decouple=weight_decouple
            )
        super(Adam3, self).__init__(params, defaults)
        self.iteration = 0
        self.is_sign = is_sign
        self.db_threshold = db_threshold
        self.linear_threshold = 10 ** (db_threshold / 20)
        self.noise_std = 10 ** (db_noise / 20)

    @torch.no_grad()
    def step(self):
        self.iteration += 1

        for group in self.param_groups:
            beta1, beta2, eps, weight_decay = group['beta1'], group['beta2'], group['eps'], group['weight_decay']
            lr = group['lr']
            db_noise = group['db_noise']
            weight_decouple = group.get('weight_decouple', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    if weight_decouple:
                        # Apply weight decay directly to parameters
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['snr'] = torch.zeros_like(p.data)
                    state['sign'] = torch.zeros_like(p.data, dtype=torch.bool)
                    state['mask'] = torch.zeros_like(p.data, dtype=torch.bool)
                    state['w'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']
                mask_s, snr, mask = state['sign'], state['snr'], state['mask']
                w = state['w']

                noisy_grad = grad
                if db_noise > -100:
                    noisy_grad += torch.normal(
                        mean=0.0, std=self.noise_std, size=grad.shape, device=grad.device, dtype=grad.dtype
                    )
                m.mul_(beta1).add_(noisy_grad, alpha=1 - beta1)
                w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                v.mul_(beta2).addcmul_((noisy_grad - w), (noisy_grad - w), value=1 - beta2)

                power_sig = m.pow(2)
                power_noise = v
                snr.copy_(10.0 * torch.log10(power_sig / (power_noise + 1e-10)))

                if self.iteration > 10:
                    beta1_t = beta1 ** self.iteration
                    beta2_t = beta2 ** self.iteration
                    alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)

                    update = m / (v.sqrt() + eps)

                    if not self.is_sign:
                        p.data.sub_(alpha * update)
                    else:
                        snr_minus_threshold = snr - self.db_threshold
                        mask.copy_(snr_minus_threshold >= 0)
                        mask_s.copy_(snr_minus_threshold < 0)

                        acc_applied = update * mask.float()
                        # acc_applied.add_(torch.sign(update * mask_s.float()))
                        acc_applied.add_(self.linear_threshold * torch.sign(update * mask_s.float()))
                        p.data.sub_(alpha * acc_applied)

        return None

    def snr_layer(self):
        snr = None
        if self.iteration > 10:
            snr = np.nanmean(self.state[self.param_groups[0]['params'][-1]]['snr'].cpu().detach().numpy())
        return snr



class Adam4(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-10, weight_decay=0.0,
                 weight_decouple=False, option=0, is_lamb=False,
                 ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
            weight_decouple=weight_decouple
            )
        super(Adam4, self).__init__(params, defaults)
        self.iteration = 0
        self.option = option
        self.is_lamb = is_lamb

    @torch.no_grad()
    def step(self):
        self.iteration += 1

        for group in self.param_groups:
            beta1, beta2, eps, weight_decay = group['beta1'], group['beta2'], group['eps'], group['weight_decay']
            lr = group['lr']
            weight_decouple = group.get('weight_decouple', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    if weight_decouple:
                        # Apply weight decay directly to parameters
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['snr'] = torch.zeros_like(p.data)
                    state['snr_linear'] = torch.zeros_like(p.data)
                    state['w'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']
                snr, snr_linear = state['snr'], state['snr_linear']
                w = state['w']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                w.mul_(beta2).add_(grad, alpha=1 - beta2)
                v.mul_(beta2).addcmul_((grad - w), (grad - w), value=1 - beta2)

                power_sig = m.pow(2)
                power_noise = v
                snr_linear = power_sig / (power_noise + 1e-10)
                snr.copy_(10 * torch.log10(snr_linear))

                if self.iteration > 10:
                    beta1_t = beta1 ** self.iteration
                    beta2_t = beta2 ** self.iteration
                    alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)

                    if self.option == 0:
                        # update = m / (v.sqrt() + eps)
                        update = torch.sign(m) * torch.log2(1 + torch.sqrt(snr_linear)) # 0.903 (AdamPlus_0.9)
                    elif self.option == 1:
                        update = torch.sign(m) * (1 + torch.log10(1 + torch.sqrt(snr_linear))) # 0.9
                    elif self.option == 2:
                        update = torch.sign(m) * (1 + torch.log2(1 + torch.sqrt(snr_linear))) # 0.902
                    elif self.option == 3:
                        update = torch.sign(m) * (1 + torch.log10(1 + snr_linear)) # 0.901
                    else:
                        raise ValueError("unsupported option {}".format(self.option))

                    if not self.is_lamb:
                        trust_ratio = 1.0
                    else:
                        m_norm = update.norm()
                        w_norm = p.data.norm()
                        # trust_ratio = w_norm / m_norm if m_norm > 0 and w_norm > 0 else 1.0
                        trust_ratio = w_norm / m_norm

                    p.data.sub_(alpha * trust_ratio * update)

        return None

    def snr_layer(self):
        snr = None
        if self.iteration > 10:
            snr = np.nanmean(self.state[self.param_groups[0]['params'][-1]]['snr'].cpu().detach().numpy())
        return snr
AdamPlusNL = Adam4

class Lamb(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-10, weight_decay=0.0,
                 weight_decouple=False):
        defaults = dict(
            lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay,
            weight_decouple=weight_decouple
            )
        super(Lamb, self).__init__(params, defaults)
        self.iteration = 0

    @torch.no_grad()
    def step(self):
        self.iteration += 1

        for group in self.param_groups:
            beta1, beta2, eps, weight_decay = group['beta1'], group['beta2'], group['eps'], group['weight_decay']
            lr = group['lr']
            weight_decouple = group.get('weight_decouple', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    if weight_decouple:
                        # Apply weight decay directly to parameters
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = m / (v.sqrt() + eps)

                beta1_t = beta1 ** self.iteration
                beta2_t = beta2 ** self.iteration
                alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)

                m_norm = update.norm()
                w_norm = p.data.norm()
                # trust_ratio = w_norm / m_norm if m_norm > 0 and w_norm > 0 else 1.0
                trust_ratio = w_norm / m_norm
                p.data.sub_(alpha * trust_ratio * update )


        return None


class AdamSNR(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-10, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super(AdamSNR, self).__init__(params, defaults)
        self.iteration = 0

    @torch.no_grad()
    def step(self):
        self.iteration += 1

        for group in self.param_groups:
            beta1, beta2, eps = group['beta1'], group['beta2'], group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['snr'] = torch.zeros_like(p.data)

                m, v, snr = state['exp_avg'], state['exp_avg_sq'], state['snr']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                snr.copy_(10.0 * torch.log10(m.pow(2) / (v + eps)))

                beta1_t = beta1 ** self.iteration
                beta2_t = beta2 ** self.iteration
                alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)
                update = m / (v.sqrt() + eps)
                p.data.sub_(alpha * update)

        return None

    def snr_layer(self):
        snr = None
        if self.iteration > 10:
            snr = np.nanmean(self.state[self.param_groups[0]['params'][-1]]['snr'].cpu().detach().numpy())
        return snr
    
    def v_layer(self):
        v = None
        if self.iteration > 10:
            v = self.state[self.param_groups[0]['params'][-1]]['exp_avg_sq'].cpu().detach().numpy()
            v = 10 * np.log10(v)
            v = np.nanmean(v)
        return v