import math
import torch
from torch.optim import Optimizer
import numpy as np
from scipy.signal import butter, freqz
import scipy.optimize as opt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AdamPlusv2(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 is_mask=False, is_lamb=False, weight_decouple=False,
                 db_threshold=50, db_noise=-140, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, weight_decouple=weight_decouple,
                        amsgrad=amsgrad)
        super(AdamPlusv2, self).__init__(params, defaults)
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
                # if beta1 != beta2:
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
                # if beta1 != beta2:
                w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                v.mul_(beta2).addcmul_(noisy_grad, noisy_grad, value=1 - beta2)
                # else:
                    # v.mul_(beta2).addcmul_(noisy_grad, noisy_grad, value=1 - beta2)

                # if amsgrad:
                    # torch.maximum(max_v, v, out=max_v)
                    # denom = max_v.sqrt().add_(eps)
                # else:
                    # if beta1 != beta2:
                    # v.addcmul(w, w, value=-1).clamp_min(0)
                        # v.sqrt()._add(eps)
                        # denom_sq = torch.clamp(denom_sq, min=0)
                        # denom = torch.sqrt(denom_sq) + eps
                    # else:
                        # denom = v.addcmul(m, m, value=-1).sqrt()

                power_sig = m.pow(2)
                power_noise = v
                snr.copy_(10.0 * torch.log10(power_sig / (power_noise + 1e-10)))

                if self.iteration > 0:
                    beta1_t = beta1 ** self.iteration
                    beta2_t = beta2 ** self.iteration
                    alpha = lr #* math.sqrt(1 - beta2_t) / (1 - beta1_t)
                    m_hat = m/(1-beta1_t)
                    w_hat = w/(1-beta2_t)
                    v_hat = v/(1-beta2_t)

                    var_hat = v_hat - w_hat.pow(2)
                    var_hat = var_hat.clamp_min(0)
                    denom = torch.sqrt(var_hat).add(eps)

                    # update = m / denom
                    update = m_hat / denom
                    if not self.is_lamb:
                        trust_ratio = 1.0
                    else:
                        m_norm = update.norm()
                        w_norm = p.data.norm()
                        # trust_ratio = w_norm / m_norm if m_norm > 0 and w_norm > 0 else 1.0
                        trust_ratio = w_norm / m_norm

                    if not self.is_mask:
                        p.data.sub_(alpha * trust_ratio * update)
                    else:
                        acc.add_(update)
                        cnt.add_(1.0)
                        snr_minus_threshold = snr - self.db_threshold
                        clipped_exponent = torch.clamp(-snr_minus_threshold, min=-20.0, max=30.0)
                        cnt_threshold = 10.0 ** (clipped_exponent / 10.0)
                        mask.copy_(cnt > cnt_threshold)

                        acc_applied = acc * mask.float()
                        cnt_applied = cnt * mask.float()
                        p.data.sub_(alpha * trust_ratio * acc_applied)
                        acc.sub_(acc_applied)
                        cnt.sub_(cnt_applied)
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
    

class AdamPlus(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 is_mask=True, is_lamb=False, weight_decouple=False, is_fir=False, mask_opt=0,
                 db_threshold=-3, db_noise=-80, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, weight_decouple=weight_decouple,
                        amsgrad=amsgrad)
        super(AdamPlus, self).__init__(params, defaults)
        self.iteration = 0
        self.is_mask = is_mask
        self.is_lamb = is_lamb
        self.is_fir = is_fir
        self.mask_opt=mask_opt
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
                    state['snr_avg'] = 1.0
                    if self.is_mask:
                        state['acc'] = torch.zeros_like(p.data)
                        state['cnt'] = torch.zeros_like(p.data)
                    if beta1 != beta2:
                        state['w'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.is_fir:
                        state['g1'] = torch.zeros_like(p.data)


                m, v = state['exp_avg'], state['exp_avg_sq']
                snr = state['snr']
                if self.is_mask:
                    acc, cnt = state['acc'], state['cnt']
                if beta1 != beta2:
                    w = state['w']
                if amsgrad:
                    max_v = state['max_exp_avg_sq']
                if self.is_fir:
                    g1 = state['g1']

                noisy_grad = grad
                if self.db_noise > -100:
                    noisy_grad += torch.normal(
                        mean=0.0, std=self.noise_std, size=grad.shape, device=grad.device, dtype=grad.dtype
                    )
                    # noisy_grad += self.noise_std * (torch.rand(
                    #     size=grad.shape, device=grad.device, dtype=grad.dtype
                    # ) - 0.5)
                # m.mul_(beta1).add_(noisy_grad, alpha=1 - beta1)
                if self.is_fir:
                    x0 = 0.5*grad+0.5*g1
                    m.mul_(beta1).add_(x0, alpha=1 - beta1)
                    g1.copy_(grad)
                else:
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                if beta1 != beta2:
                    w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                    v.mul_(beta2).add_((noisy_grad - w).pow(2), alpha=1 - beta2)
                else:
                    v.mul_(beta2).add_((noisy_grad - m).pow(2), alpha=1 - beta2)
                    # v.mul_(beta2).addcmul_((noisy_grad - m), (noisy_grad - m), value=1 - beta2)

                if amsgrad:
                    torch.maximum(max_v, v, out=max_v)
                    denom = max_v.sqrt().add_(eps)
                else:
                    denom = v.sqrt().add_(eps)

                power_sig = m.pow(2)
                power_noise = v
                snr_t = 10.0 * torch.log10(power_sig / (power_noise + 1e-10))
                state['snr_avg'] = beta2 * state['snr_avg'] + (1 - beta2) * snr_t.mean().cpu().item()
                snr_avg = state['snr_avg']

                if self.iteration > 1:
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
                        if self.mask_opt == 0:
                            snr_minus_threshold = snr_t - self.db_threshold
                        else: 
                            snr_minus_threshold = snr_t - np.amin([snr_avg + self.db_threshold, 0.0])
                        clipped_exponent = torch.clamp(-snr_minus_threshold, min=-20.0, max=20.0)
                        cnt_threshold = 10.0 ** (clipped_exponent / 10.0)
                        if self.mask_opt == 2:
                            mask = (snr_t > snr).float()
                        else:
                            mask = (cnt > cnt_threshold).float()

                        acc_applied = acc * mask
                        p.data.sub_(alpha * trust_ratio * acc_applied)
                        acc.sub_(acc_applied)
                        cnt.sub_(cnt * mask)

                snr.copy_(snr_t)


        return None

    def snr_layer(self):
        snr = None
        if self.iteration > 1:
            snr = np.nanmean(self.state[self.param_groups[0]['params'][-1]]['snr'].cpu().detach().numpy())
        return snr



class Adam5(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0,
                 is_mask=False, is_lamb=False, weight_decouple=False, option=0,
                 db_threshold=10, db_noise=-80, amsgrad=False):
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
                    state['snr'] = torch.ones_like(p.data)
                    state['snr_avg'] = 1.0
                    if self.is_mask:
                        state['acc'] = torch.zeros_like(p.data)
                        state['cnt'] = torch.zeros_like(p.data)
                        state['mask'] = torch.zeros_like(p.data, dtype=torch.bool)
                    # state['x1'] = torch.zeros_like(p.data)
                    # state['x2'] = torch.zeros_like(p.data)
                    # state['y1'] = torch.zeros_like(p.data)
                    # state['y2'] = torch.zeros_like(p.data)
                    state['m1'] = torch.zeros_like(p.data)
                    state['v1'] = torch.zeros_like(p.data)
                    state['w'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']
                snr = state['snr']
                g1, v1 = state['m1'], state['v1']
                # x1, x2, y1, y2 = state['x1'], state['x2'], state['y1'], state['y2']
                if self.is_mask:
                    acc, cnt, mask = state['acc'], state['cnt'], state['mask']
                # acc, cnt, snr, mask = state['acc'], state['cnt'], state['snr'], state['mask']
                w = state['w']
                if amsgrad:
                    max_v = state['max_exp_avg_sq']

                # noise_pwr_tensor = 10.0 * torch.log10(v)
                # noise_pwr = np.nanmean(noise_pwr_tensor.detach().cpu().item()) - 15.0
                # if np.isnan(noise_pwr):
                #     noise_pwr = -30
                # else:
                #     noise_pwr = np.amin([-30, noise_pwr])
                # noise_std = 10 ** (noise_pwr / 20)

                noisy_grad = grad
                noise = torch.normal(
                        mean=0.0, std=self.noise_std, size=grad.shape, device=grad.device, dtype=grad.dtype
                    )
                # if self.db_noise > -130:
                noisy_grad += noise
                # v.mul_(beta2).addcmul_(noisy_grad - w, noisy_grad - w, value=1 - beta2)
                # v.copy_(self.butt_iir.update( (noisy_grad - w).pow(2) ))
                # 2nd order Butterworth IIR filter
                # x2.copy_(x1)
                if self.option in [3, 5]:
                    g0 = 0.5*grad+0.5*g1
                    m.mul_(beta1).add_(g0, alpha=1 - beta1)
                    v1.copy_(g0)
                    g1.copy_(grad)
                    w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                    v.mul_(beta2).add_((noisy_grad - w).pow(2), alpha=1 - beta2)
                else:
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                    v.mul_(beta2).add_((noisy_grad - w).pow(2), alpha=1 - beta2)

                # v.mul_(beta2).add_(10.0*torch.log10((noisy_grad - w).pow(2) + eps), alpha=1 - beta2)

                # m.mul_(beta1).add_(grad, alpha=1 - beta1)

                if amsgrad:
                    torch.maximum(max_v, v, out=max_v)
                    denom = max_v.sqrt().add_(eps)
                else:
                    denom = v.sqrt().add_(eps)

                power_sig = m.pow(2)
                power_noise = v
                # power_noise = exp_v.pow(2)
                snr_linear = power_sig / (power_noise + eps)
                snr.copy_(10.0 * torch.log10(snr_linear))
                state['snr_avg'] = beta2 * state['snr_avg'] + (1 - beta2) * snr.mean().cpu().item()
                snr_avg = state['snr_avg']


                # snr_delta = snr - snr_ema
                # snr_delta_linear = torch.pow(10, snr_delta/10)
                # snr_delta_linear = snr_linear / snr_ema
                # Flatten and compute threshold
                # flat_update = snr_linear.view(-1)
                # k = int(flat_update.numel() * 0.4)
                # # Get top-k indices by absolute value
                # _, topk_indices = torch.topk(flat_update.abs(), k, sorted=False)
                # mask = torch.zeros_like(flat_update)
                # mask[topk_indices] = 1.0
                # mask = mask.view_as(snr_linear)
                lr_scale = 1.0
                if self.iteration > 1:
                    beta1_t = beta1 ** self.iteration
                    beta2_t = beta2 ** self.iteration
                    alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)

                    # update = m * torch.log2(1+snr_linear.sqrt()) / denom
                    if self.option == 0:
                        update = m * torch.log2(1+snr_linear.sqrt()) / denom
                    elif self.option == 1:
                        update = m * snr_linear.sqrt() / denom
                    elif self.option == 2:
                        # snr_clipped_sqrt = snr_linear.sqrt().clamp(min=1.0,max=10.0)
                        snr_clipped_sqrt = snr_linear.sqrt().clamp(min=1.0)
                        update = m * snr_clipped_sqrt / denom
                    elif self.option in [4,5]:
                        # lr_scale = 0.2+0.8*2*torch.sigmoid(
                        #     (snr - snr_avg - 3.0)*0.1
                        # )
                        lr_scale = 0.5+torch.sigmoid(
                            (snr - snr_avg)*0.2
                        )
                        update = m / denom
                        # update = m / denom
                    else:
                        update = m / denom
                        
                    if not self.is_lamb:
                        trust_ratio = 1.0 * lr_scale
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
                        # snr_minus_threshold = snr - np.amin([snr_avg + self.db_threshold, 0.0])
                        snr_minus_threshold = snr - self.db_threshold
                        clipped_exponent = torch.clamp(-snr_minus_threshold, min=-20.0, max=20.0)
                        cnt_threshold = 10.0 ** (clipped_exponent / 10.0)
                        mask.copy_(cnt > cnt_threshold)

                        acc_applied = acc * mask.float() 
                        p.data.sub_(alpha * trust_ratio * acc_applied)
                        acc.sub_(acc_applied)
                        cnt.sub_(cnt * mask.float())


        return None

    @torch.no_grad()
    def snr_track(self):
        result = {}
        g_id = 0
        p_id = 0
        # param_to_name = {param: name for name, param in model.named_parameters()}

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                v, snr = state['exp_avg_sq'], state['snr']
                # name = param_to_name.get(p, None)

                v_db = 10.0 * torch.log10(v + 1e-6)
                # v_db = v 
                # v_db_avg = np.nanmean(v_db.cpu().detach().numpy())
                # snr_avg = np.nanmean(snr.cpu().detach().numpy())
                snr_np = snr.cpu().detach().numpy()
                v_db_np = v_db.cpu().detach().numpy()

                if np.isfinite(snr_np).any():
                    snr_avg = np.nanmean(snr_np)
                else:
                    snr_avg = np.nan  # or 0.0, depending on what you want

                if np.isfinite(v_db_np).any():
                    v_db_avg = np.nanmean(v_db_np)
                else:
                    v_db_avg = np.nan

                l_id = "l{}_{}".format(g_id, p_id)    
                result[l_id + '_snr'] = snr_avg
                result[l_id + '_vdb'] = v_db_avg

                p_id += 1
            g_id += 1

        return result



class Adam6(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=0.03, weight_decay=0.0,
                 is_mask=False, is_lamb=False, weight_decouple=False, option=0,
                 db_threshold=-10, db_noise=-80, amsgrad=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2)
        super(Adam6, self).__init__(params, defaults)
        self.iteration = 0
        self.is_mask = is_mask
        self.is_lamb = is_lamb
        self.db_noise = db_noise
        self.db_threshold = db_threshold
        self.noise_std = 10 ** (db_noise / 20)
        self.option = option
        self.eps = eps
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.amsgrad = amsgrad

    @torch.no_grad()
    def step(self):
        self.iteration += 1

        for group in self.param_groups:
            beta1, beta2 = group['beta1'], group['beta2']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if self.weight_decay != 0:
                    if self.weight_decouple:
                        p.data.mul_(1 - lr * self.weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=self.weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['snr'] = 0.0
                    state['snr_avg'] = 1.0
                    state['g1'] = torch.zeros_like(p.data)
                    state['v1'] = torch.zeros_like(p.data)
                    state['w'] = torch.zeros_like(p.data)
                    if self.is_mask:
                        state['acc'] = torch.zeros_like(p.data)
                        state['cnt'] = 0.0
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']
                # snr = state['snr']
                w = state['w']
                g1, v1 = state['g1'], state['v1']
                if self.is_mask:
                    acc = state['acc']
                if self.amsgrad:
                    max_v = state['max_exp_avg_sq']

                noisy_grad = grad
                if self.db_noise > -100:
                    noise = torch.normal(
                            mean=0.0, std=self.noise_std, size=grad.shape, device=grad.device, dtype=grad.dtype
                        )
                    noisy_grad += noise
                if self.option in [1]:
                    g0 = 0.5*grad+0.5*g1
                    m.mul_(beta1).add_(g0, alpha=1 - beta1)
                    v1.copy_(g0)
                    g1.copy_(grad)
                else:
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                w.mul_(beta2).add_(noisy_grad, alpha=1 - beta2)
                v.mul_(beta2).add_((noisy_grad - w).pow(2), alpha=1 - beta2)


                if self.amsgrad:
                    torch.maximum(max_v, v, out=max_v)
                    denom = max_v.sqrt().add_(self.eps)
                else:
                    denom = v.sqrt().add_(self.eps)

                power_sig = m.pow(2)
                power_noise = v
                snr_linear = power_sig / (power_noise + self.eps)
                state['snr'] = 10.0*torch.log10(snr_linear).mean().cpu().item()
                # snr.copy_(10.0 * torch.log10(snr_linear))
                state['snr_avg'] = beta2 * state['snr_avg'] + (1 - beta2) * state['snr']
                snr_avg = state['snr_avg']
                snr = state['snr']

                if self.iteration > 1:
                    beta1_t = beta1 ** self.iteration
                    beta2_t = beta2 ** self.iteration
                    alpha = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)
                    lr_scale = 1.0

                    if self.option == 1:
                        lr_scale = 0.5 + sigmoid((snr - snr_avg) * 0.2)

                    update = m / denom
                        
                    if not self.is_lamb:
                        trust_ratio = 1.0 * lr_scale
                    else:
                        m_norm = update.norm()
                        w_norm = p.data.norm()
                        # trust_ratio = w_norm / m_norm if m_norm > 0 and w_norm > 0 else 1.0
                        trust_ratio = w_norm / m_norm

                    if not self.is_mask:
                        p.data.sub_(alpha * trust_ratio* update)
                    else:
                        acc.add_(update)
                        state['cnt'] += 1.0
                        # snr_minus_threshold = snr - np.amin([snr_avg + self.db_threshold, 0.0])
                        snr_minus_threshold = snr - np.amin([self.db_threshold, 3.0])
                        clipped_exponent = np.clip(-snr_minus_threshold, -20.0, 20.0)
                        cnt_threshold = 10.0 ** (clipped_exponent / 10.0)
                        mask = (state['cnt'] > cnt_threshold)  # boolean scalar or array
                        # convert boolean to float (0.0 or 1.0)
                        mask_float = float(mask)  

                        acc_applied = acc * mask_float
                        p.data.sub_(alpha * trust_ratio * acc_applied)
                        acc.sub_(acc_applied)
                        state['cnt'] -= state['cnt'] * mask_float


        return None

    @torch.no_grad()
    def snr_track(self):
        result = {}
        g_id = 0
        p_id = 0
        # param_to_name = {param: name for name, param in model.named_parameters()}

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                v, snr = state['exp_avg_sq'], state['snr']
                # name = param_to_name.get(p, None)

                v_db = 10.0 * torch.log10(v + 1e-6)
                # v_db = v 
                # v_db_avg = np.nanmean(v_db.cpu().detach().numpy())
                # snr_avg = np.nanmean(snr.cpu().detach().numpy())
                snr_np = snr.cpu().detach().numpy()
                v_db_np = v_db.cpu().detach().numpy()

                if np.isfinite(snr_np).any():
                    snr_avg = np.nanmean(snr_np)
                else:
                    snr_avg = np.nan  # or 0.0, depending on what you want

                if np.isfinite(v_db_np).any():
                    v_db_avg = np.nanmean(v_db_np)
                else:
                    v_db_avg = np.nan

                l_id = "l{}_{}".format(g_id, p_id)    
                result[l_id + '_snr'] = snr_avg
                result[l_id + '_vdb'] = v_db_avg

                p_id += 1
            g_id += 1

        return result


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

