import torch
import torch.optim as optim

from collections import defaultdict
import math

class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.batch_averaged = batch_averaged

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.inputs_outer = defaultdict(dict)  # |a^(l-1)><a^(l-1)|
        self.grad_ouputs_outer = defaultdict(dict)  # |delta^(l)><delta^(l)|, delta^(l) = dL / dz^(l)

        self.D_inputs = defaultdict(dict)  # D for eigenval. decomp of |z^(l-1)><z^(l-1)| = QDQ^{-1}
        self.Q_inputs = defaultdict(dict)  # Q for eigenval. decomp of |z^(l-1)><z^(l-1)| = QDQ^{-1}

        self.D_grads = defaultdict(dict)  # D for eigenval. decomp of |delta^(l)><delta^(l)| = QDQ^{-1}
        self.Q_grads = defaultdict(dict)  # Q for eigenval. decomp of |delta^(l)><delta^(l)| = QDQ^{-1}

        #         self.m_aa, self.m_gg = {}, {}
        #         self.Q_a, self.Q_g = {}, {}
        #         self.d_a, self.d_g = {}, {}

        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

    def _update_running_stat(self, new_outer, running_outer):
        # using inplace operation to save memory!
        running_outer *= self.stat_decay / (1 - self.stat_decay)
        running_outer += new_outer
        running_outer *= (1 - self.stat_decay)
        return running_outer

    def _save_input(self, module, layer_input):
        # Save z^(l-1), averaged across batch.
        # layer_input : [batch, in_channel, qubit]
        #     inputs[module] = layer_input[0]
        inp = layer_input[0]
        batch_size = inp.size(0)
        inp = inp.mean(0)
        for idx in range(module.n):
            inp_block = inp[:module.in_channels[idx], :min(idx + 1, module.n)]
            inp_block = inp_block.contiguous().view(-1, 1)
            if module.bias is not None:
                inp_block = torch.cat([inp_block, torch.ones(1, inp_block.size(-1))], 0)
            inp_outer = inp_block @ (inp_block.t() * batch_size)
            if self.steps > 0:
                running_outer = self.inputs_outer[module][idx]
            else:
                running_outer = torch.diag(inp_outer.new(inp_outer.size(0)).fill_(1))
            self.inputs_outer[module][idx] = self._update_running_stat(inp_outer, running_outer)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Save delta^(l) = dL / dz^(l), averaged across batch.
        # grad_output : [batch, out_channel, qubit]
        grad = grad_output[0]
        if (grad != grad).any().item():
            print("nan in grad")
            print(grad)
        batch_size = grad.size(0)
        grad = grad.mean(0)
        for idx in range(module.n):
            grad_block = grad[:module.out_channels[idx], [idx]]
            grad_outer = grad_block @ (grad_block.t() * batch_size)

            if self.steps > 0:
                running_outer = self.grad_ouputs_outer[module][idx]
            else:
                running_outer = torch.diag(grad_outer.new(grad_outer.size(0)).fill_(1))
            self.grad_ouputs_outer[module][idx] = self._update_running_stat(grad_outer, running_outer)

    def _prepare_model(self):
        count = 0
        #         print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.layers:
            self.modules.append(module)
            module.register_forward_pre_hook(self._save_input)
            module.register_backward_hook(self._save_grad_output)
            print('(%s): %s' % (count, module))
            count += 1

    def update_decomps(self, module, n=None):
        if n is not None:
            self._update_decomps(module, n)
        else:
            for idx in range(module.n):
                self._update_decomps(module, idx)

    def _update_decomps(self, module, n):
        eps = 1e-10  # for numerical stability
        D_inp, Q_inp = torch.symeig(self.inputs_outer[module][n], eigenvectors=True)
        try:
            D_grad, Q_grad = torch.symeig(self.grad_ouputs_outer[module][n], eigenvectors=True)
        except Exception as e:
            print(self.grad_ouputs_outer[module][n].min(),
                  self.grad_ouputs_outer[module][n].abs().min(),
                  self.grad_ouputs_outer[module][n].max())
            raise e

        D_inp.mul_((D_inp > eps).float())
        D_grad.mul_((D_grad > eps).float())

        self.D_inputs[module][n], self.Q_inputs[module][n] = D_inp, Q_inp
        self.D_grads[module][n], self.Q_grads[module][n] = D_grad, Q_grad

    def get_param_block(self, module, row_idx):
        max_in = max(module.in_channels[:row_idx + 1])
        max_out = module.out_channels[row_idx]
        params = module.weight[:max_out, :max_in, row_idx, :min(row_idx + 1, module.n)].contiguous()
        params = params.view(params.size(0), -1)
        if module.bias is not None:
            b = module.bias[:max_out, [row_idx]]
            params = torch.cat([params, b], -1)
        return params

    def get_grad_block(self, module, row_idx):
        max_in = max(module.in_channels[:row_idx + 1])
        max_out = module.out_channels[row_idx]
        g = module.weight.grad.data[:max_out, :max_in, row_idx, :min(row_idx + 1, module.n)].contiguous()
        g = g.view(g.size(0), -1)
        if module.bias is not None:
            g_bias = module.bias.grad.data[:max_out, [row_idx]]
            g = torch.cat([g, g_bias], -1)
        return g

    def set_grad_block(self, module, row_idx, new_grads, nu=1):
        max_in = max(module.in_channels[:row_idx + 1])
        max_out = module.out_channels[row_idx]

        if module.bias is not None:
            w_grads = new_grads[..., :-1]
            b_grads = new_grads[..., -1]
            module.bias.grad[:max_out, row_idx] = b_grads
            module.bias.grad.data.mul_(nu)
        else:
            w_grads = new_grads

        w_grads = w_grads.view(max_out, max_in, 1, min(row_idx + 1, module.n))

        module.weight.grad[:max_out, :max_in, [row_idx], :min(row_idx + 1, module.n)] = w_grads
        module.weight.grad.data.mul_(nu)

    def get_natural_grad(self, module, n=None, damping=0.001):
        # grads are first order gradients from .backward
        if n is not None:
            grads = self.get_grad_block(module, n)
            grads = self._get_natural_grad(module, n, grads, damping)
        else:
            grads = [self._get_natural_grad(module, idx, self.get_grad_block(module, idx))
                     for idx in range(module.n)]
        return grads

    def _get_natural_grad(self, module, n, grads, damping=0.001):
        # grads are first order gradients from .backward
        # Identity 1 : (A \otimes B) * vec(X) = vec(B X A^T).
        # Identity 2 : If A = QDQ^{-1}, A^{-1} = Q^{-1}D^{-1}Q.
        #              for QDQ^{-1} as eigendecomposition, Q^{-1}=Q^{T}, D^{-1}_{ii}=1/D_{ii}.
        D_A, Q_A = self.D_inputs[module][n], self.Q_inputs[module][n]
        D_B, Q_B = self.D_grads[module][n], self.Q_grads[module][n]

        grads =  Q_B.t() @ grads @ Q_A
        grads = grads / (D_B.unsqueeze(1) * D_A.unsqueeze(0) + damping)
        grads = Q_B @ grads @ Q_A.t()

        return grads

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for module in self.modules:
            for idx in range(module.n):
                v = updates[module][idx]
                g = self.get_grad_block(module, idx)
                vg_sum += (v[..., -1:] * g[..., -1:] * lr ** 2).sum().item()
                if module.bias is not None:
                    vg_sum += (v[..., -1] * g[..., -1] * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for module in self.modules:
            for idx in range(module.n):
                v = updates[module][idx]
                self.set_grad_block(module, idx, v, nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = defaultdict(dict)
        for module in self.modules:
            for idx in range(module.n):
                if self.steps % self.TInv == 0:
                    self.update_decomps(module)
                grads = self.get_natural_grad(module, idx, damping)
                updates[module][idx] = grads
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1