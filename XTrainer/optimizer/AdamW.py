import math
import torch
from .OptimizerBase import OptimizerBase
from .build import OPTIMIZER_REGISTRY

@OPTIMIZER_REGISTRY.register()
class AdamW(OptimizerBase):
    """ AdamW Optimizer """        
    def __init__(self, cfg, params=None):
        """ Initialization method 
        
        Args:
            - cfg (CfgNode): Configuration
            - params (iterable): Model parameters

        Configuration:
            - Default optimizer parameters
                - OPTIMIZER.LR (float): Learning rate
                - OPTIMIZER.betas (Tuple[float, float]): Beta parameters for Adam
                - OPTIMIZER.eps (float): Constant for numerical stability
                - OPTIMIZER.weight_decay (float): Weight decay
                - OPTIMIZER.warmup_steps (int): Warmup steps

        Main steps:
            - Read configuration
            - Validate parameters
            - Pass default parameters to the parent class
        """

        # ----Read configuration-----
        # Read default optimizer parameters
        lr = float(cfg.OPTIMIZER.LR)
        betas = list(map(float, cfg.OPTIMIZER.betas))
        eps = float(cfg.OPTIMIZER.eps)
        weight_decay = float(cfg.OPTIMIZER.weight_decay)
        warmup_steps = int(cfg.OPTIMIZER.warmup_steps)

        # ----Validate parameters-----
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup=warmup_steps
        )
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(
                        p_data_fp32
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1**state["step"]
                bias_correction2 = 1 - beta2**state["step"]

                if group["warmup"] > state["step"]:
                    scheduled_lr = 1e-8 + state["step"] * group["lr"] / group[
                        "warmup"]
                else:
                    scheduled_lr = group["lr"]

                step_size = (
                    scheduled_lr * math.sqrt(bias_correction2) /
                    bias_correction1
                )

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        -group["weight_decay"] * scheduled_lr, p_data_fp32
                    )

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
