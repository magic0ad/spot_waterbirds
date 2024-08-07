import cooper
import torch
from torch.distributions import Categorical

class CustomCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, reg_type="constraint", schedule=False):
        self.reg_type = reg_type
        self.schedule = schedule
        self.is_constrained = (reg_type in ["constraint", "constraint_identity"])

        super().__init__(is_constrained=self.is_constrained)

    def closure(self, model, obs, args, temperature=1., penalty_coeff=0.):
        misc = {}
        mse, slot_attn, _, _, _, _, slot_attn_list = model(obs, temperature)
        bs, num_slots, num_inputs = slot_attn_list[0].shape
        device, dtype = mse.device, mse.dtype
        if self.reg_type == "penalty_identity":
            misc['num_constraints_per_layer'] = num_slots ** 2
        else:
            misc['num_constraints_per_layer'] = (num_slots ** 2 - num_slots)
        assert args.num_iterations == len(slot_attn_list)
        misc['num_constraints'] = misc['num_constraints_per_layer'] * args.num_iterations

        misc['defects_list'] = [None] * args.num_iterations
        misc['entropies_list'] = [None] * args.num_iterations
        defects_sum = 0
        constraint_list = []
        for i in range(args.num_iterations):
            defect = torch.matmul(slot_attn_list[i], slot_attn_list[i].transpose(1, 2))
            if self.reg_type in ["penalty_identity", "constraint_identity"]:
                defect = torch.mean(torch.abs(defect - torch.eye(num_slots, dtype=dtype, device=device)), 0)
            else:
                defect = torch.mean(defect, 0) * (1 - torch.eye(num_slots, dtype=dtype, device=device))
            defects_sum = defects_sum + defect
            constraint_list.append(defect)
            misc['defects_list'][i] = defect.sum() / misc['num_constraints_per_layer']
            if not args.sa_variant:
                misc['entropies_list'][i] = Categorical(probs=slot_attn_list[i].transpose(1, 2)).entropy().mean()
            else:
                # HACK: Logging L2 norm instead of entropy
                misc['entropies_list'][i] = torch.linalg.vector_norm(slot_attn_list[i], ord=2, dim=-1).mean()
        constraint = torch.stack(constraint_list)

        misc['mse'] = mse
        misc['penalty'] = defects_sum.sum() / misc['num_constraints']

        if self.reg_type in ["constraint", "constraint_identity"]:
            return cooper.CMPState(loss=mse, ineq_defect=None, eq_defect=constraint, misc=misc)
        elif self.reg_type in ["penalty", "penalty_identity"]:
            return cooper.CMPState(loss=mse + penalty_coeff * misc['penalty'], ineq_defect=None, eq_defect=None, misc=misc)
        elif self.reg_type == 'none':
            return cooper.CMPState(loss=mse, ineq_defect=None, eq_defect=None, misc=misc)
        else:
            raise NotImplementedError(f"--reg_type {self.reg_type} is not implemented")

