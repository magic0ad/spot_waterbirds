import cooper
import torch

class CustomCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, reg_type="constraint", schedule=False, reg_coeff=0.1):
        self.reg_type = reg_type
        self.schedule = schedule
        self.is_constrained = (reg_type == "constraint")
        self.reg_coeff = reg_coeff

        super().__init__(is_constrained=self.is_constrained)

    def closure(self, model, obs, args):
        misc = {}
        mse, slots_attns, _, _, _, _ = model(obs)
        if self.reg_type == "none":
            return cooper.CMPState(loss=mse, ineq_defect=None, eq_defect=None, misc=misc)
        elif self.reg_type == "constraint":
            bs, num_slots, h, w = slots_attns.shape
            device, dtype = mse.device, mse.dtype
            slots_attns = slots_attns.reshape(bs, num_slots, -1)
            defects = torch.matmul(slots_attns, torch.permute(slots_attns, (0, 2, 1)))
            defects = torch.mean(defects, 0) * (1 - torch.eye(num_slots, dtype=dtype, device=device))
            return cooper.CMPState(loss=mse, ineq_defect=None, eq_defect=defects, misc=misc)
        else:
            raise NotImplementedError(f"--reg_type {self.reg_type} is not implemented")

