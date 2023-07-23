import torch.nn.functional as F

from model.base_arch import SingleScalePlainCNN
from utils.build import BACKBONE_REGISTRY, META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class DenoiseRSCArch(SingleScalePlainCNN):
    def postprocess(self, *outputs):
        return outputs[0][0][0]

    def update_params(self, batch_data, optimizer):
        # forward to generate model outputs
        model_outputs = self.model(self.preprocess(batch_data))

        # calculate losses
        gt_frames = batch_data["gt_frames"].to(self.device).flatten(0, 1)
        out_frames = model_outputs[0]  # get the output multi-scale images
        offsets = model_outputs[1]
        denoise_loss = model_outputs[2]

        loss_dict = {k: 0. for k in self.criterion.keys()}
        gt_frames_multi_level = [gt_frames]
        for i in range(1, len(out_frames)):
            gt_frames_multi_level.append(
                F.interpolate(gt_frames_multi_level[-1], scale_factor=0.5))

        for i in range(len(out_frames)):
            # print(gt_frames_multi_level[i].shape, out_frames[i].shape)
            loss_dict["CharbonnierLoss"] += self.criterion["CharbonnierLoss"](
                gt_frames_multi_level[i], out_frames[i]) * self.criterion_weights["CharbonnierLoss"]
            loss_dict["PerceptualLossVGG19"] += self.criterion["PerceptualLossVGG19"](
                gt_frames_multi_level[i], out_frames[i]) * self.criterion_weights["PerceptualLossVGG19"]
            loss_dict["VariationLoss"] += self.criterion["VariationLoss"](
                offsets[i]) * self.criterion_weights["VariationLoss"]
            loss_dict["FlowDistillationLoss"] += self.criterion["FlowDistillationLoss"](
                gt_frames_multi_level[i], self.preprocess(batch_data), offsets[i]) * self.criterion_weights["FlowDistillationLoss"]
            loss_dict["ZeroShotDenoiseLoss"] += denoise_loss * self.criterion_weights["ZeroShotDenoiseLoss"]

        loss = sum(loss_dict.values())
        # 2 optimize model parameters:
        # a) zero_grad, b) backward, c) update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "results": model_outputs,
            "loss": loss_dict
        }
