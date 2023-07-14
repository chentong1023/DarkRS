import torch.nn.functional as F

from model.base_arch import SingleScalePlainCNN
from utils.build import BACKBONE_REGISTRY, META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class VIArch(SingleScalePlainCNN):
    def postprocess(self, *outputs):
        return outputs[0][0][0]

    def update_params(self, batch_data, optimizer):
        # forward to generate model outputs
        gt_frames = batch_data["gt_frames"].to(self.device).flatten(0, 1)
        model_outputs = self.model(self.preprocess(batch_data), gt_frames)

        # calculate losses
        out_frames, feat_pred, feat_gt, flow_0, flow_1 = model_outputs  # get the output multi-scale images

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

        for i in range(len(feat_pred)):
            loss_dict["GeometryLoss"] += self.criterion["GeometryLoss"](
                feat_pred[i], feat_gt[i]) * self.criterion_weights["GeometryLoss"]

        for i in range(len(flow_0)):
            loss_dict["VariationLoss"] += self.criterion["VariationLoss"](
                flow_0[i]) * self.criterion_weights["VariationLoss"]
            loss_dict["VariationLoss"] += self.criterion["VariationLoss"](
                flow_1[i]) * self.criterion_weights["VariationLoss"]
        loss_dict["VIFlowDistillationLoss"] += self.criterion["VIFlowDistillationLoss"](
            gt_frames_multi_level[0], self.preprocess(batch_data), flow_0, flow_1) * self.criterion_weights["VIFlowDistillationLoss"]

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
