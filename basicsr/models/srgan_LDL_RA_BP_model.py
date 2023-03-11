import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel
from basicsr.losses.LDL_loss import get_refined_artifact_map
from ..utils import region_seperator, resizer


@MODEL_REGISTRY.register()
class SRGAN_LDL_RA_BP(SRGANModel):

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_g_ema.parameters():
            p.requires_grad = False

        # update G first, then D
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        self.output_ema = self.net_g_ema(self.lq)

        # Region-Aware
        flat_mask = 0.
        if train_opt.get('region_aware'):
            flat_mask = region_seperator.get_flat_mask(self.gt,
                                                       kernel_size=train_opt['region_aware']['k_size'],
                                                       std_thresh=train_opt['region_aware']['std'],
                                                       scale=self.opt['scale'])
        output_det = self.output * (1 - flat_mask)  # SR细节部分，需进行对抗学习
        hr_det = self.gt * (1 - flat_mask)  # HR细节部分,对抗学习的真实图像

        l_g_total = 0.
        loss_dict = OrderedDict()
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix  # l_g_pix: 生成SR的像素（L1）损失
            if self.cri_artifacts:
                pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)  # 伪影惩罚系数：M_refine
                l_g_artifacts = self.cri_artifacts(torch.mul(pixel_weight, self.output),
                                                   torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_artifacts
                loss_dict['l_g_artifacts'] = l_g_artifacts  # l_g_artifacts: 伪影惩罚损失，L1损失
            if self.back_projection:
                bp_lr_img = resizer.imresize(self.output, scale=1 / self.opt['scale'])
                l_g_bp = self.back_projection(bp_lr_img, self.lq)
                l_g_total += l_g_bp
                loss_dict['l_g_bp'] = l_g_bp
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep  # l_g_percep：感知损失
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # gan loss (relativistic gan)
            # real_d_pred = self.net_d(self.gt).detach()
            # fake_g_pred = self.net_d(self.output)
            real_d_pred = self.net_d(hr_det).detach()
            fake_g_pred = self.net_d(output_det)
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        # real
        # fake_d_pred = self.net_d(self.output).detach()
        # real_d_pred = self.net_d(self.gt)
        fake_d_pred = self.net_d(output_det).detach()
        real_d_pred = self.net_d(hr_det)
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        # fake_d_pred = self.net_d(self.output.detach())
        fake_d_pred = self.net_d(output_det.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
