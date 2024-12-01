import torch
import sys
import importlib
import os
from omegaconf import OmegaConf
import numpy as np
from torch import autocast
from contextlib import nullcontext
import copy

root = os.environ.get("STYLE_ROOT")
sys.path.append(root)
sys.path.append(f"{root}/style-transfer-modules/StyleID")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from NSTModule import NSTModule


class StyleIDNSTModule(NSTModule):
    def __init__(self):
        super().__init__("StyleID")

        self.ddim_inversion_steps = 50
        self.save_feature_timesteps = 50
        self.ddim_steps = self.save_feature_timesteps
        self.start_step = 49
        self.ddim_eta = 0.0
        self.T = 5.0
        self.gamma = 0.75
        self.attn_layer = "6,7,8,9,10,11"
        self.self_attn_output_block_indices = list(map(int, self.attn_layer.split(",")))
        self.model_config = "StyleID/models/ldm/stable-diffusion-v1/v1-inference.yaml"
        self.ckpt = "StyleID/models/ldm/stable-diffusion-v1/model.ckpt"
        self.precision = "autocast"
        self.precision_scope = autocast if self.precision == "autocast" else nullcontext
        self.without_init_adain = False
        self.without_attn_injection = False

        self.feat_maps = [
            {"config": {"gamma": self.gamma, "T": self.T}} for _ in range(50)
        ]

    def get_obj_from_str(self, string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)

    def instantiate_from_config(self, config):
        if not "target" in config:
            if config == "__is_first_stage__":
                return None
            elif config == "__is_unconditional__":
                return None
            raise KeyError("Expected key `target` to instantiate.")
        return self.get_obj_from_str(config["target"])(**config.get("params", dict()))

    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = self.instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        return model

    def ddim_sampler_callback(self, pred_x0, xt, i):
        self.save_feature_maps_callback(i)
        self.save_feature_map(xt, "z_enc", i)

    def save_feature_maps(self, blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self.self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    self.save_feature_map(
                        q, f"{feature_type}_{block_idx}_self_attn_q", i
                    )
                    self.save_feature_map(
                        k, f"{feature_type}_{block_idx}_self_attn_k", i
                    )
                    self.save_feature_map(
                        v, f"{feature_type}_{block_idx}_self_attn_v", i
                    )
            block_idx += 1

    def save_feature_maps_callback(self, i):
        self.save_feature_maps(
            self.model["unet_model"].output_blocks, i, "output_block"
        )

    def save_feature_map(self, feature_map, filename, time):
        cur_idx = self.idx_time_dict[time]
        self.feat_maps[cur_idx][f"{filename}"] = feature_map

    def _load_model(self):
        config = OmegaConf.load(f"{self.model_config}")
        self.model["model"] = self.load_model_from_config(config, f"{self.ckpt}").to(self.device)
        self.model["unet_model"] = self.model["model"].model.diffusion_model.to(self.device)

        self.sampler = DDIMSampler(self.model["model"])
        self.sampler.make_schedule(
            ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False
        )
        time_range = np.flip(self.sampler.ddim_timesteps)
        self.idx_time_dict = {}
        self.time_idx_dict = {}
        for i, t in enumerate(time_range):
            self.idx_time_dict[t] = i
            self.time_idx_dict[i] = t

        self.uc = self.model["model"].get_learned_conditioning([""])

    def _preprocess_content_image(self, image):
        scaled_image = 2.0 * image + 1
        super()._preprocess_content_image(scaled_image)
        init_cnt = self.model["model"].get_first_stage_encoding(
            self.model["model"].encode_first_stage(self.content_image)
        )
        cnt_z_enc, _ = self.sampler.encode_ddim(
            init_cnt.clone(),
            num_steps=self.ddim_inversion_steps,
            unconditional_conditioning=self.uc,
            end_step=self.time_idx_dict[
                self.ddim_inversion_steps - 1 - self.start_step
            ],
            callback_ddim_timesteps=self.save_feature_timesteps,
            img_callback=self.ddim_sampler_callback,
        )
        self.cnt_feat = copy.deepcopy(self.feat_maps)
        self.cnt_z_enc = self.feat_maps[0]["z_enc"]

    def _preprocess_style_image(self, image):
        scaled_image = 2.0 * image + 1
        super()._preprocess_style_image(scaled_image)
        init_sty = self.model["model"].get_first_stage_encoding(
            self.model["model"].encode_first_stage(self.style_image)
        )
        sty_z_enc, _ = self.sampler.encode_ddim(
            init_sty.clone(),
            num_steps=self.ddim_inversion_steps,
            unconditional_conditioning=self.uc,
            end_step=self.time_idx_dict[
                self.ddim_inversion_steps - 1 - self.start_step
            ],
            callback_ddim_timesteps=self.save_feature_timesteps,
            img_callback=self.ddim_sampler_callback,
        )
        self.sty_feat = copy.deepcopy(self.feat_maps)
        self.sty_z_enc = self.feat_maps[0]["z_enc"]

    def adain(self, cnt_feat, sty_feat):
        cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
        cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
        sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
        sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
        output = ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean
        return output

    def feat_merge(self, cnt_feats, sty_feats, start_step=0):
        self.feat_maps = [
            {"config": {"gamma": self.gamma, "T": self.T, "timestep": _,}}
            for _ in range(50)
        ]

        for i in range(len(self.feat_maps)):
            if i < (50 - start_step):
                continue
            cnt_feat = cnt_feats[i]
            sty_feat = sty_feats[i]
            ori_keys = sty_feat.keys()

            for ori_key in ori_keys:
                if ori_key[-1] == "q":
                    self.feat_maps[i][ori_key] = cnt_feat[ori_key]
                if ori_key[-1] == "k" or ori_key[-1] == "v":
                    self.feat_maps[i][ori_key] = sty_feat[ori_key]
        return self.feat_maps

    def _transfer_style(self, content_image, style_image):
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model["model"].ema_scope():
                    if self.without_init_adain:
                        adain_z_enc = self.cnt_z_enc
                    else:
                        adain_z_enc = self.adain(self.cnt_z_enc, self.sty_z_enc)
                    self.feat_maps = self.feat_merge(
                        self.cnt_feat, self.sty_feat, start_step=self.start_step
                    )
                    if self.without_attn_injection:
                        self.feat_maps = None
                    
                    # inference
                    samples_ddim, intermediates = self.sampler.sample(
                        S=self.ddim_steps,
                        batch_size=1,
                        shape=self.content_image.squeeze().shape,
                        verbose=False,
                        unconditional_conditioning=self.uc,
                        eta=self.ddim_eta,
                        x_T=adain_z_enc,
                        injected_features=self.feat_maps,
                        start_step=self.start_step,
                    )

                    x_samples_ddim = self.model['model'].decode_first_stage(samples_ddim)
                    
                    return x_samples_ddim
    
    def _postprocess_output_image(self, image, content_name, style_name):
        image = (image + 1.0) / 2.0
        return super()._postprocess_output_image(image, content_name, style_name)


if __name__ == "__main__":
    sidnst = StyleIDNSTModule()
    sidnst.run()