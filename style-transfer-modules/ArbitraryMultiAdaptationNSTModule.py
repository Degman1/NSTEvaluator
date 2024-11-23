import torch
import torch.nn as nn
import sys
import os
import importlib
from collections import OrderedDict

root = os.environ.get('STYLE_ROOT')
sys.path.append(root)
sys.path.append(f'{root}/style-transfer-modules/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network')

import function
import net

from NSTModule import NSTModule

class ArbitraryMultiAdaptationNSTModule(NSTModule):
    def __init__(self):
        super().__init__("ArbitraryMultiAdaptation")
        
        self.decoder_dir = 'Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/models/decoder_iter_160000.pth'
        self.transform_dir = 'Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/models/ma_module_iter_160000.pth'
        self.vgg_dir = 'Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/model/vgg_normalised.pth'

        self.alpha = 1.0
        
    def style_transfer(self, vgg, decoder, ma_module, content, style, alpha=1.0,
                    interpolation_weights=None):
        assert (0.0 <= alpha <= 1.0)
        style_fs, content_f, style_f=self.feat_extractor(vgg, content, style)
        Fccc = ma_module(content_f,content_f)
        #content_f[-2] = content_f.permute(0,1,3,2)
        if interpolation_weights:
            _, C, H, W = Fccc.size()
            feat = torch.FloatTensor(1, C, H, W).zero_().to(self.device)
            base_feat = ma_module(content_f, style_f)
            for i, w in enumerate(interpolation_weights):
                feat = feat + w * base_feat[i:i + 1]
            Fccc=Fccc[0:1]
        else:
            feat = ma_module(content_f, style_f)
        print(type(feat),type(Fccc))
        feat = feat * alpha + Fccc * (1 - alpha)
        feat_norm = function.normal(feat)
        feat = feat 
        return decoder(feat)
    
    def feat_extractor(self, vgg, content, style):
        norm = nn.Sequential(*list(vgg.children())[:1])
        enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
        enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
        enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
        enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
        enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

        norm.to(self.device)
        enc_1.to(self.device)
        enc_2.to(self.device)
        enc_4.to(self.device)
        enc_5.to(self.device)
        
        Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
        Content5_1 = enc_5(Content4_1)

        Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
        Style5_1 = enc_5(Style4_1)

        content_f=[Content4_1,Content5_1]
        style_f=[Style4_1,Style5_1]
        
        style_fs = [enc_1(style),enc_2(enc_1(style)),enc_3(enc_2(enc_1(style))),Style4_1, Style5_1]
        
        return style_fs,content_f, style_f
    
    def _load_model(self):
        self.model['decoder'] = net.decoder
        self.vgg = net.vgg
        network = net.Net(self.vgg, self.model['decoder'])
        self.model['ma_module'] = network.ma_module

        # Since not saving vgg as a direct model parameter, call eval here
        self.vgg.eval()
        
        new_state_dict = OrderedDict()
        state_dict = torch.load(self.decoder_dir)
        for k, v in state_dict.items():
            #namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        self.model['decoder'].load_state_dict(new_state_dict)
        
        new_state_dict = OrderedDict()
        state_dict = torch.load(self.transform_dir)
        for k, v in state_dict.items():
            #namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        self.model['ma_module'].load_state_dict(new_state_dict)

        self.vgg.load_state_dict(torch.load(self.vgg_dir))
        
        self.model['norm'] = nn.Sequential(*list(self.vgg.children())[:1])
        self.model['enc_1'] = nn.Sequential(*list(self.vgg.children())[:4])  # input -> relu1_1
        self.model['enc_2'] = nn.Sequential(*list(self.vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.model['enc_3'] = nn.Sequential(*list(self.vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.model['enc_4'] = nn.Sequential(*list(self.vgg.children())[18:31])  # relu3_1 -> relu4_1
        self.model['enc_5'] = nn.Sequential(*list(self.vgg.children())[31:44])  # relu4_1 -> relu5_1
    
    def _transfer_style(self, content_image, style_image):
        with torch.no_grad():
            output = self.style_transfer(self.vgg, self.model['decoder'], self.model['ma_module'], content_image, style_image)
        return output

if __name__ == '__main__':
    amanst = ArbitraryMultiAdaptationNSTModule()
    amanst.run()