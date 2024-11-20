import torch
import torch.nn as nn
import sys
import importlib
from collections import OrderedDict

sys.path.append('/work/pi_hongkunz_umass_edu/dgerard/NSTEvaluator')
sys.path.append('/work/pi_hongkunz_umass_edu/dgerard/NSTEvaluator/style-transfer-modules')

# Dynamic imports to allow dashes
function = importlib.import_module('style-transfer-modules.Arbitrary-Style-Transfer-via-Multi-Adaptation-Network.function')

from NSTModule import NSTModule


class CA(nn.Module):
    def __init__(self, in_dim):
        super(CA, self).__init__()
        self.f = nn.Conv2d(in_dim , in_dim , (1,1))
        self.g = nn.Conv2d(in_dim , in_dim , (1,1))
        self.h = nn.Conv2d(in_dim , in_dim , (1,1))
        self.softmax  = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))
     
    def forward(self,content_feat,style_feat):
    
        B,C,H,W = content_feat.size()
        F_Fc_norm  = self.f(function.normal(content_feat)).view(B,-1,H*W).permute(0,2,1)
        
    

        B,C,H,W = style_feat.size()
        G_Fs_norm =  self.g(function.normal(style_feat)).view(B,-1,H*W) 
   
        energy =  torch.bmm(F_Fc_norm,G_Fs_norm)
        attention = self.softmax(energy)
        

        H_Fs = self.h(style_feat).view(B,-1,H*W)
        out = torch.bmm(H_Fs,attention.permute(0,2,1) )
        B,C,H,W = content_feat.size()
        out = out.view(B,C,H,W)
        out = self.out_conv(out)
     
        out += content_feat
        
        return out
 
class Style_SA(nn.Module):
    def __init__(self, in_dim):
        super(Style_SA, self).__init__()
        self.f = nn.Conv2d(in_dim , in_dim , (1,1))
        self.g = nn.Conv2d(in_dim , in_dim , (1,1))
        self.h = nn.Conv2d(in_dim , in_dim , (1,1))
        self.softmax  = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self,style_feat):

        B,C,H,W = style_feat.size()
        F_Fc_norm  = self.f(style_feat).view(B,-1,H*W)
        
   
        B,C,H,W = style_feat.size()
        G_Fs_norm =  self.g(style_feat).view(B,-1,H*W).permute(0,2,1) 

        energy =  torch.bmm(F_Fc_norm,G_Fs_norm)
        attention = self.softmax(energy)


        H_Fs = self.h(function.normal(style_feat)).view(B,-1,H*W)
        out = torch.bmm(attention.permute(0,2,1), H_Fs)
        
        out = out.view(B,C,H,W)
        out = self.out_conv(out)
        out += style_feat
        return out
    
class Content_SA(nn.Module):
    def __init__(self, in_dim):
        super(Content_SA, self).__init__()
        self.f = nn.Conv2d(in_dim , in_dim , (1,1))
        self.g = nn.Conv2d(in_dim , in_dim , (1,1))
        self.h = nn.Conv2d(in_dim , in_dim , (1,1))
        self.softmax  = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    
    def forward(self,content_feat):

        B,C,H,W = content_feat.size()
        F_Fc_norm  = self.f(function.normal(content_feat)).view(B,-1,H*W).permute(0,2,1)


        B,C,H,W = content_feat.size()
        G_Fs_norm =  self.g(function.normal(content_feat)).view(B,-1,H*W) 

        energy =  torch.bmm(F_Fc_norm,G_Fs_norm)
        attention = self.softmax(energy)
        
        H_Fs = self.h(content_feat).view(B,-1,H*W)
        out = torch.bmm(H_Fs,attention.permute(0,2,1) )
        B,C,H,W = content_feat.size()
        out = out.view(B,C,H,W)
        out = self.out_conv(out)
        out += content_feat
  
        return out

class Multi_Adaptation_Module(nn.Module):
    def __init__(self, in_dim):
        super(Multi_Adaptation_Module, self).__init__()

        self.CA=CA(in_dim)
        self.CSA=Content_SA(in_dim)
        self.SSA=Style_SA(in_dim)

    def forward(self, content_feats, style_feats):
      
        content_feat = self.CSA(content_feats[-2])
        style_feat = self.SSA(style_feats[-2])
        Fcsc = self.CA(content_feat, style_feat)
      
        return Fcsc

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        #transform
        self.ma_module = Multi_Adaptation_Module(512)
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

           
    def forward(self, content, content1, style, style1):
        #print(content.size())
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        style_feats1 = self.encode_with_intermediate(style1)
        content_feats1 = self.encode_with_intermediate(content1)


        Ics = self.decoder(self.ma_module(content_feats, style_feats))
        Ics_feats = self.encode_with_intermediate(Ics)
        # Content loss
   
        Ics1 = self.decoder(self.ma_module(content_feats, style_feats1))
        Ics1_feats = self.encode_with_intermediate(Ics1)
        Ic1s = self.decoder(self.ma_module(content_feats1, style_feats))
        Ic1s_feats = self.encode_with_intermediate(Ic1s)
        
        #Identity losses lambda 1
        Icc = self.decoder(self.ma_module(content_feats, content_feats))
        Iss = self.decoder(self.ma_module(style_feats, style_feats)) 
    
        #Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)
        return style_feats, content_feats, style_feats1, content_feats1 ,Ics_feats,Ics1_feats,Ic1s_feats,Icc,Iss,Icc_feats,Iss_feats

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
        self.model['decoder'] = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        
        network = Net(self.vgg, self.model['decoder'])
        self.model['ma_module'] = network.ma_module
        
        self.model['decoder'].eval()
        self.model['ma_module'].eval()
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
    
    # def _preprocess_content_image(self, image):
    #     pass
    
    # def _preprocess_style_image(self, image):
    #     pass
    
    def _transfer_style(self, content_image, style_image):
        with torch.no_grad():
            output = self.style_transfer(self.vgg, self.model['decoder'], self.model['ma_module'], content_image, style_image)
        return output

if __name__ == '__main__':
    lnst = ArbitraryMultiAdaptationNSTModule()
    lnst.load_model()
    lnst.benchmark_style_transfer()