import torch
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import sys
import importlib

sys.path.append('/work/pi_hongkunz_umass_edu/dgerard/NSTEvaluator')

# Dynamic imports to allow dashes
Matrix = importlib.import_module('style-transfer-modules.LinearStyleTransfer.libs.Matrix')
models = importlib.import_module('style-transfer-modules.LinearStyleTransfer.libs.models')

from NSTModule import NSTModule

class LinearNSTModule(NSTModule):
    def __init__(self):
        super().__init__("Linear")
        
        cudnn.benchmark = True
        
        self.layer = 'r41'
        self.vgg_dir = 'LinearStyleTransfer/models/vgg_r41.pth'
        self.decoder_dir = 'LinearStyleTransfer/models/dec_r41.pth'
        self.matrix_path = 'LinearStyleTransfer/models/r41.pth'

    def _load_model(self):
        self.model['vgg'] = models.encoder4()
        self.model['dec'] = models.decoder4()
        self.model['matrix'] = Matrix.MulLayer(self.layer)
        
        self.model['vgg'].load_state_dict(torch.load(self.vgg_dir, map_location=self.device))
        self.model['dec'].load_state_dict(torch.load(self.decoder_dir, map_location=self.device))
        self.model['matrix'].load_state_dict(torch.load(self.matrix_path, map_location=self.device))
    
    def _preprocess_content_image(self, image):
        self.content_image.resize_(image.size()).copy_(image).to(self.device)
    
    def _preprocess_style_image(self, image):
        self.style_image.resize_(image.size()).copy_(image).to(self.device)
    
    def _transfer_style(self, content_image, style_image):
        with torch.no_grad():
            sF = self.model['vgg'](self.style_image)
            cF = self.model['vgg'](self.content_image)
            
            feature, transmatrix = self.model['matrix'](cF[self.layer],sF[self.layer])
            
            transfer = self.model['dec'](feature)
        
        return transfer

if __name__ == '__main__':
    lnst = LinearNSTModule()
    lnst.load_model()
    content_paths = ['LinearStyleTransfer/data/content/1.jpg', 'LinearStyleTransfer/data/content/chicago.png']
    style_paths = ['LinearStyleTransfer/data/style/27.jpg', 'LinearStyleTransfer/data/style/antimonocromatismo.jpg']
    lnst.benchmark_style_transfer(content_paths, style_paths)