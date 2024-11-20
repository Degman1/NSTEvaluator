import torch
import sys
import importlib

sys.path.append('/work/pi_hongkunz_umass_edu/dgerard/NSTEvaluator')

from NSTModule import NSTModule

class StyleIDNSTModule(NSTModule):
    def __init__(self):
        super().__init__("StyleID")
        
    def _load_model(self):
        pass
    
    def _preprocess_content_image(self, image):
        pass
    
    def _preprocess_style_image(self, image):
        pass
    
    def _transfer_style(self, content_image, style_image):
        pass

if __name__ == '__main__':
    lnst = StyleIDNSTModule()
    lnst.load_model()
    lnst.benchmark_style_transfer()