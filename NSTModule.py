from abc import ABC, abstractmethod
from time import time
import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms

class NSTModule(ABC):
    """Parent class from which all neural style transfer classes inherit"""

    def __init__(self, name):
        super().__init__()
        
        self.name = name
        
        # Check if running on gpu or cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Store model parameters necessary for the transfer operation
        self.model = {}
        
        # Define number of images per batch
        self.batch_size = 1
        
        # Allocate memory for these tensors for reuse on the GPU
        self.fine_size = 256
        self.content_image = torch.Tensor(self.batch_size, 3, self.fine_size, self.fine_size).to(self.device)
        self.style_image = torch.Tensor(self.batch_size, 3, self.fine_size, self.fine_size).to(self.device)
        
        # Image directories
        self.content_image_directory = '../content_images/'
        self.style_image_directory = '../style_images/'
        self.output_directory = f'../output/{name}'
        self._setup_output_directory()
        
        self.resize_transform = transforms.Resize((256, 256))

    def _setup_output_directory(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
    def is_running_on_gpu(self):
        """Check if a GPU is available"""
        return self.device == 'cuda'

    @abstractmethod
    def _load_model(self):
        """Load the trained model parameters"""
        pass

    @abstractmethod
    def _transfer_style(self, content_image, style_image):
        """
        Transfer the style from one image onto the content of another
        
        @param content_image Numpy Array The base content image
        @param style_image Numpy Array The style image
        @return String The diretory at which the output image will be stored. This output 
                will be named by concatenating the input file names with a dash seperator.
        """
        pass

    def load_model(self):
        """Load the style transfer model and set up for inference"""
        self._load_model()

        if (self.is_running_on_gpu()):
            print("Using cuda")
            for name, params in self.model.items():
                params.to(self.device)
        else:
            print("Using cpu")
    
    def _preprocess_content_image(self, image):
        """
        Resize or mutate content image before transferring
        
        @param path Numpy Array The base content image
        """
        return image
    
    def _preprocess_style_image(self, image):
        """
        Resize or mutate content style before transferring
        
        @param image Numpy Array The style image
        """
        return image
    
    def _postprocess_output_image(self, image, content_name, style_name):
        """
        Ensure valid pixel values after performing style transfer and properly store
        
        @param image Numpy Array The transfer output image
        """
        image = image.clamp(0,1).squeeze(0)
        output_path = '%s/%s_%s.png' % (self.output_directory, content_name, style_name)
        vutils.save_image(image, output_path, normalize=True, scale_each=True, nrow=self.batch_size)
        print(f'Transferred image saved at {output_path}')
        return output_path

    def _benchmark_single_style_transfer(self, content_image, style_image):
        """
        Benchmark a single style transfer operation

        @param content_image Numpy Array The base content image
        @param style_image Numpy Array The style image
        @return String The path at which the output image will be stored, named by 
                       concatenating the input file names with a dash seperator.
        @return Float The total time taken to perform the transfer in microseconds
        """
        start_time = time()
        output_path = self._transfer_style(content_image, style_image)
        end_time = time()
        total_time_microseconds = (end_time - start_time) * 1e6
        return output_path, total_time_microseconds

    def _load_image(self, path):
        """
        Load an image from disk into a numpy array
        
        @param path String The path to the image
        @return The image as a Numpy array
        """
        image = Image.open(path).convert("RGB")
        image = self.resize_transform(image)
        # Reorder to [C, H, W] and add batch dimension to make it [1, C, H, W]
        image_array = torch.Tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        image_name = os.path.splitext(os.path.basename(path))[0]
        return image_array, image_name

    def benchmark_style_transfer(self, content_paths, style_paths):
        """
        Benchmark multiple style transfer operations
        
        @param content_path String The paths to the base content images
        @param style_path String The paths to the style images
        @return [(String, Float)] A list of pairs of image output paths and transfer times
        """

        outputs = {}

        for style_path in style_paths:
            style_image, style_name = self._load_image(style_path)
            self._preprocess_style_image(style_image)
            for content_path in content_paths:
                content_image, content_name = self._load_image(content_path)
                self._preprocess_content_image(content_image)
                output, time = self._benchmark_single_style_transfer(self.content_image, self.style_image)
                output_path = self._postprocess_output_image(output, content_name, style_name)
                outputs[output_path] = time
                # Clear cache to save memory
                torch.cuda.empty_cache()
            # Clear cache to save memory
            torch.cuda.empty_cache()
            
        return outputs
