from abc import ABC, abstractmethod
from time import time
import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import json
from Evaluator import Evaluator

from Loader import Dataset

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
        self.fine_size = 512
        self.load_size = 512
        self.content_image = torch.Tensor(self.batch_size, 3, self.fine_size, self.fine_size).to(self.device)
        self.style_image = torch.Tensor(self.batch_size, 3, self.fine_size, self.fine_size).to(self.device)
        
        # Image directories
        self.content_image_directory = '../content_images/'
        self.style_image_directory = '../style_images/'
        self.output_directory = f'../output/{name}'
        self._setup_output_directory()

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
        """Load the style transfer model and set up for inference on the proper device"""
        
        start_time = time()
        
        self._load_model()

        if (self.is_running_on_gpu()):
            print("Using cuda")
            for name, params in self.model.items():
                params.to(self.device)
                params.eval()
        else:
            print("Using cpu")
            for name, params in self.model.items():
                params.eval()
                
        end_time = time()
        total_time = end_time - start_time
        return total_time
            
    def _preprocess_content_image(self, image):
        """
        Resize or mutate content image before transferring
        
        @param path Numpy Array The base content image
        """
        self.content_image.resize_(image.size()).copy_(image).to(self.device)
    
    def _preprocess_style_image(self, image):
        """
        Resize or mutate content style before transferring
        
        @param image Numpy Array The style image
        """
        self.style_image.resize_(image.size()).copy_(image).to(self.device)
    
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
        @return Float The total time taken to perform the transfer in nanoseconds
        """
        start_time = time()
        output_path = self._transfer_style(content_image, style_image)
        end_time = time()
        total_time = end_time - start_time
        return output_path, total_time

    def _get_dataloaders(self):
        content_dataset = Dataset(self.content_image_directory, self.load_size, self.fine_size)
        content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                                    batch_size = self.batch_size,
                                                    shuffle = False,
                                                    num_workers = 1)
        style_dataset = Dataset(self.style_image_directory, self.load_size, self.fine_size)
        style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = False,
                                                num_workers = 1)
        return content_loader, style_loader

    def benchmark_style_transfer(self):
        """
        Benchmark multiple style transfer operations

        @return [(String, Float)] A list of pairs of image output paths and transfer times
        """

        outputs = {}
        
        content_loader, style_loader = self._get_dataloaders()
        
        for ci, (content, content_name) in enumerate(content_loader):
            content_name = content_name[0]
            self._preprocess_content_image(content)
            for sj, (style, style_name) in enumerate(style_loader):
                style_name = style_name[0]
                self._preprocess_style_image(style)
                output, time = self._benchmark_single_style_transfer(self.content_image, self.style_image)
                output_path = self._postprocess_output_image(output, content_name, style_name)
                outputs[output_path] = time
        
        return outputs

    def run(self):
        """
        Load the model and run inference
        """
        
        load_time = self.load_model()
        outputs = self.benchmark_style_transfer()
        outputs['load_time'] = load_time
        outputs['unit'] = 'seconds'
        
        print("Saving output paths and benchmarking results")
        
        with open(f"{self.output_directory}/time_sec.json", "w") as f:
            json.dump(outputs, f)

    def evaluate(self):
        results = {}
        for model_dir in os.listdir(self.output_directory):
            stylized_folder_path = os.path.join(self.output_directory, model_dir)
            artfid = Evaluator.artFid(self.style_image_directory, self.content_image_directory, stylized_folder_path)
            ssims = Evaluator.structuralSimilarity(self.style_image_directory, self.content_image_directory, stylized_folder_path)
            colorsims = Evaluator.colorSimilarity(self.style_image_directory, self.content_image_directory, stylized_folder_path)
            avg_times = Evaluator.timePerformance(self.style_image_directory, self.content_image_directory, stylized_folder_path)
            results[model_dir] = {
            "ArtFID": np.mean(artfid),
            "SSIM": np.mean(ssims),
            "ColorSim": np.mean(colorsims),
            "AvgTime": avg_times
            }
        print(results)