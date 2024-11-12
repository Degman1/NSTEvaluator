from abc import ABC, abstractmethod
from time import time

class NSTModule(ABC):
    """Parent class from which all neural style transfer classes inherit"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loadModel(self):
        """Load the trained model parameters"""
        pass

    @abstractmethod
    def transferStyle(self, content_path, style_path):
        """
        Transfer the style from one image onto the content of another
        
        @param content_path String The path to the base content image
        @param style_path String The path to the style image
        @return String The diretory at which the output image will be stored. This output 
                will be named by concatenating the input file names with a dash seperator.
        """
        pass

    def benchmarkSingleStyleTransfer(self, content_path, style_path):
        """
        Benchmark a single style transfer operation

        @param content_path String The path to the base content image
        @param style_path String The path to the style image
        @return String The path at which the output image will be stored, named by 
                       concatenating the input file names with a dash seperator.
        @return Float The total time taken to perform the transfer in microseconds
        """
        start_time = time()
        output_path = self.transferStyle(content_path, style_path)
        end_time = time()
        total_time_microseconds = (end_time - start_time) * 1e6
        return output_path, total_time_microseconds

    def benchmarkStyleTransfer(self, data):
        """
        Benchmark multiple style transfer operations
        
        @param data [(String, String)] A list of pairs of content image and style image paths
        @return [(String, Float)] A list of pairs of image output paths and transfer times
        """

        outputs = []

        for pair in data:
            content_path = pair[0]
            style_path = pair[1]
            output_path, time = self.benchmarkSingleStyleTransfer(content_path, style_path)
            outputs.append((output_path, time))

        return outputs
