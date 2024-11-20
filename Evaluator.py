import numpy as np
import subprocess
from skimage.metrics import structural_similarity as ssim
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image

class Evaluator:

    def artFid(style_images_path, content_images_path, stylized_images_path, cuda_device='0'):
        """
        Calls the art_fid module to compute FID metric

        @parameters
            style_images_path (str): Path to the directory containing style images.
            content_images_path (str): Path to the directory containing content images.
            stylized_images_path (str): Path to the directory containing stylized images.
            cuda_device (str): CUDA device to use (default is '0').

        @return output (str): Output from the art_fid command.
        """
        try:
            command = [
                'python', '-m', 'art_fid',
                '--style_images', style_images_path,            #specify the path to directory containing input style images
                '--content_images', content_images_path,        #specify the path to directory containing input content images
                '--stylized_images', stylized_images_path       #specify the path to directory containing output stylized images
            ]
            env = {'CUDA_VISIBLE_DEVICES': cuda_device}         #set the CUDA_VISIBLE_DEVICES variable (specified in art-fid usage prompt)
            result = subprocess.run(        #execute the command as a subprocess
                command,                    #command portion of the usage command line prompt
                capture_output=True,        #capture the output
                text=True,                  #treat the output as text
                check=True,                 #if command fails return error
                env=env                     #specify the cuda environment (default is 0)
            )
            return result.stdout  #return the output from art-fid
        except subprocess.CalledProcessError as e:
            print("Error calling art_fid:", e)
            print("Error output:", e.stderr)
            return None

    

    def structuralSimilarity (content_folder_path, stylized_folder_path):
        #may need to expand this to follow the same idea as artfid, where it iterates through the folders of the content and stylized images to do this calculation all at once
        """
        Compute the structural similarity between the content image and the stylized image

        @parameters
            content_folder_path (str): Path to the content image folder.
            stylized_folder_path (str): Path to the stylized image folder.

        @return ssim_scores [] (float): Strucutral similarity scores between the two images, all saved in a list.
        """
        ssim_scores = []
        content_files = os.listdir(content_folder_path)
        stylized_files = os.listdir(stylized_folder_path)
        for i in range(len(content_files)):
            content_image_path = content_files[i]
            stylized_image_path = stylized_files[i]
            content_image = cv2.imread(content_image_path, cv2.IMREAD_GRAYSCALE)       #load the respective images
            stylized_image = cv2.imread(stylized_image_path, cv2.IMREAD_GRAYSCALE)
            if content_image is None or stylized_image is None:
                raise ValueError("One of the images could not be loaded.")              #error if one of the images isn't loaded properly
            if content_image.shape != stylized_image.shape:
                stylized_image = cv2.resize(stylized_image, (content_image.shape[1], content_image.shape[0]))       #reshape if the two images aren't the same size
            ssim_score, _ = ssim(content_image, stylized_image, full=True)              #compute the ssim score for the two images
            ssim_scores.append(ssim_score)
        return ssim_scores


    def colorSimilarity (style_folder_path, stylized_folder_path) :
        #may need the same idea as structual similarity
        """
        Compute the structural similarity between the style image and the stylized image

        @parameters
            style_image_path (str): Path to the style folder.
            stylized_image_path (str): Path to the stylized folder.

        @return color_similarity_score [] (float): Color similarity scores between the two images, all saved in a list.
        """
        color_similarity_scores = []
        style_files = os.listdir(style_folder_path)
        stylized_files = os.listdir(stylized_folder_path)
        for i in range(len(style_files)):
            style_image_path = style_files[i]
            stylized_image_path = stylized_files[i]
            style_image = cv2.imread(style_image_path)                                #load the respective images
            stylized_image = cv2.imread(stylized_image_path)
            if style_image is None or stylized_image is None:
                raise ValueError("One of the images failed to load.")             #error if one of the images isn't loaded properly
            if style_image.shape != stylized_image.shape:
                stylized_image = cv2.resize(stylized_image, (style_image.shape[1], style_image.shape[0]))
            style_hist = [cv2.calcHist([style_image], [i], None, [256], [0, 256]) for i in range(3)]                #generate the color histograms for the two respective images
            stylized_hist = [cv2.calcHist([stylized_image], [i], None, [256], [0, 256]) for i in range(3)]
            style_hist = [cv2.normalize(h, h).flatten() for h in style_hist]                                        #normalize the histograms
            stylized_hist = [cv2.normalize(h, h).flatten() for h in stylized_hist]
            color_similarity_score = np.mean([cv2.compareHist(style_hist[i], stylized_hist[i], cv2.HISTCMP_CORREL) for i in range(3)])          #compute the similarity score based on the average correlation for the two normalized histograms
            color_similarity_scores.append(color_similarity_score)
        return color_similarity_scores



    def contentStyleLoss(content_folder_path, style_folder_path, stylized_folder_path):
        """
        Compute the combined content and style loss between the given images.

        @parameters
            content_folder_path (str): Path to the content folder.
            style_folder_path (str): Path to the style folder.
            stylized_folder_path (str): Path to the stylized folder.

        @return content_loss [], style_loss [] (float): Computed Content and Style loss between the respective stylized and content/style images, return as lists
        """
        pass


    def timePerformance ():
        """
        Compute the timed performance of a given Style Transfer method
        """
        pass