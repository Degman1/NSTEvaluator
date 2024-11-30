import numpy as np
import shutil
import subprocess
from skimage.metrics import structural_similarity as ssim
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
import json
import sys
import tempfile
import re
from torchvision.models import SqueezeNet1_1_Weights

def prepare_directory(src_dir, dst_dir, valid_extensions=('.png', '.jpg', '.jpeg')):
        """
        Prepares a directory by copying only valid image files.
        
        Parameters:
            src_dir (str): Source directory containing files.
            dst_dir (str): Destination directory to store valid files.
            valid_extensions (tuple): Valid file extensions to include.
        """
        # Create destination directory if it doesn't exist
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir)
        # Copy only valid image files
        for file in os.listdir(src_dir):
            if file == ".DS_Store":
                continue
            if file.lower().endswith(valid_extensions):
                shutil.copy(os.path.join(src_dir, file), dst_dir)

def create_copies(src_dir, copies): 
    """
    Ensure that the style directory contains the proper number of copies for ART_FID to properly execute
    """
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)
        if os.path.isfile(file_path):
            for i in range(copies):
                new_filename = f"{os.path.splitext(filename)[0]}_copy{i+1}{os.path.splitext(filename)[1]}" #copies name
                new_file_path = os.path.join(src_dir, new_filename)
                shutil.copy(file_path, new_file_path) #copy the file to the directory


def create_interleaved_copies(src_dir, copies):
    """"
    Ensure that the content directory is in the proper ordering and contains the proper number of copies of images for ART_FID
    """
    files = sorted(os.listdir(src_dir))
    for i in range(copies):
        for filename in files:
            file_path = os.path.join(src_dir, filename)
            if os.path.isfile(file_path):
                new_filename = f"{os.path.splitext(filename)[0]}_copy{i+1}{os.path.splitext(filename)[1]}"
                new_file_path = os.path.join(src_dir, new_filename)
                shutil.copy(file_path, new_file_path)

def resize_images(image_folder, target_size=(224, 224)):
    """
    Resize images in given directory to 224x224 for ART_FID
    """
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        img = Image.open(image_path)
        img_resized = img.resize(target_size)
        img_resized.save(image_path)


class Evaluator:

    def artFidHandler(style_images_path, content_images_path, stylized_images_path):
        """
        Passes the corrected inputs to the art_fid module

        @parameters
            style_images_path (str): Path to the directory containing style images.
            content_images_path (str): Path to the directory containing content images.
            stylized_images_path (str): Path to the directory containing stylized images.
        
        @return results (float): Output from the art_fid value.
        """
        temp_stylized_dir = "temp_stylized_images" #create temporary directories that will all have the same size
        temp_content_dir = "temp_content_images"
        temp_style_dir = "temp_style_images"
        ogstylized = os.listdir(stylized_images_path)
        ogscontent = os.listdir(content_images_path)
        ogstyle = os.listdir(style_images_path)
        prepare_directory(stylized_images_path, temp_stylized_dir) #ensure only image files
        prepare_directory(content_images_path, temp_content_dir)
        prepare_directory(style_images_path, temp_style_dir)
        ogstylized = os.listdir(temp_stylized_dir)
        ogscontent = os.listdir(temp_content_dir)
        ogstyle = os.listdir(temp_style_dir)
        stylized_images_path = temp_stylized_dir
        content_images_path = temp_content_dir
        style_images_path = temp_style_dir
        create_copies(temp_style_dir, len(sorted(os.listdir(content_images_path))) - 1)
        create_interleaved_copies(temp_content_dir, len(sorted(os.listdir(style_images_path))) - 1)
        resize_images(style_images_path)
        resize_images(content_images_path)
        resize_images(stylized_images_path)
        results = Evaluator.artFid(style_images_path, content_images_path, stylized_images_path)
        shutil.rmtree(temp_stylized_dir) #clear the temporary directories
        shutil.rmtree(temp_content_dir)
        shutil.rmtree(temp_style_dir)
        return results

    def artFid(style_images_path, content_images_path, stylized_images_path, cuda_device='0'):
        """
        Calls the art_fid module to compute FID metric

        @parameters
            style_images_path (str): Path to the directory containing style images.
            content_images_path (str): Path to the directory containing content images.
            stylized_images_path (str): Path to the directory containing stylized images.
            cuda_device (str): CUDA device to use (default is '0').

        @return artfid_value (float): Output from the art_fid value.
        """
        try:
            command = [
                'python', '-m', 'art_fid',
                '--style_images', style_images_path,            #specify the path to directory containing input style images
                '--content_images', content_images_path,        #specify the path to directory containing input content images
                '--stylized_images', stylized_images_path,       #specify the path to directory containing output stylized images
                '--batch_size', "1"
            ]
            env = os.environ.copy()        #set the CUDA_VISIBLE_DEVICES variable (specified in art-fid usage prompt)
            result = subprocess.run(        #execute the command as a subprocess
                command,                    #command portion of the usage command line prompt
                capture_output=True,        #capture the output
                text=True,                  #treat the output as text
                check=True,                 #if command fails return error
                env=env                     #specify the cuda environment (default is 0)
            )
            match = re.search(r"ArtFID value: ([\d.]+)", result.stdout)
            artfid_value = None
            if match is not None and match.group(1) is not None:
                artfid_value = float(match.group(1))
            return artfid_value #return the output from art-fid
        
        except subprocess.CalledProcessError as e:
            print("Error calling art_fid:", e)
            print("Error output:", e.stderr)
            return None

    

    def structuralSimilarity (content_folder_path, stylized_folder_path):
        """
        Compute the structural similarity between the content image and the stylized image

        @parameters
            content_folder_path (str): Path to the content image folder.
            stylized_folder_path (str): Path to the stylized image folder.

        @return ssim_scores [] (float): Strucutral similarity scores between the two images, all saved in a list.
        """
        temp_stylized_dir = "temp_stylized_folder"
        prepare_directory(stylized_folder_path, temp_stylized_dir)
        stylized_folder_path = temp_stylized_dir
        ssim_scores = []
        content_files = sorted(os.listdir(content_folder_path))
        stylized_files = sorted(os.listdir(stylized_folder_path))
        for content_file in content_files:
            if content_file == ".DS_Store":
                continue
            content_image_path = os.path.join(content_folder_path, content_file)
            content_image = cv2.imread(content_image_path, cv2.IMREAD_GRAYSCALE)       #load the respective images
            content_prefix = os.path.splitext(content_file)[0]
            for stylized_file in stylized_files:
                if stylized_file.startswith(content_prefix):  # Match stylized images based on content image prefix
                    stylized_image_path = os.path.join(stylized_folder_path, stylized_file)
                    stylized_image = cv2.imread(stylized_image_path, cv2.IMREAD_GRAYSCALE)
                    if content_image is None or stylized_image is None:
                        raise ValueError("One of the images could not be loaded.")              #error if one of the images isn't loaded properly
                    if content_image.shape != stylized_image.shape:
                        stylized_image = cv2.resize(stylized_image, (content_image.shape[1], content_image.shape[0]))       #reshape if the two images aren't the same size
                    ssim_score, _ = ssim(content_image, stylized_image, full=True)              #compute the ssim score for the two images
                    ssim_scores.append(ssim_score)
        shutil.rmtree(temp_stylized_dir)
        return ssim_scores


    def colorSimilarity (style_folder_path, stylized_folder_path) :
        """
        Compute the structural similarity between the style image and the stylized image

        @parameters
            style_image_path (str): Path to the style folder.
            stylized_image_path (str): Path to the stylized folder.

        @return color_similarity_score [] (float): Color similarity scores between the two images, all saved in a list.
        """
        temp_stylized_dir = "temp_stylized_folder"
        prepare_directory(stylized_folder_path, temp_stylized_dir)
        stylized_folder_path = temp_stylized_dir
        color_similarity_scores = []
        style_files = sorted(os.listdir(style_folder_path))
        stylized_files = sorted(os.listdir(stylized_folder_path))
        for style_file in style_files:
            if style_file == ".DS_Store":
                continue
            style_image_path = os.path.join(style_folder_path, style_file)
            style_image = cv2.imread(style_image_path) #load the respective style image
            style_suffix = os.path.splitext(style_file)[0]
            for stylized_file in stylized_files:
                if os.path.splitext(stylized_file)[0].endswith(style_suffix):  #match stylized images by the suffix
                    stylized_image_path = os.path.join(stylized_folder_path, stylized_file)
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
        shutil.rmtree(temp_stylized_dir)
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
        content_losses = []
        style_losses = []
        content_files = sorted(os.listdir(content_folder_path))
        style_files = sorted(os.listdir(style_folder_path))
        temp_stylized_dir = "temp_stylized_folder"
        prepare_directory(stylized_folder_path, temp_stylized_dir)
        stylized_folder_path = temp_stylized_dir
        stylized_files = sorted(os.listdir(stylized_folder_path))
        k = 0
        for i in range(len(style_files)):
            for j in range(len(content_files)): 
                content_file = content_files[j]
                style_file = style_files[i]
                stylized_file = stylized_files[k]
                if style_file == ".DS_Store" or content_file == ".DS_Store":
                    continue
                content_image_path = os.path.join(content_folder_path, content_file)
                style_image_path = os.path.join(style_folder_path, style_file)
                stylized_image_path = os.path.join(stylized_folder_path, stylized_file)
                image_size = 192
                content_layer = 3
                content_weight = 6e-2
                cnn = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1).features
                cnn = cnn.eval()
                for param in cnn.parameters():
                    param.requires_grad = False
                def preprocess(img, size=512):
                    transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        transforms.Lambda(lambda x: x[None]),
                    ])
                    return transform(img)
                def features_from_img(imgpath, imgsize):
                    img = preprocess(Image.open(imgpath), size=imgsize)
                    img_var = img.type(torch.FloatTensor)
                    return extract_features(img_var, cnn), img_var
                def extract_features(x, cnn):
                    features = []
                    prev_feat = x
                    for i, module in enumerate(cnn._modules.values()):
                        next_feat = module(prev_feat)
                        features.append(next_feat)
                        prev_feat = next_feat
                    return features
                content_feats, content_img_var = features_from_img(content_image_path, image_size)
                style_feats, style_img_var = features_from_img(style_image_path, image_size)
                stylized_feats, stylized_img_var = features_from_img(stylized_image_path, image_size)
                content_loss = content_weight * torch.sum(torch.pow(content_feats[content_layer] - stylized_feats[content_layer], 2)).cpu().data.numpy()
                def gram_matrix(features):
                    N, C, H, W = features.size() #extract the feature dimensions
                    feat = features.view(N, C, -1) #reshape to (N, C, H * W)
                    gram = torch.bmm(feat, feat.transpose(1, 2)) #compute the gram matrix (N, C, C)
                    return gram
                def compute_style_loss(feats, style_layers, style_targets, style_weights):
                    cur_style_loss = 0
                    for i in range(len(style_layers)): #iterate for each layer
                        gram = gram_matrix(feats[style_layers[i]]) #retrieve the respective gram matrix
                        cur_style_loss += style_weights[i] * torch.sum(torch.pow(gram - style_targets[i], 2)) #compute the style loss given the equation
                    return cur_style_loss
                style_targets = []
                for idx in [1, 4, 6, 7]:
                    style_targets.append(gram_matrix(style_feats[idx].clone()))
                style_loss = compute_style_loss(stylized_feats, [1, 4, 6, 7], style_targets, [300000, 1000, 15, 3]).cpu().data.numpy()
                content_losses.append(content_loss)
                style_losses.append(style_loss)
                k += 1
        shutil.rmtree(temp_stylized_dir)
        return content_losses, style_losses


    def timePerformance (output_path):
        """
        Compute the timed performance of a given Style Transfer method
        @parameters
            output_path (str): string for the directory for the respective models output

        @return average_time (float), load_time (float), unit (str): average time it took the model to execute, the load_time for the model, and then the unit for the times
        """
        for file in os.listdir(output_path):
            if file.lower().endswith('.json'):  
                time_file = os.path.join(output_path, file) #identify the json file containin the time it took for each image style transfer to be executed
        try:
            with open(time_file, 'r') as file:
                data = json.load(file) #load the json file
            time_sum = 0
            time_count = 0
            load_time = 0
            unit = ""
            for key, value in data.items():
                if key not in ["load_time", "unit"]:
                    time_sum += value
                    time_count += 1
                elif key == "load_time":
                    load_time = value
                elif key == "unit":
                    unit = value
            average_time = time_sum / time_count #compute the average time
            return average_time, load_time, unit
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading JSON file: {e}")
            return None
    
    def animationPerformance(ssim_score, color_score, content_loss, style_loss, avg_time, load_time):
        """
        Weighted metric that combines the other metrics to more accurately assess model performance
        
        @parameters

        @return result (float): output metric computed for the respective model
        """
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        zetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        scores = {}
        norm_content_loss = 1 - (content_loss / 250000)
        norm_style_loss = 1 - (style_loss / 1e+20)
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    for delta in deltas:
                        for epsilon in epsilons:
                            for zeta in zetas:
                                if alpha + beta + gamma + delta + epsilon + zeta == 1:
                                    out = alpha * ssim_score + beta * color_score + gamma * norm_content_loss + delta * norm_style_loss + epsilon * (1 / avg_time) + zeta * (1 / load_time)
                                    scores[(alpha, beta, gamma, delta, epsilon, zeta)] = out
        
        maxKey = max(scores, key = scores.get)
        return scores[maxKey]
        
    def evaluate(style, content, output):
        results = {}
        for model_dir in os.listdir(output):
            if model_dir == ".DS_Store":
                continue
            stylized_folder_path = os.path.join(output, model_dir)
            print(model_dir, "started")
            artfid = Evaluator.artFidHandler(style, content, stylized_folder_path)
            print("artfid done")
            ssims = Evaluator.structuralSimilarity(content, stylized_folder_path)
            print("ssims done")
            colorsims = Evaluator.colorSimilarity(style, stylized_folder_path)
            print("colors done")
            contentLoss, styleLoss = Evaluator.contentStyleLoss(content, style, stylized_folder_path)
            print("loss done")
            avg_times, load_time, unit = Evaluator.timePerformance(stylized_folder_path)
            print("times done")
            ani_score = Evaluator.animationPerformance(np.mean(ssims), np.mean(colorsims), np.mean(contentLoss), np.mean(styleLoss), avg_times, load_time)
            results[model_dir] = {
            "ArtFID": artfid,
            "ArtFID": artfid,
            "SSIM": np.mean(ssims),
            "ColorSim": np.mean(colorsims),
            "ContentLoss": np.mean(contentLoss),
            "StyleLoss": np.mean(styleLoss),
            "AnimationScore": ani_score,
            "AvgTime": avg_times,
            "LoadTime": load_time
            }
            print(model_dir, " done")
        return results
    

content_path = sys.argv[1]
style_path = sys.argv[2]
output_path = sys.argv[3]
#content_path = "/Users/bmhall17/Desktop/UMass/Fall 2024/CS682 Neural Networks/cs682finalproject/NSTEvaluator/content_images"
#style_path = "/Users/bmhall17/Desktop/UMass/Fall 2024/CS682 Neural Networks/cs682finalproject/NSTEvaluator/style_images"
#output_path = "/Users/bmhall17/Downloads/output"
evaluation_results = Evaluator.evaluate(style_path, content_path, output_path)
print(evaluation_results)
