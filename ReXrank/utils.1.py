import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import skimage.morphology, skimage.io
import cv2
import numpy as np
import random
from transformers import StoppingCriteria, StoppingCriteriaList
from copy import deepcopy
from medomni.common.config import Config
from medomni.common.dist_utils import get_rank
from medomni.common.registry import registry
import torchio as tio
import nibabel as nib
from scipy import ndimage, misc
import time
import ipdb

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def seg_2d_process(image_path, pred_mask, img_size=224):
    image = cv2.imread(image_path[0])
    if pred_mask.sum() != 0:
        labels = skimage.morphology.label(pred_mask)
        labelCount = np.bincount(labels.ravel())
        largest_label = np.argmax(labelCount[1:]) + 1
        pred_mask[labels != largest_label] = 0
        pred_mask[labels == largest_label] = 255
        pred_mask = pred_mask.astype(np.uint8) * 255
        binary_array = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = [binary_array / 255]
        # mask = [cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)]
    else:
        mask = [np.zeros((image.shape[0], image.shape[1]))]
    image = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))]
    return image, mask

def seg_3d_process(image_path, seg_mask):
    img  = nib.load(image_path[0]).get_fdata()
    image = window_scan(img).transpose(2,0,1).astype(np.uint8)
    if seg_mask.sum() != 0:
        seg_mask = resize_back_volume_abd(seg_mask, image.shape).astype(np.uint8)
        image_slices = []
        contour_slices = []
        for i in range(seg_mask.shape[0]):
            slice_img = np.fliplr(np.rot90(image[i]))
            slice_mask = np.fliplr(np.rot90(seg_mask[i]))
            contours, _ = cv2.findContours(slice_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            image_slices.append(Image.fromarray(slice_img))
            if contours:
                binary_array = np.zeros(seg_mask.shape[1:])
                binary_array = cv2.drawContours(binary_array, contours, -1, 255, thickness=cv2.FILLED) / 255
                binary_array = cv2.resize(binary_array, slice_img.shape, interpolation = cv2.INTER_NEAREST)
                contour_slices.append(binary_array)
            else:
                contour_slices.append(np.zeros_like(slice_img))
    else:
        image_slices = []
        contour_slices = []
        slice_img = np.fliplr(np.rot90(image[i]))
        image_slices.append(Image.fromarray(slice_img))
        contour_slices.append(np.zeros_like(slice_img))

    return image_slices, contour_slices

def det_2d_process(image_path, box):
    image_slices = []
    image = cv2.imread(image_path[0])
    if box is not None:
        hi,wd,_ = image.shape
        color = tuple(np.random.random(size=3) * 256)
        x1, y1, x2, y2 = int(box[0]*wd), int(box[1]*hi), int(box[2]*wd), int(box[3]*hi)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
    image_slices.append(Image.fromarray(image))
    return image_slices

def window_scan(scan, window_center=50, window_width=400):
    """
    Apply windowing to a scan.

    Parameters:
    scan (numpy.ndarray): 3D numpy array of the CT scan
    window_center (int): The center of the window
    window_width (int): The width of the window

    Returns:
    numpy.ndarray: Windowed CT scan
    """
    lower_bound = window_center - (window_width // 2)
    upper_bound = window_center + (window_width // 2)
    
    windowed_scan = np.clip(scan, lower_bound, upper_bound)
    windowed_scan = (windowed_scan - lower_bound) / (upper_bound - lower_bound)
    windowed_scan = (windowed_scan * 255).astype(np.uint8)
    
    return windowed_scan

def task_seg_2d(model, preds, hidden_states, image):
    token_mask = preds == model.seg_token_idx_2d  
    indices = torch.where(token_mask == True)[0].cpu().numpy()
    feats = model.model_seg_2d.encoder(image.unsqueeze(0)[:, 0])
    last_feats = feats[-1]
    target_states = [hidden_states[ind][-1] for ind in indices]
    if target_states:
        target_states = torch.cat(target_states).squeeze()
        seg_states = model.text2seg_2d(target_states).unsqueeze(0)
        last_feats = last_feats + seg_states.unsqueeze(-1).unsqueeze(-1)
        last_feats = model.text2seg_2d_gn(last_feats)
        feats[-1] = last_feats
        seg_feats = model.model_seg_2d.decoder(*feats)
        seg_preds = model.model_seg_2d.segmentation_head(seg_feats)
        seg_probs = F.sigmoid(seg_preds)
        seg_mask = seg_probs.to(torch.float32).cpu().squeeze().numpy() >= 0.5
        return seg_mask
    else:
        return None

def task_sam_2d(model, preds, hidden_states, image):
    token_mask = preds == model.seg_token_idx_2d  
    indices = torch.where(token_mask == True)[0].cpu().numpy()
    feats = model.model_sam_2d.encoder(image.unsqueeze(0)[:, 0])
    last_feats = feats[-1]
    target_states = [hidden_states[ind][-1] for ind in indices]
    if target_states:
        target_states = torch.cat(target_states).squeeze()
        seg_states = model.text2sam_2d(target_states).unsqueeze(0)
        last_feats = last_feats + seg_states.unsqueeze(-1).unsqueeze(-1)
        last_feats = model.text2sam_2d_bn(last_feats)
        feats[-1] = last_feats
        seg_feats = model.model_sam_2d.decoder(*feats)
        seg_preds = model.model_sam_2d.segmentation_head(seg_feats)
        seg_probs = F.sigmoid(seg_preds)
        seg_mask = seg_probs.to(torch.float32).cpu().squeeze().numpy() >= 0.5
        return seg_mask
    else:
        return None  

def task_seg_3d(model, preds, hidden_states, img_embeds_list):
    new_img_embeds_list = deepcopy(img_embeds_list)
    token_mask = preds == model.seg_token_idx_3d  
    indices = torch.where(token_mask == True)[0].cpu().numpy()
    target_states = [hidden_states[ind][-1] for ind in indices]
    if target_states:
        target_states = torch.cat(target_states).squeeze().unsqueeze(0)
        seg_states = model.text2seg_3d(target_states)
        last_feats = new_img_embeds_list[-1]
        last_feats = last_feats + seg_states.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        last_feats = model.text2seg_3d_gn(last_feats)
        new_img_embeds_list[-1] = last_feats
        seg_preds = model.visual_encoder_3d(encoder_only=False, x_=new_img_embeds_list)
        seg_probs = F.sigmoid(seg_preds)
        seg_mask = seg_probs.to(torch.float32).cpu().squeeze().numpy() >= 0.5
        return seg_mask

def task_det_2d(model, preds, hidden_states):
    token_mask = preds == model.det_token_idx
    indices = torch.where(token_mask == True)[0].cpu().numpy()
    target_states = [hidden_states[ind][-1] for ind in indices]
    if target_states:
        target_states = torch.cat(target_states).squeeze()
        det_states = model.text_det(target_states).detach().cpu()
        return det_states.to(torch.float32).numpy()
    return torch.zeros_like(indices)

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def resize_back_volume_abd(img, target_size):
    desired_depth = target_size[0]
    desired_width = target_size[1]
    desired_height = target_size[2]

    current_depth = img.shape[0] # [d, w, h]
    current_width = img.shape[1] 
    current_height = img.shape[2]
 
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=0)
    return img

def resize_volume_abd(img):
    img[img<=-200] = -200
    img[img>=300] = 300

    desired_depth = 64
    desired_width = 192
    desired_height = 192

    current_width = img.shape[0] # [w, h, d]
    current_height = img.shape[1]
    current_depth = img.shape[2]
 
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=0)
    return img

def load_and_preprocess_image(image, modality, task):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    if task.lower() == 'vqa':
        transform = transforms.Compose([
            transforms.Resize([512, 512]),
            # transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    image = transform(image).type(torch.bfloat16).unsqueeze(0)
    return image

def load_and_preprocess_volume(image):
    img  = nib.load(image).get_fdata()
    image = torch.from_numpy(resize_volume_abd(img)).permute(2,0,1)
    transform = tio.Compose([
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])
    image = transform(image.unsqueeze(0)).type(torch.bfloat16)
    return image

def read_image(image_path, modality, task):
    if image_path.endswith(('.jpg', '.jpeg', '.png')):
        return load_and_preprocess_image(Image.open(image_path).convert('RGB'), modality, task)
    elif image_path.endswith('.nii.gz'):
        return load_and_preprocess_volume(image_path)
    else:
        raise ValueError("Unsupported file format")

def generate(model, image_path, image, context, modal, task, num_imgs, prompt, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature, device):
    if task.lower() == 'report' or (modal.lower() == 'derm' and task.lower() == 'classification'):
        if len(context) == 0 and task.lower() == 'report':
            context = 'Comparison:None.'
        prompt = '<context>' + context + '</context>' + prompt
    img_embeds, atts_img, img_embeds_list = model.encode_img(image.unsqueeze(0), modals = [modal.lower()], task_types = [task.lower()])
    if task.lower() == 'vqa':
        placeholder = ['<ImageHere>'] * 16
    else:
        placeholder = ['<ImageHere>'] * 9
    prefix = '###Human:' + ''.join([f'<img{i}>' + ''.join(placeholder) + f'</img{i}>' for i in range(num_imgs)])
    img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, [prefix], [num_imgs])
    prompt += '###Assistant:'
    prompt_tokens = model.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(image.device)
    new_img_embeds, new_atts_img = model.prompt_concat(img_embeds, atts_img, prompt_tokens)

    outputs = model.llama_model.generate(
        inputs_embeds=new_img_embeds,
        max_new_tokens=450,
        stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub(stops=[
            torch.tensor([835]).type(torch.bfloat16).to(image.device),
            torch.tensor([2277, 29937]).type(torch.bfloat16).to(image.device)
        ])]),
        num_beams=num_beams,
        do_sample=do_sample,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=float(repetition_penalty),
        length_penalty=length_penalty,
        temperature=temperature,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    
    hidden_states = outputs.hidden_states
    preds = outputs.sequences[0]
    output_image = None
    seg_mask_2d = None
    seg_mask_3d = None
    if sum(preds == model.seg_token_idx_2d) and (modal.lower() == 'cxr' or modal.lower() == 'derm'):
        seg_mask = task_seg_2d(model, preds, hidden_states, image)
        output_image, seg_mask_2d = seg_2d_process(image_path, seg_mask)
    if sum(preds == model.seg_token_idx_2d) and (modal.lower() != 'cxr' and modal.lower() != 'derm'):
        seg_mask = task_sam_2d(model, preds, hidden_states, image)
        output_image, seg_mask_2d = seg_2d_process(image_path, seg_mask)
    if sum(preds == model.seg_token_idx_3d):
        seg_mask = task_seg_3d(model, preds, hidden_states, img_embeds_list)
        output_image, seg_mask_3d = seg_3d_process(image_path, seg_mask)
    if sum(preds == model.det_token_idx):
        det_box = task_det_2d(model, preds, hidden_states)
        output_image = det_2d_process(image_path, det_box)
    
    if preds[0] == 0:  # Remove unknown token <unk> at the beginning
        preds = preds[1:]
    if preds[0] == 1:  # Remove start token <s> at the beginning
        preds = preds[1:]
    
    output_text = model.llama_tokenizer.decode(preds, add_special_tokens=False)
    output_text = output_text.split('###')[0].split('Assistant:')[-1].strip()

    return output_image, seg_mask_2d, seg_mask_3d, output_text

def generate_predictions(model, images, context, prompt, modality, task, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature, device):
    num_imgs = len(images)
    modal = modality.lower()
    image_tensors = [read_image(img, modality, task).to(device) for img in images]
    if modality == 'ct volume':
        time.sleep(2)
    else:
        time.sleep(1)
    image_tensor = torch.cat(image_tensors)
    
    with torch.autocast(device):
        with torch.no_grad():
            generated_image, seg_mask_2d, seg_mask_3d, output_text = generate(model, images, image_tensor, context, modal, task, num_imgs, prompt, num_beams, do_sample, min_length, top_p, repetition_penalty, length_penalty, temperature, device)
    
    return seg_mask_2d, seg_mask_3d, output_text
