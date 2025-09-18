import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
import random


# valori di normalizzazione per ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def load_img(image_path, target_width):
    
    # carichiamo l'immagine.
    image = Image.open(image_path).convert("RGB")
    
    # riscala l'immagine (di default non lo facciamo)
    if target_width is not None:
        w, h = image.size
        aspect_ratio = h / w
        new_height = int(target_width * aspect_ratio)
        image = image.resize((target_width, new_height))
        
    # convertiamo image (oggetto PIL) in tensore (PyTorch)
    # in particolare, forma tensore: forma [C, H, W] normalizzato in [0, 1] float
    tensor = transforms.ToTensor()(image)

    # aggiungiamo la dimensione del batch (input modello)
    # risultato: tensore forma [1, C, H, W]
    tensor = tensor.unsqueeze(0) 

    return tensor



def resize_img(tensor, size):
    return F.interpolate(tensor, size=size, mode='bilinear')



def save_img(tensor):

    # il tensore deve essere sulla CPU.
    tensor = tensor.cpu()
    # rimuoviamo la dimensione del batch.
    tensor = tensor.squeeze(0)
    # tensore con valori in [0, 1].
    tensor = tensor.clamp(0, 1)

    # convertiamo da tensore PyTorch a immagine PIL.
    image = transforms.ToPILImage()(tensor)

    # creiamo file di output con timestamp.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"dream_{ts}.jpg"
    # salviamo l'immagine.
    image.save(output_path)

    return output_path



def shift(img):
    
    h_shift = random.randint(-32, 32)   
    w_shift = random.randint(-32, 32)

    # shift circolare.
    img = torch.roll(img, shifts=(h_shift, w_shift), dims=(-2, -1))

    return img



def clamp(img):
    # ci assicuriamo di fare il broadcasting,
    # e nel device corretto.
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(img.device)
    std  = IMAGENET_STD.view(1, 3, 1, 1).to(img.device)

    lower = (0 - mean) / std
    upper = (1 - mean) / std

    img = img.clamp_(lower, upper)

    return img



def preprocess_img(img):

    # ci assicuriamo di fare il broadcasting,
    # e nel device corretto.
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(img.device)
    std  = IMAGENET_STD.view(1, 3, 1, 1).to(img.device)
    
    # normalizzazione.
    img = (img - mean) / std

    return img



def postprocess_img(img):
    
    # ci assicuriamo di fare il broadcasting,
    # e nel device corretto.
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(img.device)
    std  = IMAGENET_STD.view(1, 3, 1, 1).to(img.device)
    
    # de-normalizzazione.
    out = (img * std) + mean

    # clampiamo in [0, 1].
    out = out.clamp(0, 1)
    
    return out