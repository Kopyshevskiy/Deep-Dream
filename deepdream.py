import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import gaussian_blur
from image_utils import preprocess_img, postprocess_img, resize_img, shift, clamp


def gradient_ascent(extractor, img, args):

    pyramid_levels = args.pyramid_levels
    iterations = args.iterations
    layers = args.layers
    lr = args.learning_rate

    # non teniamo traccia della storia delle operazioni sul tensore img.
    img = img.detach()             

    # questa è l'istruzione più importante.
    # i gradienti vengono calcolati rispetto all'immagine.
    img = img.requires_grad_(True)  
    
    for i in range(iterations):

        # di default, PyTorch accumula i gradienti.
        # di conseguenza, dobbiamo azzerare il gradiente accumulato,
        # in maniera da considerare (solo) il gradiente sull'immagine attuale
        # che cambia ad ogni iterazione.
        img.grad = None

        # jitter.
        img_shift = shift(img)

        # forward modello.
        # attenzione: calcoliamo tutto su img_shift, ma poi aggiorniamo img.
        outs = extractor(img_shift) # modificato ora
        # otteniamo le activation maps che vogliamo massimizzare.
        activations = [outs[n] for n in layers]
        # calcoliamo la loss finale.
        loss = torch.stack([a.norm() for a in activations]).mean()
        # calcoliamo i gradienti della loss rispetto all'immagine
        loss.backward()

        
        # otteniamo i gradienti e normalizziamo (stabilità).
        grad = img.grad
        grad = (grad) / (grad.abs().mean() + 1e-8)
        grad = gaussian_blur(grad, kernel_size=[9, 9], sigma=[2, 2])

        # disabilitiamo il calcolo dei gradienti per le operazioni che seguono.
        with torch.no_grad():

            # passo di gradient ascent.
            # nota interessante: se si scrivesse img = img + lr * grad,
            # si creerebbe un nuovo tensore img, con requires_grad=False 
            # (out-of-place operation).
            img += lr * grad

            # clampiamo i valori dei pixel.
            img = clamp(img)

        if i % 5 == 0:
            print(f"[INFO] iterazione {i}/{iterations} del livello {pyramid_levels} completata. Loss: {loss.item():.4f}")
    

    return img



def deep_dream(model, img, args):

    layers = args.layers
    pyramid_levels = args.pyramid_levels
    pyramid_ratio  = args.pyramid_ratio

    # pre-processing: lo facciamo subito.
    # necessario siccome usiamo modelli pre-trainati su ImageNet.
    # nota: volendo, si potrebbe il pre-processing per ogni step del gradient ascent,
    # in quanto non abbiamo garanzia che dopo uno step del gradient ascent mean sia zero e std 1.
    # tuttavia, in pratica, non sembra essere un problema.
    img = preprocess_img(img)
    

    # prendiamo la terza e quarta dimensione del tensore
    # rispettivamente: height e width
    h, w = img.shape[2:]

    # estrattore per i layer richiesti.
    extractor = create_feature_extractor(model, return_nodes={n: n for n in layers})

    for lv in range(pyramid_levels):
       
        # calcola le nuove dimensioni per il livello corrente della piramide.
        # si parte da una versione ridotta dell'immagine,
        # e si arriva alla dimensione originale
        new_h = int(h * (pyramid_ratio ** (lv - pyramid_levels + 1)))
        new_w = int(w * (pyramid_ratio ** (lv - pyramid_levels + 1)))        
        img = resize_img(img, (new_h, new_w))  
      

        # applica il gradient ascent
        img = gradient_ascent(extractor, img, args)
        
        print(f"[INFO] livello {lv+1}/{pyramid_levels} completato!")

    # post-processing
    out = postprocess_img(img)

    return out