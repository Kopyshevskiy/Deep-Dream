import argparse

import torch
from torchvision.models import vgg16, resnet50, alexnet, googlenet

from deepdream   import deep_dream
from image_utils import load_img, save_img


def main():
    
    parser = argparse.ArgumentParser(description="DeepDream in PyTorch")

    # percorso immagine di input.
    parser.add_argument("--input_image", type=str, required=True)
    # larghezza a cui ridimensionare (eventualmente) l'immagine.
    parser.add_argument("--image_width", type=int, default=None)
    # scelta modello di rete neurale.
    parser.add_argument("--model", type=str, default='vgg16', choices=["vgg16", "resnet50", "alexnet", "googlenet"])
    # nome del layer (o dei layer) da massimizzare.    
    parser.add_argument("--layers", type=str, nargs='+', required=True)


    # numero di livelli nella piramide di immagini (di default: 4).
    parser.add_argument("--pyramid_levels", type=int, default=4)
    # rapporto di scala tra i livelli della piramide (di default: 1.8).
    parser.add_argument("--pyramid_ratio", type=float, default=1.8)
    # numero di iterazioni di gradient ascent, per ciascun livello della piramide.
    parser.add_argument("--iterations", type=int, default=10)
    # "learning rate" del gradient ascent.
    parser.add_argument("--learning_rate", type=float, default=0.09)

    
    args = parser.parse_args()


    # caricamento del modello.
    print(f"[INFO] caricamento del modello: {args.model}")
    if args.model == 'vgg16':
        model = vgg16(pretrained=True)
    elif args.model == 'resnet50':
        model = resnet50(pretrained=True)
    elif args.model == 'alexnet':
        model = alexnet(pretrained=True)
    elif args.model == 'googlenet':
        model = googlenet(pretrained=True)
    
    # disabilitiamo calcolo dei gradienti per i parametri del modello.
    for p in model.parameters():
      p.requires_grad_(False)
   
    # mettiamo modello modalità inferenziale (disabilitando dropout e batchnorm).
    model.eval()

    # impostazione del dispositivo (GPU se disponibile).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)


    # caricamento immagine.
    img = load_img(args.input_image, args.image_width).to(device)
    print(f"[INFO] immagine caricata!")


    # esecuzione di DeepDream.
    print("[INFO] avvio di DeepDream.")
    dream_img = deep_dream(model, img, args)
    print("[INFO] DeepDream completato.")


    # genera un nome file di output se non fornito.    
    save_img(dream_img)
    print(f"[INFO] immagine salvata!")


if __name__ == "__main__":
    main()