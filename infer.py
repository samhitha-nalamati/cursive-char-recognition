"""
Classify a single character image with the trained model.

Usage:
    python infer.py --checkpoint checkpoints/best.pt --image samples/my_char.png --topk 3
"""
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from model import CharCNN
from labels import NUM_CLASSES, IDX_TO_CLASS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    img_size = ckpt["img_size"]

    model = CharCNN(num_classes=NUM_CLASSES, img_size=img_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = T.Compose([
        T.Grayscale(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    img = Image.open(args.image)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(args.topk)
    print("Top predictions:")
    for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        print(f"    {IDX_TO_CLASS[idx]!r}: {prob:.3f}")


if __name__ == "__main__":
    main()
