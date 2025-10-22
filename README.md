# face-wrap
State-of-the-Art Face-Swap Technique in Digital Video

requirements:
1. 從 https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth 下載模型，下載後放在experiments/pretrained_models/下
2. pip3 install -r requirements.txt 安裝必要套件

arguments:
--source: Face to swap onto the target
--input: Input video
--output: Output video
--reference: Reference face for main character
--gfpgan: Use GFPGAN enhancement
--threhold: Cosine similarity threshold
ex: python face-swap.py --source data/source.jpg --reference data/reference.jpg --input data/input.mp4 --output results/output.mp4 --gfpgan --threshold 0.3

bug:
from torchvision.transforms.functional_tensor import rgb_to_grayscale
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
如果出現上面錯誤:
from torchvision.transforms.functional_tensor import rgb_to_grayscale
改成
from torchvision.transforms.functional import rgb_to_grayscale