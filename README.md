## MirrorGS 
An unofficial implementation of [**Mirror3DGS: Incorporating Mirror Reflections
into 3D Gaussian Splatting**](https://arxiv.org/pdf/2404.01168).  
  
https://github.com/TingtingLiao/MirrorGS/assets/45743512/e6779cd6-28ec-4e70-9fba-30de6ab92253

https://github.com/TingtingLiao/MirrorGS/assets/45743512/2b8e019d-fbd4-4136-93d5-1cd760095033

## Implementation Notes

You can find detailed explainations from this [blog](https://tingtingliao.github.io/blog/2024/MirrorGS/). 
  
## Install
```bash
git clone --recursive https://github.com/TingtingLiao/MirrorGS.git 
cd MirrorGS
conda create -n mirrorgs python=3.10 
conda activate mirrorgs 

# torch2.3.0+cu12.1 
pip install torch torchvision torchaudio
 
# requirements
pip install -r requirements.txt

# xformers  
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
 
pip install -e submodules/diff-surfel-rasterization 
pip install -e submodules/simple-knn 
``` 

## Data
We provide an example. Please download it from [here](https://drive.google.com/file/d/1XTgvua2WPXJZmGPhJXGaIMHhaAl-cPBZ/view?usp=sharing). 
 
For customized data, the following steps are required:

![output](https://github.com/TingtingLiao/MirrorGS/assets/45743512/aadc9422-0ee5-4bf4-8b54-cffa8f03fb7d)

- mirror segmention 
- remove the mirror area (the right image), then run colmap 

```bash    
colmap automatic_reconstructor --workspace_path data/{}  --image_path data/{}/images --camera_model SIMPLE_PINHOLE  
``` 

## Usage 
```bash   
python train.py -s data/{}   
  
# validation 
python render.py -m ./output/{your_path} -s data/{} --skip_mesh  
```
  
## Acknowledgement 
Special thanks to the projects and their contributors:
* [Mirror3DGS](https://arxiv.org/pdf/2404.01168) 
* [MirrorGaussian](https://mirror-gaussian.github.io/) 
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
* [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D)
