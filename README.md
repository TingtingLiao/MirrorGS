## Mirror3DGS 
An unofficial implementation of [**Mirror3DGS**](https://arxiv.org/pdf/2404.01168).  
  
## Install
```bash
git clone  --recursive https://github.com/TingtingLiao/MirrorGS.git 
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
## TODOs 
- depth initilization (dreanscene360)
- plane regularizer
- rendering 

## Usage 
```bash 
python train.py -s data/colmap/synthetic/livingroom/

# convert gs to mesh
python train.py -s data/colmap/synthetic/livingroom/ 
```

## Implementation Notes

### Using colmap 

## Acknowledgement 
Special thanks to the projects and their contributors:
* [Mirror3DGS](https://arxiv.org/pdf/2404.01168) 
* [MirrorGaussian](https://mirror-gaussian.github.io/) 
 