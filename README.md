## Introduction
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
 
```
 

## Acknowledgement 
Special thanks to the projects and their contributors:
* [DreamScene360](https://dreamscene360.github.io/)
* [Diffusion360](https://github.com/ArcherFMY/SD-T2I-360PanoImage)
* [360monodepth](https://github.com/manurare/360monodepth)
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
* [Equirec2Perspec](https://github.com/fuenwang/Equirec2Perspec)