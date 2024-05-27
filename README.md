## Mirror3DGS 
An unofficial implementation of [**Mirror3DGS**](https://arxiv.org/pdf/2404.01168).  
  
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
## TODOs  
- train / test split 
- stage1: coarse stage: 2dgs with mask 
- stage2: estimate mirror plane 
- stage3: full rendering  

## Usage 
```bash   
python train.py -s data/colmap/real/discussion_room/ --eval  --resolution 4

# convert gs to mesh 
python render.py -m ./output/9c059d97-2 -s data/colmap/real/discussion_room/ --eval  --skip_mesh --render_path --resolution 4
```

## Implementation Notes

### Using colmap 

## Acknowledgement 
Special thanks to the projects and their contributors:
* [Mirror3DGS](https://arxiv.org/pdf/2404.01168) 
* [MirrorGaussian](https://mirror-gaussian.github.io/) 
 