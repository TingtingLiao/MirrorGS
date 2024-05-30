## MirrorGS 
An unofficial implementation of [**Mirror3DGS: Incorporating Mirror Reflections
into 3D Gaussian Splatting**](https://arxiv.org/pdf/2404.01168).  
 

https://github.com/TingtingLiao/MirrorGS/assets/45743512/b31d3a28-24fa-4389-8d9a-4d246feb6778

https://github.com/TingtingLiao/MirrorGS/assets/45743512/92f96bba-f382-4a73-9a4d-ba97c26ec863

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

## Usage 
```bash   
python train.py -s data/colmap/real/discussion_room/  --resolution 4 
  
# validation 
python render.py -m ./output/{your_path} -s data/colmap/real/discussion_room/ --skip_mesh --resolution 4
```

## Implementation Notes

I wrote a [blog](https://tingtingliao.github.io/blog/2024/MirrorGS/) where you can find detailed explainations. 
  

## Acknowledgement 
Special thanks to the projects and their contributors:
* [Mirror3DGS](https://arxiv.org/pdf/2404.01168) 
* [MirrorGaussian](https://mirror-gaussian.github.io/) 
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
* [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D)