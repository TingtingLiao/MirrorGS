## MirrorGS 
An unofficial implementation of [**Mirror3DGS: Incorporating Mirror Reflections
into 3D Gaussian Splatting**](https://arxiv.org/pdf/2404.01168).  
 
https://github.com/TingtingLiao/MirrorGS/assets/45743512/b78de686-58db-4c8f-8668-8ceeaca7e75c


https://github.com/TingtingLiao/MirrorGS/assets/45743512/94dab8a3-a6ef-4144-b692-e0c738874827
 

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
python train.py -s data/colmap/real/discussion_room/  --resolution 4 --iterations 20000
  
# validation 
python render.py -m ./output/dfaf0f54-8 -s data/colmap/real/discussion_room/ --skip_mesh --render_path --resolution 4
```

## Implementation Notes

<!-- ### Mirror Plane
For any point $p = (x, y, z)^⊤$ on the plane satisfies:

$$n^{T}p+d=0$$

This equals to minimizing the least squares error:

$$\sum_{i=1}^{m} (n^T(p_i - \mathbf{c}))^2$$

we approximate the solution as $\mathbf{c} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{p}_i$, $d=-n^T\mathbf{c}$ and normal $n$ of the plane as the eigenvector of the smallest eigenvalue of the covariance matrix:

$$\sum_{i=1}^{m} (\mathbf{p}_i - \mathbf{c})(\mathbf{p}_i - \mathbf{c})^T$$ -->



## Acknowledgement 
Special thanks to the projects and their contributors:
* [Mirror3DGS](https://arxiv.org/pdf/2404.01168) 
* [MirrorGaussian](https://mirror-gaussian.github.io/) 
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
* [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D)