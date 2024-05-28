## MirrorGS 
An unofficial implementation of [**Mirror3DGS:Incorporating Mirror Reflections
into 3D Gaussian Splatting**](https://arxiv.org/pdf/2404.01168).  
  
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
- remove noise of the mirror plane using RANSAC
- add depth supervision 


## Usage 
```bash   
python train.py -s data/colmap/real/discussion_room/ --eval  --resolution 4 --start_checkpoint ./output/580d40fc-5/point_cloud/iteration_30000/point_cloud.ply

# convert gs to mesh 
python render.py -m ./output/d9810cb3-e -s data/colmap/real/discussion_room/ --eval  --skip_mesh --render_path --resolution 4
```

## Implementation Notes

### Mirror Plane
For any point $p = (x, y, z)^⊤$ on the plane satisfies:

$$n^{T}p+d=0$$

This equals to minimizing the least squares error:

$$\sum_{i=1}^{m} (n^T(p_i - \mathbf{c}))^2$$

we approximate the solution as $\mathbf{c} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{p}_i$, $d=-n^T\mathbf{c}$ and normal $n$ of the plane as the eigenvector of the smallest eigenvalue of the covariance matrix:

$$\sum_{i=1}^{m} (\mathbf{p}_i - \mathbf{c})(\mathbf{p}_i - \mathbf{c})^T$$



## Acknowledgement 
Special thanks to the projects and their contributors:
* [Mirror3DGS](https://arxiv.org/pdf/2404.01168) 
* [MirrorGaussian](https://mirror-gaussian.github.io/) 
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)