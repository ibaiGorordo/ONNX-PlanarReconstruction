# ONNX PlanarReconstruction
 Python scripts performing plane segmentation using PlanarReconstruction model in ONNX.

![! ONNX PlanarReconstruction Plane Segmentation](https://github.com/ibaiGorordo/ONNX-PlanarReconstruction/blob/main/doc/img/planes.png)
*Original image: https://commons.wikimedia.org/wiki/File:Bedroom_Mitcham.jpg*

# Important
- The post processing is not completelly the same as the original implementation. 
- The Mean Shift was replaced with a custom method using Kmeans. It is faster (x10) than using MeanShift from scikit-learn ([previous commit](https://github.com/ibaiGorordo/ONNX-PlanarReconstruction/tree/459e0924c32c8cd6f77343f603a226550e0a8a15)), but it requires some fine tuning and is still slower than the model itself.

# Requirements

 * Check the **requirements.txt** file. 
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 
# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-PlanarReconstruction.git
cd ONNX-PlanarReconstruction
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model 
The original model was converted to ONNX using the following Colab notebook:
1. Convert the model to ONNX [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S97iUqw0T_2CjfZPz_waTj4pXXwuyR54?usp=sharing)
2. Save the download model into the **[models  folder](https://github.com/ibaiGorordo/ONNX-PlanarReconstruction/tree/main/models)**

- The License of the models is MIT: [License](https://github.com/svip-lab/PlanarReconstruction/blob/master/LICENSE)

# Original PlanarReconstruction model
The original PlanarReconstruction model can be found in this repository: [PlanarReconstruction Repository](https://github.com/svip-lab/PlanarReconstruction)
 
# Examples

 * **Image inference**:
 ```
 python image_plane_segmentation.py
 ```
 
# References:
* PlanarReconstruction model: https://github.com/svip-lab/PlanarReconstruction
* Paper: https://arxiv.org/abs/1902.09777
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
