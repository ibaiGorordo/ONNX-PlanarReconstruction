# ONNX PlanarReconstruction (WIP)
 Python scripts performing plane segmentation using PlanarReconstruction model in ONNX.

![! ONNX PlanarReconstruction Plane Segmentation](https://github.com/ibaiGorordo/ONNX-PlanarReconstruction/blob/main/doc/img/planes.png)
*Original image: https://commons.wikimedia.org/wiki/File:Bedroom_Mitcham.jpg*

# Important
- The post processing is not completelly the same as the original implementation. 
- The Mean Shift algorithm is done with Scikit Learn which takes a lot of time.

# Requirements

 * Check the **requirements.txt** file. 
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
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

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

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