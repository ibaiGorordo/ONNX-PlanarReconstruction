import cv2
from imread_from_url import imread_from_url

from PlanarReconstruction import PlanarReconstruction

model_path = "models/plane_ae_sim.onnx"

# Initialize model
planeSeg = PlanarReconstruction(model_path)

img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/0/0d/Bedroom_Mitcham.jpg")

# Perform the inference in the image
segmentation, cluster_normals = planeSeg(img)

# Draw model output
output_img = planeSeg.draw(img, alpha=0.2)
cv2.namedWindow("Detected planes", cv2.WINDOW_NORMAL)
cv2.imshow("Detected planes", output_img)
cv2.imwrite("doc/img/planes.png", output_img)
cv2.waitKey(0)
