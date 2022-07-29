import time
import cv2
import numpy as np
import onnxruntime

from .utils import PlaneCluster

rng = np.random.default_rng(0)
colors = rng.uniform(0, 255, size=(50, 3))

class PlanarReconstruction:

    def __init__(self, path, init_cluster_num=10, max_cluster_distance=70):

        self.planeCluster = PlaneCluster(init_cluster_num, max_cluster_distance)

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.update(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def update(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        self.outputs = self.inference(input_tensor)

        # Process output data
        self.segmentation, self.cluster_normals = self.process_output(self.outputs)

        return self.segmentation, self.cluster_normals

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Normalize input image
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_img = ((input_img / 255.0 - mean) / std)
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()

        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, outputs):

        prob, embedding, param = outputs

        prob = prob.squeeze()
        embedding = embedding.squeeze()
        param = param.squeeze() # Normal direction

        # Get segmentation
        segmentation = self.planeCluster(embedding)

        # Get the avg. param of each segmentation cluster
        param = param.transpose(1, 2, 0)
        cluster_normals = np.zeros((segmentation.max()+1, 3))
        for i in range(segmentation.max() + 1):
            if param[segmentation == i].shape[0] == 0:
                continue
            cluster_normals[i] = param[segmentation == i].mean(axis=0)
        segmentation += 1

        return segmentation, cluster_normals

    def draw(self, image, alpha=0.5):

        # Draw segmentation
        color_segmap = colors[self.segmentation, :].astype(np.uint8)
        color_segmap = cv2.resize(color_segmap, (image.shape[1], image.shape[0]))

        # Fuse both images
        if alpha == 0:
            combined_img = np.hstack((image, color_segmap))
        else:
            combined_img = cv2.addWeighted(image, alpha, color_segmap, (1 - alpha), 0)

        return combined_img

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    model_path = "../models/plane_ae_sim.onnx"

    # Initialize model
    planeSeg = PlanarReconstruction(model_path)

    img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/0/0d/Bedroom_Mitcham.jpg")

    # Perform the inference in the image
    segmentation, cluster_normals = planeSeg(img)

    # Draw model output
    output_img = planeSeg.draw(img, alpha=0.2)
    cv2.namedWindow("Detected planes", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected planes", output_img)
    cv2.waitKey(0)
