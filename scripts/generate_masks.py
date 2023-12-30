from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
from functions import save_image_with_masks
import matplotlib.pyplot as plt


# ViT-B SAM model checkpoint 
checkpoint_path = "/media/jorrit/SegmentAnything/checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"

# # ViT-H SAM model checkpoint
# checkpoint_path = "/media/jorrit/SegmentAnything/checkpoints/sam_vit_h_4b8939.pth"
# model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)

# Set an example image (replace with your image path)
image_path = "/media/jorrit/SegmentAnything/images_inference/sucre.jpeg"
image = Image.open(image_path)
image_np = np.array(image)
print('image_np.shape:', image_np.shape)
predictor.set_image(image_np)


# # Single input point
# input_point = np.array([[50, 100]])
# input_label = np.array([1])

# Multiple input points
input_point = np.array([[50, 100], [60, 80]])
input_label = np.array([1, 1]) 

# get masks
masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
print('masks.shape:', masks.shape)
print('len(masks)', len(masks))

bounding_boxes = save_image_with_masks(image_np, input_point, input_label, masks)
print('Bounding Boxes:', bounding_boxes)
