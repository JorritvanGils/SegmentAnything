from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# Load the ViT-H SAM model checkpoint
checkpoint_path = "/media/jorrit/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)

# Create a predictor
predictor = SamPredictor(sam)

# Set an example image (replace with your image path)
image_path = "/media/jorrit/segment-anything/images_inference/Sucre.jpeg"
image = Image.open(image_path)
image_np = np.array(image)
print('image_np.shape:', image_np.shape)
predictor.set_image(image_np)

# Example: Get masks from input prompts
input_prompts = ["sky", "trees", "buildings"]

# Add empty point_coords and point_labels to avoid the assertion error
input_point = np.array([[50, 100]])
input_label = np.array([1])

plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  

masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label)
# masks, _, _ = predictor.predict(input_prompts, point_coords=input_point, point_labels=input_label)

print('masks.shape:', masks.shape)

# print('masks:', masks)
# for mask in masks:
#     print('mask:', mask)
#     print('mask.shape:', mask.shape)

# # Assuming masks is a list of binary masks
# for i, mask in enumerate(masks):
#     plt.subplot(1, len(masks), i + 1)
#     plt.imshow(mask, cmap='gray')
#     plt.title(f'Mask for prompt "{input_prompts[i]}"')
#     plt.axis('off')

# plt.show()






# from segment_anything import SamPredictor, sam_model_registry
# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)
