import matplotlib.pyplot as plt
import numpy as np

def save_image_with_masks(image_np, input_coords, input_labels, masks, marker_size=375):
    plt.figure(figsize=(10, 10))
    bounding_boxes = []

    for i, mask in enumerate(masks):
        image_np[mask != 0] = [0, 255, 0] 

        non_zero_coords = np.argwhere(mask != 0)
        min_y, min_x = np.min(non_zero_coords, axis=0)
        max_y, max_x = np.max(non_zero_coords, axis=0)

        image_np[min_y:max_y, min_x, :] = [0, 0, 255] 
        image_np[min_y:max_y, max_x, :] = [0, 0, 255]
        image_np[min_y, min_x:max_x, :] = [0, 0, 255]
        image_np[max_y, min_x:max_x, :] = [0, 0, 255]

        bounding_boxes.append([min_y, min_x, max_y, max_x])

    plt.imshow(image_np)
    
    pos_points = input_coords[input_labels == 1]
    neg_points = input_coords[input_labels == 0]
    plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    plt.axis('on')

    if len(input_coords) > 1:
        filename = '/media/jorrit/SegmentAnything/output_plots/inputs_and_'
    else:
        filename = '/media/jorrit/SegmentAnything/output_plots/input_and_'
    if len(masks) > 1:
        filename += 'masks.png'
    else:
        filename += 'mask.png'
    plt.savefig(filename)

    return bounding_boxes

def save_image_output_masks(masks):
    for i, mask in enumerate(masks):
        plt.figure()
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {i+1}')
        plt.axis('off')

    plt.savefig('/media/jorrit/SegmentAnything/output_plots/output_mask.png')

