import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import os

"""
## Set-up
"""

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def img_layers(folder_name, img, anns):
    if len(anns) == 0:
        return [img]

    # Sort the masks according to there area in descending order
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    layers = []
    cnt = 0
    for ann in sorted_anns:
        cnt = cnt + 1

        # Acquire the mask of the current object
        m = ann['segmentation']
        m = m.astype(np.uint8)

        # Extract the pixels pf the object, the transparency of other pixels is 0
        layer = img
        layer = cv2.bitwise_and(layer, layer, mask=m)
        layer[np.where(m == False)] = [0, 0, 0, 0]

        # Output the image layers in the folder
        os.makedirs(folder_name, exist_ok=True)

        filename = os.path.join(folder_name, 'layer'+str(cnt)+'.png')
        cv2.imwrite(filename, layer)

        # Add the current layer to the layers of the input image
        layers.append(layer)

    return layers

"""
## Input image
"""

image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

"""
## Automatic mask generation
"""

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()


"""
## Output image layers

Save the result layers in the folder as '.png' form and in the list image_layers
"""
folder = 'layers'

# Ensure the form of the image uses color space RGBA
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
image[:, :, [0, 1, 2, 3]] = image[:, :, [2, 1, 0, 3]]

image_layers = img_layers(folder, image, masks)
