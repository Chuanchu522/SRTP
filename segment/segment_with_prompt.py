import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import os


"""
## Set-up
"""

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def prompt_img_layers(folder_name, img, anns):
    if len(anns) == 0:
        return [img]

    # Ensure the form of the image uses color space RGBA
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img[:, :, [0, 1, 2, 3]] = img[:, :, [2, 1, 0, 3]]

    layers = []
    cnt = 0
    for ann in anns:
        cnt = cnt + 1

        # Acquire the mask of the current object
        m = ann.astype(np.uint8)

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

def batched_img_layers(folder_name, img, anns):
    # Ensure the form of the image uses color space RGBA
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img[:, :, [0, 1, 2, 3]] = img[:, :, [2, 1, 0, 3]]

    if len(anns) == 0:
        return [img]

    layers = []
    cnt = 0
    for ann in anns:

        ann_masks = ann.cpu().numpy()

        if len(ann_masks) == 0:
            break

        for ann_mask in ann_masks:
            cnt = cnt + 1

            # Acquire the mask of the current object
            m = ann_mask.astype(np.uint8)

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

    if len(layers) == 0:
        return [img]

    return layers

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


"""
## Input image
"""

image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()

predictor.set_image(image)


"""
## Specifying a specific object with a single points
"""

# The position of the single prompt point
input_point = np.array([[500, 375]])
# The label of the single prompt point, where 1 respresent positive point and 0 represent the negative point
input_label = np.array([1])

# Show the single prompt point on the input image
plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

# Predict the mask
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Show the predicted mask
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()


# Save the object layers in the folder1
folder1_1 = 'single_point_layers1'
single_prompt_layers1 = prompt_img_layers(folder1_1, image, masks[0:1,:,:])
folder1_2 = 'single_point_layers2'
single_prompt_layers2 = prompt_img_layers(folder1_2, image, masks[1:2,:,:])
folder1_3 = 'single_point_layers3'
single_prompt_layers3 = prompt_img_layers(folder1_3, image, masks[2:3,:,:])

"""
## Specifying a specific object with additional points
"""

# The position of the two positive prompt points
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])

# Choose the model's best mask
mask_input = logits[np.argmax(scores), :, :]

# Predict the mask
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

# Show the masks with two positive prompt points
plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

# Save the object layers in the folder2
folder2 = 'pos_2points_layers'
pos_2points_layers = prompt_img_layers(folder2, image, masks)


# The position of the two prompt points, where the first one is positive and the other one is negative
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 0])

# Predict the mask
mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

# Show the predicted mask with two positive prompt points
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

# Show the mask with one positive prompt points and one negative prompt points
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

# Save the object layers in the folder3
folder3 = 'pos_neg_2points_layers'
pos_neg_2points_layers = prompt_img_layers(folder3, image, masks)


"""
## Specifying a specific object with a box
"""

# The position information of the prompt box
input_box = np.array([425, 600, 700, 875])

# Predict the mask
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

# Show the predicted mask with a prompt box
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.show()

# Save the object layers in the folder4
folder4 = 'single_box_layers'
single_box_layers = prompt_img_layers(folder4, image, masks)


"""
## Combining points and boxes
"""

# The position information of a prompt box and a negative prompt point
input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])

# Predict the mask
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)

# Show the predicted mask with a prompt box and a negative prompt point
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

# Save the object layers in the folder5
folder5 = 'box_point_layers'
box_point_layers = prompt_img_layers(folder5, image, masks)


"""
## Batched prompt inputs
"""

# The position information of multiple prompt boxes
input_boxes = torch.tensor([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
], device=predictor.device)

# Predict the mask
# The shape of the mask is (batch_size) x (num_predicted_masks_per_input) x H x W
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# Show the prediected mask with the prompt boxes
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()

# Save the object layers in the folder6
folder6 = 'batch_boxes_layers'
batch_boxes_layers = batched_img_layers(folder6, image, masks)


"""
## End-to-end batched inference
"""

# The first image with its prompt boxes
image1 = image  # truck.jpg from above
image1_boxes = torch.tensor([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
], device=sam.device)

# The second image with its prompt boxes
image2 = cv2.imread('images/groceries.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2_boxes = torch.tensor([
    [450, 170, 520, 350],
    [350, 190, 450, 350],
    [500, 170, 580, 350],
    [580, 170, 640, 350],
], device=sam.device)


resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

batched_input = [
     {
         'image': prepare_image(image1, resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]),
         'original_size': image1.shape[:2]
     },
     {
         'image': prepare_image(image2, resize_transform, sam),
         'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
         'original_size': image2.shape[:2]
     }
]

batched_output = sam(batched_input, multimask_output=False)

batched_output[0].keys()

# Show the predict masks with prompt boxes of the two input images respectly
fig, ax = plt.subplots(1, 2, figsize=(20, 20))

ax[0].imshow(image1)
for mask in batched_output[0]['masks']:
    show_mask(mask.cpu().numpy(), ax[0], random_color=True)
for box in image1_boxes:
    show_box(box.cpu().numpy(), ax[0])
ax[0].axis('off')

# Save the object layers in the folder7
masks1 = batched_output[0]['masks']
folder7 = 'img1_batch_boxes_layers'
img1_batch_boxes_layers = batched_img_layers(folder7, image1, masks1)

ax[1].imshow(image2)
for mask in batched_output[1]['masks']:
    show_mask(mask.cpu().numpy(), ax[1], random_color=True)
for box in image2_boxes:
    show_box(box.cpu().numpy(), ax[1])
ax[1].axis('off')

# Save the object layers in the folder8
masks2 = batched_output[1]['masks']
folder8 = 'img2_batch_boxes_layers'
img2_batch_boxes_layers = batched_img_layers(folder8, image2, masks2)

plt.tight_layout()
plt.show()

