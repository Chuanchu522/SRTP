import torch
'''
pip install --upgrade diffusers transformers scipy
'''
from diffusers import StableDiffusionPipeline
import os

'''
The model used for the test is from 
https://huggingface.co/CompVis/stable-diffusion-v1-4
'''
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

folder_name = "sd_v1_4_img"
os.makedirs(folder_name, exist_ok=True)
filename = os.path.join(folder_name, "astronaut_rides_horse.png")
image.save(filename)