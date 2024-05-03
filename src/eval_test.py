import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform 
import json


model_path = "output/cyclegan_turbo/food_extended/checkpoints/model_11501.pkl"
direction = "b2a"
fixed_prompt = "a typical Indian food,  authentic food from India"
input_folder = "data/food_extended/test_B"
input_saving_path = os.path.join("inference/inputs", "japan")
output_saving_path = os.path.join("inference/outputs", "japan2india_unsup_edited")
captions_path = "data/food_extended/captions_b.json"

os.makedirs(output_saving_path, exist_ok=True)

if __name__ == "__main__":
    img_prep = "resize_256"
    T_val = build_transform(img_prep)
    model = CycleGAN_Turbo(pretrained_name=None, pretrained_path=model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()

    with open(captions_path, "r") as json_file:
        captions = json.load(json_file)


    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        input_image = Image.open(input_path).convert('RGB')
        prompt = captions[filename]["cap_edited"] if filename in captions else fixed_prompt
    # translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            output = model(x_t, direction=direction, caption=prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

        #input_img.save(os.path.join(input_saving_path, filename))
        
        output_pil.save(os.path.join(output_saving_path, filename))

        print(f"image {filename} processed with caption '{prompt}'")


   