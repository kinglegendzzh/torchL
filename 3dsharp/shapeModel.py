"""
显卡不好千万别跑
"""
import torch
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif, load_image

ckpt_id = "openai/shap-e-img2img"
pipe = ShapEImg2ImgPipeline.from_pretrained(ckpt_id).to("cuda")

img_url = "https://hf.co/datasets/diffusers/docs-images/resolve/main/shap-e/corgi.png"
image = load_image(img_url)


generator = torch.Generator(device="cuda").manual_seed(0)
batch_size = 1
guidance_scale = 3.0

images = pipe(
    image,
    num_images_per_prompt=batch_size,
    generator=generator,
    guidance_scale=guidance_scale,
    num_inference_steps=2,
    size=128,
    output_type="pil"
).images

gif_path = export_to_gif(images, "corgi_sampled_3d.gif")
