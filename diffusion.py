from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline

import torch

class ImageGenerator:
    def __init__(self, device):

        self.pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        )
        self.pipe_prior.to(device)

        self.pipe = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        )
        self.pipe.to(device)

    def generate(self, encodings, negative_prompt = ''):
        with torch.no_grad():
            out_zero = self.pipe_prior(
                negative_prompt,
            )
            zero_image_emb = out_zero.negative_image_embeds if negative_prompt == "" else out_zero.image_embeds
            zero_image_emb = zero_image_emb.repeat(encodings.shape[0], 1)
            print(zero_image_emb.shape)
            images = self.pipe(
                # prompt = 'panda sitting on tree',
                image_embeds=encodings,
                negative_image_embeds=zero_image_emb,
                height=768,
                width=768,
                strength=0,
                # guidance_scale = 10,
                num_inference_steps=50,
                add_predicted_noise=False
            )

        return images

