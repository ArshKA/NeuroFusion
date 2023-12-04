import os
import torch
import numpy as np
from PIL import Image
import re
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
from diffusers.utils import load_image
from diffusion import ImageGenerator
import h5py

device = 'cuda:3'
pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
)
pipe_prior.to(device)

stimuli_list_path = '/scratch/arsh/raw/images/BOLD5000_Stimuli/Stimuli_Presentation_Lists'
stimuli_file_path = '/scratch/arsh/raw/images/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli'

with torch.no_grad(), h5py.File('/scratch/arsh/mri_h5/stimuli/stimuli_data.hdf5', 'a') as file:

  for subject in os.listdir(stimuli_list_path):
    subject_int = int(subject[-1])
    if subject_int != 1: continue
    for session in os.listdir(f'{stimuli_list_path}/{subject}'):
      if os.path.isdir(f'{stimuli_list_path}/{subject}/{session}'):
        session_int = int(session[-2:])
        group = file.require_group(f"CSI{subject_int}")

        print(f"Subject: {subject_int}, Session: {session_int}")
        runs = [x for x in os.listdir(f'{stimuli_list_path}/{subject}/{session}') if x[-4:] == '.txt']
        encodings = torch.zeros((len(runs)*37, 1280))

        for run in runs:
          run_int = int(run[-6:-4])
          stim_names = open(f'{stimuli_list_path}/{subject}/{session}/{run}', 'r').read().split('\n')
          stim_names = [x for x in stim_names if len(x) > 0]
          if len(stim_names) != 37: raise Exception()

          for i, stim in enumerate(stim_names):
            if stim[:4] == 'rep_':
              stim = stim[4:]
            if re.match('COCO_.+_\d+', stim):
              image_path = f'{stimuli_file_path}/COCO/' + stim
            elif re.match('n\d+_\d+', stim):
              image_path = f'{stimuli_file_path}/ImageNet/' + stim
            else:
              image_path = f'{stimuli_file_path}/Scene/' + stim

            img = load_image(image_path)
            # img.save(f'generated_images/{37*(run_int-1)+i}_action.png')
            img = pipe_prior.image_processor(img, return_tensors="pt").pixel_values[0].unsqueeze(0).to(dtype=torch.half, device=device)

            img_embeds = pipe_prior.image_encoder(img)['image_embeds'][0]
            encodings[37 * (run_int - 1) + i] = img_embeds


        # torch.save(encodings, f"/scratch/arsh/mri_torch/Encoded_Images_Kandinsky/CSI{subject_int}_sess{session_int:02}.pt")
        group.create_dataset(f"sess{session_int:02}", data=encodings)

