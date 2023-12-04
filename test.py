

from DataLoaderH5 import DatasetH5
from model import BrainModel
from diffusion import ImageGenerator
import torch
from PIL import Image

import torch.optim as optim
from tqdm import tqdm

device = 'cuda:3'

def combine_images(image_list1, image_list2, output_path):
    # Check if the number of images in both lists is the same
    if len(image_list1) != len(image_list2):
        raise ValueError("Number of images in both lists must be the same")

    # Assuming all images have the same size, get the size of one image
    width, height = image_list1[0].size

    # Set the number of columns and rows based on the number of images
    num_columns = len(image_list1)
    num_rows = 2  # Two lists of images

    # Create a new image with the combined size
    new_image = Image.new("RGB", (width * num_columns, height * num_rows))

    # Paste images from the first list into the first row
    for i, img in enumerate(image_list1):
        new_image.paste(img, (i * width, 0))

    # Paste images from the second list into the second row
    for i, img in enumerate(image_list2):
        new_image.paste(img, (i * width, height))

    # Save or display the resulting image
    new_image.save(output_path)


with torch.no_grad():
    mri_model = BrainModel()
    mri_model.load_state_dict(torch.load('basic_model10.pt'))
    mri_model.eval()
    mri_model.to(device)

    test_data = DatasetH5('/scratch/arsh/mri_h5/scans/mri_data.hdf5', '/scratch/arsh/mri_h5/stimuli/stimuli_data.hdf5',
                          1, 'D', [14, 15], device=device)
    test_generator = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    image_gen = ImageGenerator(device)
    X, y = next(iter(test_generator))
    output = mri_model.forward(X)

    print(output.shape)

    print(f"Loss: {((output-y)**2).mean()}")

    print(output.abs().mean(), output.max(), output.min())
    print((output[0]-output[1]).sum())


    true_images = image_gen.generate(y[10:15]).images

    print(true_images)

    true_images[0].save('generated_images/img5_halftrue.png')

    pred_images = image_gen.generate(output[10:15]).images

    print(pred_images)

    pred_images[0].save('generated_images/img5_pred.png')

    print(pred_images)





combine_images(true_images, pred_images, 'generated_images/img100.png')






