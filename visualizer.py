import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

def img2np(img):
    img = img.numpy().transpose((1, 2, 0))
    img = np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    return img

def show(model, test_dataset, device):
    with torch.no_grad():
        data = next(iter(test_dataset))[0].to(device)

        img = data.cpu()
        transf_img = model.stn(data).cpu()

        fig, ax = plt.subplots(1, 2)

        ax[0].set_title('Original')
        ax[1].set_title('Transformed')

        ax[0].imshow( img2np(torchvision.utils.make_grid(img)) )
        ax[1].imshow( img2np(torchvision.utils.make_grid(transf_img)) )

    plt.show()