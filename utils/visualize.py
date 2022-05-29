from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch


def show_batch_imgs(imgs_batch:torch.Tensor, nrow=8, figsize=(10, 10)):
    imgs = make_grid(imgs_batch, nrow=nrow)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
