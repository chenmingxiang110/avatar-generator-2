import os
import cv2
import numpy as np
import torch
from lib.stylegan.model import StyledGenerator

def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

def sample(generator, step, mean_style, n_sample, device):
    code = torch.randn(n_sample, 512).to(device)
    image = generator(
        code,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return code.detach().cpu().numpy(), image

def style_mixing(generator, step, mean_style, n_source, n_target, device):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images

class FaceGen:
    
    def __init__(self):
        self.device = torch.device("cuda:0")
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "train_step-10-8000.model"
        )
        generator = StyledGenerator(512).to(self.device)
        generator.load_state_dict(torch.load(model_path)['g_running'])
        self.generator = generator.eval()
    
    def generate(self):
        with torch.no_grad():
            mean_style = get_mean_style(self.generator, self.device)
            step = 6

            code, img = sample(
                self.generator, step, mean_style, 1, self.device
            )
            img = img.detach().cpu().numpy().transpose([0,2,3,1])
            img = np.array(np.clip((img+1)/2, 0, 1)[0] * 255, np.uint8)
        return img, code
