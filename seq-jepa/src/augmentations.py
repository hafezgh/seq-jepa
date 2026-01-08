import random
import torch
import torchvision
import torchvision.transforms.functional as F

### Based on https://github.com/ADevillers/EquiMod/blob/SimCLR%2BEquiMod/src/transforms.py

class ParamRandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def __init__(self, pflip=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pflip = pflip
        self.nb_params = 4

    def get_params(self, img):
        i, j, h, w = super().get_params(img, self.scale, self.ratio)
        flip = int(random.random() < self.pflip)

        return [i, j, h, w, flip]

    def apply(self, img, params):
        i, j, h, w, flip = params

        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        
        if flip:
            img = F.hflip(img)

        params = torch.FloatTensor([i, j, h, w, flip])

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)



class ParamColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, pjitter=0.8, pgray=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pjitter = pjitter
        self.pgray = pgray
        self.nb_params = 4

    def get_params(self, img):
        jitter = int(random.random() < self.pjitter)

        if jitter:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                super().get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                [[0, 1, 2, 3], 1., 1., 1., 0.]
        
        gray = int(random.random() < self.pgray)

        return [jitter, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gray]

    def apply(self, img, params):
        jitter, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gray = params

        if jitter:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
        
        if gray:
            img = F.rgb_to_grayscale(img, num_output_channels=3)

        params = torch.FloatTensor([jitter, *fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gray])

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)



class ParamGaussianBlur(torchvision.transforms.GaussianBlur):
    def __init__(self, pblur=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pblur = pblur
        self.nb_params = 1
    
    def get_params(self, img):
        blur = int(random.random() < self.pblur)

        if blur:
            sigma = super().get_params(self.sigma[0], self.sigma[1])
        else:
            sigma = 0.

        return [blur, sigma]

    def apply(self, img, params):
        blur, sigma = params

        if blur:
            img = F.gaussian_blur(img, self.kernel_size, [sigma, sigma])

        params = torch.FloatTensor([blur, sigma])

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)



class ParamSolarize(torchvision.transforms.RandomSolarize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_params = 0
    
    def get_params(self, img):
        solarize = int(random.random() < self.p)

        return [solarize]

    def apply(self, img, params):
        solarize, = params

        if solarize:
            img = F.solarize(img, self.threshold)

        params = torch.FloatTensor([solarize])

        return img, params

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)



class ParamCompose(torch.nn.Module):
    def __init__(self, param_transforms, nonparam_transforms):
        super().__init__()
        self.param_transforms = param_transforms
        self.nonparam_transforms = nonparam_transforms
        self.nb_params = sum([transform.nb_params for transform in self.param_transforms])
        #print("nb_params for the transforms: ", self.nb_params)

    def get_params(self, img):
        return [transform.get_params(img) for transform in self.param_transforms]

    def apply(self, img, params):
        res_params = []

        for transform, transform_params in zip(self.param_transforms, params):
            img, sub_params = transform.apply(img, transform_params)
            if isinstance(transform, ParamRandomResizedCrop):
                cond_params = sub_params[:4]
                res_params.append(cond_params)
            elif isinstance(transform, ParamGaussianBlur):
                cond_params = sub_params[-1].unsqueeze(0)
                res_params.append(cond_params)
            elif isinstance(transform, ParamColorJitter):
                cond_params = sub_params[-5:-1]
                res_params.append(cond_params)
            else:
                pass
            
        for transform in self.nonparam_transforms:
            img = transform(img)

        return img, torch.cat(res_params)

    def forward(self, img):
        params = self.get_params(img)
        return self.apply(img, params)


