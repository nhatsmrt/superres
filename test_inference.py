from superres.metrics import PSNR
from superres.models import DeepLaplacianPyramidNetV2
from superres.inference import SuperResolutionizer

from generative_models.metrics import SSIM

import torch

upscale_factor = 4.0
batch_size = 64


from PIL import Image
from nntoolbox.vision.utils import pil_to_tensor, tensor_to_pil
from torch.nn.functional import interpolate

image = Image.open("data/cat.jpg")

high_res = pil_to_tensor(image)
# high_res = interpolate(high_res, (high_res.shape[2] // 4 * 4, high_res.shape[3] // 4 * 4))
high_res = torch.clamp(interpolate(high_res, 512, mode='bicubic'), 0.0, 1.0)
high_res_im = tensor_to_pil(high_res)
high_res_im.show()
high_res_im.save("demo/high_res.jpg")

low_res = torch.clamp(interpolate(high_res, scale_factor=1.0 / upscale_factor, mode='bicubic'), 0.0, 1.0)
low_res_interpolated = torch.clamp(interpolate(low_res, scale_factor=upscale_factor, mode='bicubic'), 0.0, 1.0)
poor_quality = tensor_to_pil(low_res_interpolated)
poor_quality.show()
poor_quality.save("demo/poor_quality.jpg")

model = DeepLaplacianPyramidNetV2(max_scale_factor=upscale_factor)
model.load_state_dict(torch.load('weights/model (13).pt', map_location=lambda storage, location: storage))
superres = SuperResolutionizer(model, False)

with torch.no_grad():
    generated_tensor = superres.upscale_tensor(low_res, upscale_factor)
    generated = tensor_to_pil(
        generated_tensor
    )
    generated.show()
    generated.save("demo/generated.jpg")


psnr_metric = PSNR()
ssim_metric = SSIM()

print(psnr_metric({"outputs": low_res_interpolated, "labels": high_res}))
print(psnr_metric({"outputs": generated_tensor, "labels": high_res}))


print(ssim_metric({"outputs": low_res_interpolated, "labels": high_res}))
print(ssim_metric({"outputs": generated_tensor, "labels": high_res}))
