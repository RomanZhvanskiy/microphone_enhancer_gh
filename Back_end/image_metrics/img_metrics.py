from skimage.metrics import structural_similarity as compare_similarity
import numpy as np

def image_res(im1, im2, meter):
    print(f'calculating similarity between the two images based on {meter}')
    if meter == "SSIM":
        ssim_score, _ = compare_similarity(im1, im2, full=True)
        print("SSIM score : ", ssim_score)
    if meter == "PSNR":
        im1 = im1.astype(np.float64)
        im2 = im2.astype(np.float64)
        mse = np.mean((im1 - im2)**2)
        max_pixel = 255.0
        psnr = 20 * np.log(max_pixel / np.sqrt(mse))
        print(f"PSNR score : ", psnr)
    else:
        print('ERROR: please select SSIM or PSNR as metrics')
