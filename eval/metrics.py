import torch
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
# from torchmetrics.video.fvd import FrechetVideoDistance

# ------------------ 图像指标 (输入形状: (B, C, H, W), 值域 [0,1]) ------------------
def compute_fid(real_images, fake_images):
    """FID：计算两组图像集合的分布距离，值越低越好"""
    fid = FrechetInceptionDistance(feature=2048)  # 使用InceptionV3
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()

def compute_lpips(real_batch, fake_batch):
    """LPIPS：计算两批图像对之间的感知相似度，值越低越相似"""
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')  # 可选 'alex' 'vgg','squeeze'
    return lpips(real_batch, fake_batch).item()

def compute_ssim(real_batch, fake_batch):
    """SSIM：结构相似性，值越接近1越好"""
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # 若输入[0,1]则data_range=1.0
    return ssim(real_batch, fake_batch).item()

def compute_psnr(real_batch, fake_batch):
    """PSNR：峰值信噪比，值越高越好"""
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    return psnr(real_batch, fake_batch).item()

# ------------------ 视频指标 (输入形状: (B, C, T, H, W), 值域 [0,1]) ------------------
# def compute_fvd(real_videos, fake_videos):
#     """FVD：视频级分布距离，值越低越好（注意：需要较多样本）"""
#     fvd = FrechetVideoDistance(feature=2048)  # 使用I3D
#     fvd.update(real_videos, real=True)
#     fvd.update(fake_videos, real=False)
#     return fvd.compute().item()

# ============== 使用示例 ==============
if __name__ == "__main__":
    
    B, C, H, W = 8, 3, 128, 128
    real_imgs = torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8)
    fake_imgs = torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8)

    print("FID:", compute_fid(real_imgs, fake_imgs))
    # 生成随机样本（仅示意，实际应加载真实图像/视频）
    B, C, H, W = 8, 3, 128, 128
    real_imgs = torch.rand(B, C, H, W)      # [0,1]
    fake_imgs = torch.rand(B, C, H, W)      # [0,1]

    print("LPIPS:", compute_lpips(real_imgs, fake_imgs))
    print("SSIM:", compute_ssim(real_imgs, fake_imgs))
    print("PSNR:", compute_psnr(real_imgs, fake_imgs))

    # 视频示例（需要 (B, C, T, H, W)）
    T = 16
    real_videos = torch.rand(B, C, T, H, W)
    fake_videos = torch.rand(B, C, T, H, W)
    # print("FVD:", compute_fvd(real_videos, fake_videos))