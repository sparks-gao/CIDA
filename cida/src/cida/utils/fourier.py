
import torch

def low_freq_mix(x_content, x_style, alpha: float = 0.08):
    B, C, H, W = x_content.shape
    Xc = torch.fft.fft2(x_content, dim=(-2, -1))
    Xs = torch.fft.fft2(x_style, dim=(-2, -1))
    Xc = torch.fft.fftshift(Xc, dim=(-2, -1))
    Xs = torch.fft.fftshift(Xs, dim=(-2, -1))
    cy, cx = H // 2, W // 2
    r = int(min(H, W) * alpha)
    Y, X = torch.meshgrid(torch.arange(H, device=x_content.device),
                          torch.arange(W, device=x_content.device), indexing='ij')
    mask = ((Y - cy)**2 + (X - cx)**2) <= r**2
    mask = mask[None, None, :, :]
    Xmix = Xc.clone()
    Xmix[..., mask.squeeze(0).squeeze(0)] = Xs[..., mask.squeeze(0).squeeze(0)]
    Xmix = torch.fft.ifftshift(Xmix, dim=(-2, -1))
    xm = torch.fft.ifft2(Xmix, dim=(-2, -1)).real
    return xm.clamp(0, 1)
