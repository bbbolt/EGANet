import math

from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import numbers

from einops import rearrange
from basicsr.archs.dyconv import *

# ---------------------------------------------------------------------------------------------------------------------
# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def blc_to_bchw(x: torch.Tensor, x_size) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):    # for better performance and less params we set bias=False
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
# ---------------------------------------------------------------------------------------------------------------------
class GDFN(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(GDFN, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# ---------------------------------------------------------------------------------------------------------------------
class MDKG(nn.Module):

    def __init__(self, dim, window_size, heads=None, bias=True, input_resolution=None):
        super().__init__()

        # for flops counting
        self.input_resolution = input_resolution

        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.heads = heads

        # pw-linear
        # in pw-linear layer we inherit settings from DWBlock. Set bias=False
        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)

        self.conv = DyConv(dim, kernel_size=window_size, group_channels=heads, bias=bias)

    def forward(self, x):
        # shortcut outside the block
        x = self.conv0(x)
        x = self.conv(x)
        x = self.conv1(x)
        return x


class PS(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.act = nn.GELU()

        self.conv0 = nn.Conv2d(self.dim, self.dim, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding="same", groups=dim, bias=True)
        self.conv2_1 = nn.Conv2d(dim // 3, dim // 3, (3, 3), padding="same",
                                 groups=dim // 3)
        self.conv2_2 = nn.Conv2d(dim // 3, dim // 3, (5, 3), padding="same",
                                 groups=dim // 3)
        self.conv2_3 = nn.Conv2d(dim // 3, dim // 3, (3, 5), padding="same",
                                 groups=dim // 3)

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))


    def forward(self, x):

        self.h, self.w = x.shape[2:]

        x1 = torch.chunk(x, 2, dim=1)
        x2 = torch.chunk(x1[1], 3, dim=1)

        x_v = torch.var(x1[0], dim=(-2,-1), keepdim=True)
        x_s = self.max_pool(self.dwconv(x1[0]))
        hfe = self.act(self.conv0(x_s * self.alpha + x_v * self.belt))+x1[0]

        s1 = self.act(self.conv2_1(x2[0]))
        s2 = self.act(self.conv2_2(x2[1] + s1))
        s3 = self.act(self.conv2_3(x2[2] + s2))
        lfe = torch.cat([s1, s2, s3], dim=1) + x1[1]

        return hfe, lfe

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, tlc_flag=True, tlc_kernel=48, input_resolution=None):
        super(Attention, self).__init__()
        self.tlc_flag = tlc_flag    # TLC flag for validation and test

        self.dim = dim
        self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=True)
        self.q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.PS_blocks = PS(dim)

        self.act = nn.Softmax()

        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)
        q_ = self.q(q)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1_, q2_ = self.PS_blocks(q_)
        q1_ = rearrange(q1_, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2_ = rearrange(q2_, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k = torch.nn.functional.normalize(k, dim=-1)
        q = q1_ * q2_
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        out = (attn @ v)

        return out

    def forward(self, x):
        b, c, h, w = x.shape
        # q = self.q(x)
        qkv = self.qkv_dwconv(self.qkv(x))

        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qkv = self.grids(qkv)  # convert to local windows
        out = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt
# ---------------------------------------------------------------------------------------------------------------------

class MGDC(nn.Module):
    def __init__(self, dim, kernel_size=7, num_heads=6, ffn_ratio=2.,  input_resolution=None):
        super(MGDC, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')

        # IDynamic Local Feature Calculate
        self.MDKG = MDKG(dim, window_size=kernel_size, heads=num_heads, input_resolution=input_resolution)

        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        # FeedForward Network
        self.GDFN = GDFN(dim, ffn_expansion_factor=ffn_ratio, bias=False, input_resolution=input_resolution)


    def forward(self, x):
        x = self.MDKG(self.norm1(x)) + x
        x = self.GDFN(self.norm2(x)) + x
        return x

class PSGA(nn.Module):
    def __init__(self, dim, num_heads=6, ffn_ratio=2., tlc_flag=True, tlc_kernel=48, input_resolution=None):
        super(PSGA, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.attn = Attention(dim, num_heads=num_heads, bias=False, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel, input_resolution=input_resolution)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.GDFN = GDFN(dim, ffn_expansion_factor=ffn_ratio, bias=False, input_resolution=input_resolution)

    def forward(self, x):
        x = self.attn(self.norm3(x)) + x
        x = self.GDFN(self.norm4(x)) + x
        return x
# ---------------------------------------------------------------------------------------------------------------------
# BuildBlocks
class BuildBlock(nn.Module):
    def __init__(self, dim, blocks=3,
                 kernel_size=7, num_heads=6, ffn_ratio=2.,
                 tlc_flag=True, tlc_kernel=48,input_resolution=None
                 ):
        super(BuildBlock, self).__init__()

        self.input_resolution = input_resolution

        # --------
        self.dim = dim
        self.blocks = blocks
        self.window_size = window_size
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.tlc = tlc_flag
        # ---------

        # buildblock body
        # ---------
        body = []

        for _ in range(blocks):
            body.append(MGDC(dim, kernel_size, num_heads, ffn_ratio, input_resolution=input_resolution))
            body.append(PSGA(dim, num_heads, ffn_ratio, tlc_flag, tlc_kernel, input_resolution=input_resolution))

        # --------

        body.append(nn.Conv2d(dim, dim, 3, 1, 1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x

# ---------------------------------------------------------------------------------------------------------------------

class UpsampleOneStep(nn.Sequential):

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops

class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
# ---------------------------------------------------------------------------------------------------------------------

@ARCH_REGISTRY.register()
class EGANet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 dim=48,
                 groups=3,
                 blocks=3,
                 kernel_size=7, num_heads=6, ffn_ratio=2.,
                 tlc_flag=True, tlc_kernel=48,
                 upscale=4,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 body_norm=False,
                 input_resolution=None,
                 **kwargs):
        super(EGANet, self).__init__()

        # for flops counting
        self.dim = dim
        self.input_resolution = input_resolution

        # MeanShift for Image Input
        # ---------
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        # -----------

        # Upsample setting
        # -----------
        self.upscale = upscale
        self.upsampler = upsampler
        # -----------

        # ------------------------- 1, shallow feature extraction ------------------------- #
        # the overlap_embed: remember to set it into bias=False
        self.overlap_embed = nn.Sequential(OverlapPatchEmbed(in_chans, dim, bias=False))

        # ------------------------- 2, deep feature extraction ------------------------- #
        m_body = []

        # Base on the Transformer, When we use pre-norm we need to build a norm after the body block
        if body_norm:       # Base on the SwinIR model, there are LayerNorm Layers in PatchEmbed Layer between body
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        for i in range(groups):
            m_body.append(BuildBlock(dim, blocks,
                 kernel_size, num_heads, ffn_ratio,
                 tlc_flag, tlc_kernel, input_resolution=input_resolution))

        if body_norm:
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        m_body.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1)))

        self.deep_feature_extraction = nn.Sequential(*m_body)

        # ------------------------- 3, high quality image reconstruction ------------------------- #

        # setting for pixelshuffle for big model, but we only use pixelshuffledirect for all our model
        # -------
        num_feat = 64
        embed_dim = dim
        num_out_ch = in_chans
        # -------

        if self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, input_resolution=self.input_resolution)

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        pass    # all are in forward function including deep feature extraction

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.upsample(x)

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        else:
            # for image denoising and JPEG compression artifact reduction
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_last(x)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.input_resolution

        # overlap_embed layer
        flops += h * w * 3 * self.dim * 9

        # BuildBlock:
        for i in range(len(self.deep_feature_extraction) - 1):
            flops += self.deep_feature_extraction[i].flops()

        # conv after body
        flops += h * w * 3 * self.dim * self.dim
        flops += self.upsample.flops()

        return flops