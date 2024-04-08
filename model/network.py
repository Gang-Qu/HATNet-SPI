import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
import torch.nn.init as init


def EuclideanProj(X,Y,H,W,HT,WT,mu):
    b, c, _, _ = X.shape
    Delta_Y = Y - torch.matmul(torch.matmul(H.repeat((b,c,1,1)),X),WT.repeat((b,c,1,1)))
    Delta_X = torch.matmul(torch.matmul(HT.repeat((b,c,1,1)),Delta_Y),W.repeat((b,c,1,1)))
    Z = X + torch.div(Delta_X, mu+1)
    return Z

class FFN(nn.Module):
    def __init__(self, dim, exp_ratio=2, bias=True):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, int(dim * exp_ratio), kernel_size=1, bias=bias)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(int(dim * exp_ratio), int(dim * exp_ratio), kernel_size=3, stride=1, padding=1,
                                groups=int(dim * exp_ratio), bias=bias)
        self.proj_out = nn.Conv2d(int(dim * exp_ratio), dim, kernel_size=1, bias=bias)

    def forward(self, x, H, W):
        x = einops.rearrange(x, 'b (h w) c-> b c h w', h=H, w=W).contiguous()
        x = self.act(self.proj_in(x))
        x = self.proj_out(self.dwconv(x))
        x = einops.rearrange(x, 'b c h w-> b (h w) c').contiguous()
        return x

class Window_Attention(nn.Module):
    def __init__(self, dim, idx, split_size=[8, 8], pool_kernel=1, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.pool_kernel = pool_kernel
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.attn_drop = nn.Dropout(attn_drop)

    def vid2win(self, x, u, v):
        x = einops.rearrange(x, 'b (d c) (u h) (v w)-> (b u v) d (h w) c', d=self.num_heads, u=u, v=v).contiguous()
        return x

    def forward(self, kv, q, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        k, v = kv[0], kv[1]
        B, C, H, W = k.shape
        num_H = H // self.H_sp
        num_W = W // self.W_sp

        # partition the q,k,v, image to window
        _, _, H_q, W_q = q.shape
        assert H * self.pool_kernel == H_q and W * self.pool_kernel == W_q, 'Query size does not match.'
        q = self.vid2win(q, num_H, num_W)
        k = self.vid2win(k, num_H, num_W)
        v = self.vid2win(v, num_H, num_W)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B head N C @ B head C N --> B head N N
        # use mask for shift window
        if mask is not None:
            attn = einops.rearrange(attn, '(b u v) d h w-> b (u v) d h w', b=B, u=num_H, v=num_W).contiguous()
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = einops.rearrange(attn, 'b n d h w-> (b n) d h w').contiguous()
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = einops.rearrange(x, '(b u v) d (h w) c-> b (d c) (u h) (v w)', b=B, u=num_H, v=num_W,
                             h=self.H_sp * self.pool_kernel, w=self.W_sp * self.pool_kernel).contiguous()
        return x


class Spatial_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    def __init__(self, inp_dim, out_dim, num_heads,
                 reso=64, split_size=[4, 8], pool=nn.AvgPool2d, pool_kernel=1, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., shift_flag=False):
        super().__init__()
        self.dim = out_dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = [split_size[0] // 2, split_size[1] // 2]
        self.q_split_size = [int(split_size[0] * pool_kernel), int(split_size[1] * pool_kernel)]
        self.q_shift_size = [self.q_split_size[0] // 2, self.q_split_size[1] // 2]
        self.shift_flag = shift_flag
        self.patches_resolution = reso
        self.pool_kernel = pool_kernel
        if self.pool_kernel > 1:
            self.pool = pool(kernel_size=self.pool_kernel, stride=self.pool_kernel)
        self.q = nn.Linear(inp_dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(inp_dim, self.dim * 2, bias=qkv_bias)
        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(drop)
        self.attns = nn.ModuleList([
            Window_Attention(
                self.dim // 2, idx=i,
                split_size=split_size, pool_kernel=pool_kernel, num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            for i in range(self.branch_num)])

        if self.shift_flag:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

    def calculate_mask(self, H, W):
        img_mask_0 = torch.zeros((1, H, W))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.q_split_size[0]),
                      slice(-self.q_split_size[0], -self.q_shift_size[0]),
                      slice(-self.q_shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                      slice(-self.q_split_size[1], -self.q_shift_size[1]),
                      slice(-self.q_shift_size[1], None))

        h_slices_1 = (slice(0, -self.q_split_size[1]),
                      slice(-self.q_split_size[1], -self.q_shift_size[1]),
                      slice(-self.q_shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                      slice(-self.q_split_size[0], -self.q_shift_size[0]),
                      slice(-self.q_shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w] = cnt
                cnt += 1
        alpha = (1 - 1 / self.pool_kernel) / 2
        beta = (1 + 1 / self.pool_kernel) / 2
        h_slice_0 = slice(int(self.q_split_size[0] * alpha), int(self.q_split_size[0] * beta))
        w_slice_0 = slice(int(self.q_split_size[1] * alpha), int(self.q_split_size[1] * beta))
        h_slice_1 = slice(int(self.q_split_size[1] * alpha), int(self.q_split_size[1] * beta))
        w_slice_1 = slice(int(self.q_split_size[0] * alpha), int(self.q_split_size[0] * beta))

        # calculate mask for window-0
        img_mask_0 = einops.rearrange(img_mask_0, 'b (u h) (v w)-> (b u v) h w', b=1, u=H // self.q_split_size[0],
                                      v=W // self.q_split_size[1]).contiguous()
        img_mask_00 = img_mask_0[:, h_slice_0, w_slice_0]
        mask_windows_0 = einops.rearrange(img_mask_0, 'b h w-> b (h w)').contiguous()
        mask_windows_00 = einops.rearrange(img_mask_00, 'b h w-> b (h w)').contiguous()  # num_Wins, sw[0], sw[1], 1
        attn_mask_0 = mask_windows_0.unsqueeze(2) - mask_windows_00.unsqueeze(1)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for window-1
        img_mask_1 = einops.rearrange(img_mask_1, 'b (u h) (v w)-> (b u v) h w', b=1, u=H // self.q_split_size[1],
                                      v=W // self.q_split_size[0]).contiguous()
        img_mask_11 = img_mask_1[:, h_slice_1, w_slice_1]
        mask_windows_1 = einops.rearrange(img_mask_1, 'b h w-> b (h w)').contiguous()
        mask_windows_11 = einops.rearrange(img_mask_11, 'b h w-> b (h w)').contiguous()
        attn_mask_1 = mask_windows_1.unsqueeze(2) - mask_windows_11.unsqueeze(1)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))
        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, L, _ = x.shape
        C = self.dim
        assert L == H * W, "flatten img_tokens has wrong size"

        if self.pool_kernel > 1:
            H_q = H
            W_q = W
            q = self.q(x)
            x = einops.rearrange(x, 'b (h w) c-> b c h w', h=H, w=W).contiguous()
            x = self.pool(x)
            _, _, H, W = x.shape
            x = einops.rearrange(x, 'b c h w-> b (h w) c', b=B).contiguous()
        else:
            H_q = H
            W_q = W
            q = self.q(x)
        kv = self.kv(x)
        kv = einops.rearrange(kv, 'b (h w) (n c)-> n b c h w', h=H, w=W, n=2).contiguous()

        ## image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        if H % max_split_size != 0 or W % max_split_size != 0:
            pad_l = pad_t = 0
            pad_r = (max_split_size - W % max_split_size) % max_split_size
            pad_b = (max_split_size - H % max_split_size) % max_split_size
            kv = F.pad(kv, (pad_l, pad_r, pad_t, pad_b))
            _H = pad_b + H
            _W = pad_r + W
            _L = _H * _W
        else:
            _H = H
            _W = W
            _L = L

        if _W * self.pool_kernel != W_q or _W * self.pool_kernel != H_q:
            pad_l = pad_t = 0
            pad_r = _W * self.pool_kernel - W_q
            pad_b = _H * self.pool_kernel - H_q
            q = einops.rearrange(q, 'b (h w) c-> b c h w', h=H_q, w=W_q).contiguous()
            q = F.pad(q, (pad_l, pad_r, pad_t, pad_b))
            _H_q = pad_b + H_q
            _W_q = pad_r + W_q
        else:
            q = einops.rearrange(q, 'b (h w) c-> b c h w', h=H_q, w=W_q).contiguous()
            _H_q = H_q
            _W_q = W_q

        # window-0 and window-1 on split channels [C/2, C/2]; for square windows (e.g., 8x8), window-0 and window-1 can be merged
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...
        if self.shift_flag:
            kv_0 = torch.roll(kv[:, :, :C // 2, :, :], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(-2, -1))
            kv_1 = torch.roll(kv[:, :, C // 2:, :, :], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(-2, -1))
            q_0 = torch.roll(q[:, :C // 2, :, :], shifts=(-self.q_shift_size[0], -self.q_shift_size[1]), dims=(-2, -1))
            q_1 = torch.roll(q[:, C // 2:, :, :], shifts=(-self.q_shift_size[1], -self.q_shift_size[0]), dims=(-2, -1))

            if self.patches_resolution != (_W * self.pool_kernel) or self.patches_resolution != (_W * self.pool_kernel):
                mask_tmp = self.calculate_mask(_H * self.pool_kernel, _W * self.pool_kernel)
                x1_shift = self.attns[0](kv_0, q_0, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](kv_1, q_1, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](kv_0, q_0, mask=self.attn_mask_0)
                x2_shift = self.attns[1](kv_1, q_1, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.q_shift_size[0], self.q_shift_size[1]), dims=(-2, -1))  # (b c t h w)
            x2 = torch.roll(x2_shift, shifts=(self.q_shift_size[1], self.q_shift_size[0]), dims=(-2, -1))
            x1 = x1[:, :, :H_q, :W_q]
            x2 = x2[:, :, :H_q, :W_q]
            # attention output
            attened_x = torch.cat([x1, x2], dim=1)

        else:
            x1 = self.attns[0](kv[:, :, :C // 2, :, :], q[:, :C // 2, :, :])[:, :, :H_q, :W_q]
            x2 = self.attns[1](kv[:, :, C // 2:, :, :], q[:, C // 2:, :, :])[:, :, :H_q, :W_q]
            # attention output
            attened_x = torch.cat([x1, x2], dim=1)
        x = attened_x.reshape(B, C, -1).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DSMHSA(nn.Module):
    def __init__(self, dim, dim_heads, dim_groups=[32, 96],
                 reso=256, split_size=[4, 16], pool=nn.AvgPool2d, pool_kernel=[1, 2],
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., shift_flag=False):
        super().__init__()
        self.dim = dim
        self.dim_lr = dim_groups[1]
        self.dim_hr = dim_groups[0]
        self.attn_hr = Spatial_Attention(
            inp_dim=dim, out_dim=self.dim_hr, num_heads=self.dim_hr // dim_heads, reso=reso, split_size=split_size,
            pool_kernel=pool_kernel[0],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, shift_flag=shift_flag
        )
        self.attn_lr = Spatial_Attention(
            inp_dim=dim, out_dim=self.dim_lr, num_heads=self.dim_lr // dim_heads, reso=reso, split_size=split_size,
            pool_kernel=pool_kernel[1],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, shift_flag=shift_flag
        )
        self.proj = nn.Conv2d(128, dim, 3, 1, 1)

    def forward(self, x, H, W):
        '''
        x: b (h w) c
        '''
        B, L, C = x.shape
        assert L == H * W, 'input size error.'
        y1 = self.attn_hr(x, H, W)
        y2 = self.attn_lr(x, H, W)
        y = torch.cat((y1, y2), dim=-1)
        y = einops.rearrange(y, 'b (h w) c-> b c h w', h=W, w=W).contiguous()
        y = self.proj(y)
        y = einops.rearrange(y, 'b c h w-> b (h w) c').contiguous()
        return y

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.attention(x)
        return x * y

class ConvWithCA(nn.Module):
    def __init__(self, dim, compress_ratio=4, reduction=16):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(dim, dim // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim // compress_ratio, dim, 3, 1, 1),
            ChannelAttention(dim, reduction),
        )
    def forward(self, x, H, W):
        '''
        x: B (H W) C
        '''
        x = einops.rearrange(x, 'b (h w) c-> b c h w', h=H, w=W).contiguous()
        x = self.cab(x)
        x = einops.rearrange(x, 'b c h w-> b (h w) c').contiguous()
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, dim_heads, dim_groups=[32, 96], reso=128, split_size=[4, 8], pool=nn.AvgPool2d,
                 pool_kernel=[1, 2],
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., exp_ratio=2., shift_flag=False):
        super().__init__()
        self.daul_atten = DSMHSA(dim=dim, dim_heads=dim_heads, dim_groups=dim_groups, reso=reso, split_size=split_size,
                                 pool=pool, pool_kernel=pool_kernel, qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, shift_flag=shift_flag)
        self.conv_atten = ConvWithCA(dim=dim)
        self.ffn = FFN(dim=dim, exp_ratio=exp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        '''
        x: (b c h w)
        '''
        _, _, H, W = x.shape
        x = einops.rearrange(x, 'b c h w-> b (h w) c').contiguous()
        x = x + self.daul_atten(self.norm1(x), H, W) + self.conv_atten(x, H, W)  ####
        x = x + self.ffn(self.norm2(x), H, W)
        x = einops.rearrange(x, 'b (h w) c-> b c h w', h=H, w=W).contiguous()
        return x

class ResidualGroup(nn.Module):
    def __init__(self, dim=128, dim_heads=16, dim_groups=[32, 96], reso=256, split_size=[4, 16], qkv_bias=True,
                 exp_ratio=2., blk_num=4):
        super().__init__()
        self.transformerlist = nn.Sequential(
            *[TransformerBlock(dim=dim, dim_heads=dim_heads, dim_groups=dim_groups, reso=reso, split_size=split_size,
                               qkv_bias=qkv_bias, exp_ratio=exp_ratio, shift_flag=bool(k % 2)) for k in range(blk_num)]
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )
    def forward(self, x):
        x = x + self.conv(self.transformerlist(x))
        return x

class BasicBlock(nn.Module):
    def __init__(self, color_channels=1, width=64, middle_blk_num=6, enc_blk_nums=[1, 2, 4], dec_blk_nums=[1, 1, 1],
                 first_stage=False):
        super(BasicBlock, self).__init__()
        self.first_stage = first_stage
        self.embedding = nn.Conv2d(in_channels=color_channels, out_channels=width, kernel_size=3, padding=1, stride=1,
                                   groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        if not first_stage:
            self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mapping = nn.Conv2d(in_channels=width, out_channels=color_channels, kernel_size=3, padding=1, stride=1,
                                 groups=1, bias=True)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            if not first_stage:
                self.convs1.append(
                    nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
                )
            self.encoders.append(
                nn.Sequential(
                    *[ResidualGroup(dim=chan, dim_heads=16, dim_groups=[32, 96],
                                    reso=256 // 2, split_size=[4, 16], qkv_bias=True,
                                    exp_ratio=2, blk_num=2) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[ResidualGroup(dim=chan, dim_heads=16, dim_groups=[32, 96],
                            reso=256 // 2, split_size=[4, 16], qkv_bias=True,
                            exp_ratio=2, blk_num=2) for _ in range(middle_blk_num)]
        )
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.convs2.append(
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
            )

            self.decoders.append(
                nn.Sequential(
                    *[ResidualGroup(dim=chan, dim_heads=16, dim_groups=[32, 96],
                                    reso=256 // 2, split_size=[4, 16], qkv_bias=True,
                                    exp_ratio=2, blk_num=2) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, last_decs=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.embedding(inp)
        encs = []
        decs = []
        if not self.first_stage:
            for encoder, down, conv1, last_dec in zip(self.encoders, self.downs, self.convs1, last_decs[::-1]):
                x = conv1(torch.cat([x, last_dec], dim=1))
                x = encoder(x)
                encs.append(x)
                x = down(x)
        else:
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                encs.append(x)
                x = down(x)
        x = self.middle_blks(x)
        for decoder, up, conv2, enc_skip in zip(self.decoders, self.ups, self.convs2, encs[::-1]):
            x = up(x)
            x = conv2(torch.cat([x, enc_skip], dim=1))
            x = decoder(x)
            decs.append(x)
        x = self.mapping(x)
        x = x + inp
        return x[:, :, :H, :W], decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class HATNet(nn.Module):
    def __init__(self, imag_size, meas_size, img_channels=1, channels=64, mid_blocks=6, enc_blocks=[1, 2, 4],
                 dec_blocks=[1, 1, 1], stages=7, matrix_train=True, only_test=False):
        super().__init__()
        self.stages = stages
        self.imag_size = imag_size
        self.only_test = only_test
        ## Mask Initialization ##
        self.H = nn.Parameter(init.xavier_normal_(torch.Tensor(meas_size[0], imag_size[0])))
        self.W = nn.Parameter(init.xavier_normal_(torch.Tensor(meas_size[1], imag_size[1])))
        self.mu = nn.Parameter(torch.Tensor([0.001]).repeat(stages))
        self.denoisers = nn.ModuleList([])
        self.denoisers.append(BasicBlock(color_channels=img_channels, width=channels, middle_blk_num=mid_blocks,
                                         enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks, first_stage=True))
        for i in range(stages - 1):
            self.denoisers.append(BasicBlock(color_channels=img_channels, width=channels, middle_blk_num=mid_blocks,
                                             enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks))

    def forward(self, X):
        """
        :input X: [b,256,256]
        """
        X = X.unsqueeze(1)
        b, c, h, w = X.shape
        H = self.H
        W = self.W
        HT = torch.transpose(H, 0, 1).contiguous()
        WT = torch.transpose(W, 0, 1).contiguous()
        Y = torch.matmul(torch.matmul(H.repeat((b, c, 1, 1)), X), WT.repeat((b, c, 1, 1)))
        X = torch.matmul(torch.matmul(HT.repeat((b, c, 1, 1)), Y), W.repeat((b, c, 1, 1)))
        for i in range(self.stages):
            mu = self.mu[i]
            Z = EuclideanProj(X, Y, H, W, HT, WT, mu)
            if self.only_test and Z.shape[0] > 1:
                Z = einops.rearrange(Z, '(a b) 1 h w-> 1 1 (a h) (b w)', a=2, b=2)
            if i == 0:
                X, features = self.denoisers[i](Z)
            else:
                X, features = self.denoisers[i](Z, features)
            if self.only_test and X.shape[2] > h:
                X = einops.rearrange(X, '1 1 (a h) (b w)-> (a b) 1 h w', a=2, b=2)
        X = X.squeeze(1)
        return X, self.H, self.W, HT, WT
