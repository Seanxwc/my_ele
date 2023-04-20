import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num    # head_num =4
        self.dk = (embedding_dim // head_num) ** (1 / 2)    #(1024 // 4) ** (1 / 2) = 256 ** 0.5 = 16

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)  # Linear层
        #print("qkv.shape:",qkv.shape)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num)) #B
        # print("query.shape:",query.shape)
        # print("key.shape:",key.shape)
        # print("value.shape:",value.shape)
        # print("============================================")
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim   # patch_dim = 16  即 h和w
        self.classification = classification # False
        self.num_tokens = (img_dim // patch_dim) ** 2  # (img_dim // patch_dim) = (H//h)或(W//w) = (8 // 1) = 8 表示图像切开后每张图的尺寸大小8x8     **2是图像切开后每张图的像素数
        self.token_dim = in_channels * (patch_dim ** 2) #输入通道数 * 一个通道图的像素数 = C*(h*w) =输入所有通道的像素数

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)   # B·C·(h H)·(w W) -> B·(H W)·(h w C)  = B·((H·W)/(h·w)))·(h乘w乘C)

        batch_size, tokens, _ = img_patches.shape
        #print(img_patches.shape)
        project = self.projection(img_patches)  # Linear层
        #print(project.shape)

        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)
        #print(token.shape)

        patches = torch.cat([token, project], dim=1) #
        #print(patches.shape)
        patches += self.embedding[:tokens + 1, :]
        #print(patches.shape)

        x = self.dropout(patches)
        #print(x.shape)
        x = self.transformer(x)
        #print(x.shape)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]  # classification是False  x = x[:, 1:, :]  否则 x=self.mlp_head(x[:, 0, :])

        return x


if __name__ == '__main__':
    vit = ViT(img_dim=128,
              in_channels=3,
              patch_dim=16,
              embedding_dim=512,
              block_num=12,
              head_num=4,
              mlp_dim=1024)
    print(sum(p.numel() for p in vit.parameters()))
    print(vit(torch.rand(2, 3, 128, 128)).shape)
