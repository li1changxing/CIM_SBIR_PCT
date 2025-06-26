from functools import partial

from timm.layers import to_2tuple
from timm.models.helpers import (
    named_apply,
)
from timm.models.layers import (
    DropPath,
    PatchEmbed,
    lecun_normal_,
    trunc_normal_,
)

import torch
import torch.nn as nn

from LAE.pet import Adapter
from LAE.pet_mixin import AdapterMixin, PromptMixin, PrefixMixin

NAME_SEP = "/"


class Mlp(nn.Module, AdapterMixin):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.adapt_module("fc1", x)  # x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.adapt_module("fc2", x)  # x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module, PromptMixin, PrefixMixin, AdapterMixin):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.add_prompt(x)

        B, N, C = x.shape
        qkv = self.adapt_module("qkv", x)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.chunk(3, dim=-1)
        k, v = self.add_prefix(k, v)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        k = k.reshape(B, -1, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)

        v = v.reshape(B, -1, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.compensate_prefix(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.adapt_module("proj", x)  # x = self.proj(x)
        x = self.proj_drop(x)

        x = self.reduce_prompt(x)

        return x


class Block(nn.Module, AdapterMixin):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(
            self.adapt_module("attn", self.norm1(x))  # self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.adapt_module("mlp", self.norm2(x))  # self.mlp(self.norm2(x))
        )
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()

        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)
    def reset(self,num_classes):
        self.fc = nn.Linear(768, num_classes)
    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        if mode.startswith("jax"):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)[:,0] # shape: B N C
        return x


def _init_vit_weights(
    module: nn.Module, name: str = "", jax_impl: bool = False
):
    """ViT weight initialization
    * When called without n, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if jax_impl:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if "mlp" in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def create_pets_vit(pet_cls="Adapter"):
    assert pet_cls in ["Adapter", "LoRA", "Prefix"]

    n = len([0, 1, 2, 3, 4])

    kwargs = {"down_sample": 5,"mode": "parallel","scale": None}

    kwargs["embed_dim"] = 768
    return nn.ModuleList([Adapter(**kwargs) for _ in range(n)])


def vit_lae():
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12)

    pets = create_pets_vit()
    for i, b in enumerate([0, 1, 2, 3, 4]):
        model.blocks[b].attach_adapter(attn=pets[i])

    state_dict = torch.load("vit.pth")
    msg = model.load_state_dict(state_dict, strict=False)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model



if __name__ == '__main__':
    c = vit_lae()
    data = torch.randn((1,3,224,224))
    out = c(data)
    print(out.shape)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(c, (3, 224,224), as_strings=True, print_per_layer_stat=False)
    print("%s %s" % (flops, params))