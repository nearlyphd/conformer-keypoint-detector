import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import VisionTransformer, _cfg

from vision_transformer import VisionTransformer, _cfg
from conformer import Conformer, ConformerKeypointDetector
from timm.models.registry import register_model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_med_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def Conformer_tiny_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch32(pretrained=False, **kwargs):
    model = Conformer(patch_size=32, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_base_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_tiny_patch16_keypoint(pretrained=False, num_keypoints=3, **kwargs):
    conformer_kwargs = dict(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                            num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model = ConformerKeypointDetector(num_keypoints=num_keypoints, conformer_kwargs=conformer_kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained weights not implemented for keypoint wrapper.")
    return model

@register_model
def Conformer_small_patch16_keypoint(pretrained=False, num_keypoints=3, **kwargs):
    conformer_kwargs = dict(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                            num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model = ConformerKeypointDetector(num_keypoints=num_keypoints, conformer_kwargs=conformer_kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained weights not implemented for keypoint wrapper.")
    return model

@register_model
def Conformer_small_patch32_keypoint(pretrained=False, num_keypoints=3, **kwargs):
    conformer_kwargs = dict(patch_size=32, channel_ratio=4, embed_dim=384, depth=12,
                            num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model = ConformerKeypointDetector(num_keypoints=num_keypoints, conformer_kwargs=conformer_kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained weights not implemented for keypoint wrapper.")
    return model

@register_model
def Conformer_base_patch16_keypoint(pretrained=False, num_keypoints=3, **kwargs):
    conformer_kwargs = dict(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                            num_heads=9, mlp_ratio=4, qkv_bias=True, **kwargs)
    model = ConformerKeypointDetector(num_keypoints=num_keypoints, conformer_kwargs=conformer_kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained weights not implemented for keypoint wrapper.")
    return model
