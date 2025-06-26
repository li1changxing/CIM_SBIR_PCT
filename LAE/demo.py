# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Dict, List,

import torch.nn as nn
from absl import logging
from timm.utils.model_ema import ModelEmaV2
from LAE.core import App, Conf,  Module, ModuleConf
from LAE.pet import Prefix, KVLoRA, Adapter


@dataclass
class MyModuleConf(ModuleConf):
    adapt_blocks: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    pet_cls: str = "Adapter"  # enum: Adapter, LoRA, Prefix
    pet_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    num_emas: int = 1
    ema_decay: float = 0.9999
    num_freeze_epochs: int = 3
    eval_only_emas: bool = False


@dataclass
class MyConf(Conf):
    module: MyModuleConf = MyModuleConf()


class MyModule(Module):
    pets: nn.Module  # online pet modules
    pets_emas: nn.ModuleList  # offline pet modules
    original_vit: nn.Module
    loss_fn: nn.Module
    def setup_model(self):
        super().setup_model()
        if getattr(self, "pets_emas", None) is None:
            self.pets_emas = nn.ModuleList([])
            self.pets = self.create_pets()
            logging.info(f"==> pets:\n{self.pets}")
        elif len(self.pets_emas) < self.cfg.num_emas:
            ema = ModelEmaV2(self.pets, decay=self.cfg.ema_decay)
            self.pets_emas.append(ema)

        self.attach_pets(self.pets)

    def create_pets_vit(self):
        assert self.cfg.pet_cls in ["Adapter", "LoRA", "Prefix"]

        n = len(self.cfg.adapt_blocks)
        embed_dim = self.model.backbone.embed_dim

        kwargs = dict(**self.cfg.pet_kwargs)
        if self.cfg.pet_cls == "Adapter":
            kwargs["embed_dim"] = embed_dim
            return nn.ModuleList([Adapter(**kwargs) for _ in range(n)])

        if self.cfg.pet_cls == "LoRA":
            kwargs["in_features"] = embed_dim
            kwargs["out_features"] = embed_dim
            return nn.ModuleList([KVLoRA(**kwargs) for _ in range(n)])

        kwargs["dim"] = embed_dim
        return nn.ModuleList([Prefix(**kwargs) for i in range(n)])

    def create_pets(self):
        return self.create_pets_vit()
    def attach_pets_vit(self, pets: nn.ModuleList):
        assert self.cfg.pet_cls in ["Adapter", "LoRA", "Prefix"]

        if self.cfg.pet_cls == "Adapter":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attach_adapter(attn=pets[i])
            return

        if self.cfg.pet_cls == "LoRA":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attn.attach_adapter(qkv=pets[i])
            return

        for i, b in enumerate(self.cfg.adapt_blocks):
            self.model.backbone.blocks[b].attn.attach_prefix(pets[i])

    def attach_pets(self, pets: nn.ModuleList):
        return self.attach_pets_vit(pets)


    def forward(self, input):
        return super().forward(input)

if __name__ == "__main__":
    kwargs = dict(
       module_cls=MyModule, config_cls=MyConf
    )
    App(**kwargs).run()
