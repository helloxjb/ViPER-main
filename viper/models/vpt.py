from timm import create_model
from functools import reduce
from operator import mul
import math
import torch
import torch.nn as nn
import time


class VPT(nn.Module):
    def __init__(self, modelname: str, num_classes: int, pretrained: bool = True,
                 prompt_tokens: int = 5, prompt_dropout: float = 0.0, prompt_type: str = 'shallow'):
        super().__init__()
        self.encoder = create_model(
            modelname, num_classes=num_classes, pretrained=pretrained)
        # Freeze parameters
        for n, p in self.encoder.named_parameters():
            if 'head' not in n:
                p.requires_grad = False

        # Prompt tuning setup
        self.prompt_tokens = prompt_tokens
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.prompt_dim = self.encoder.embed_dim
        self.prompt_type = prompt_type
        assert self.prompt_type in ['shallow', 'deep']

        # Initialize prompt embeddings
        val = math.sqrt(
            6. / float(3 * reduce(mul, self.encoder.patch_embed.patch_size, 1) + self.prompt_dim))
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, self.prompt_tokens, self.prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self.prompt_type == 'deep':
            self.total_d_layer = len(self.encoder.blocks)
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(self.total_d_layer-1, self.prompt_tokens, self.prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def incorporate_prompt(self, x, prompt_embeddings, n_prompt: int = 0):
        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(prompt_embeddings.expand(B, -1, -1)),
            x[:, (1+n_prompt):, :]
        ), dim=1)
        return x

    def forward_features(self, x, shuffle=False):
        # Get patch embeddings (including class token)
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        # Apply position embedding (with optional shuffle)
        pos_embed = self.encoder.pos_embed
        # print(pos_embed.shape)
        if shuffle:
            shuffle_ratio = 1  # 设置打乱比例（0~1），例如打乱50%
            cls_pos = pos_embed[:, :1, :]    # class token位置 [1, 1, D]
            patch_pos = pos_embed[:, 1:, :]  # patch位置 [1, num_patches, D]
            num_patches = patch_pos.size(1)
            
            # 计算实际需要打乱的数量
            num_to_shuffle = int(num_patches * shuffle_ratio)
            
            if num_to_shuffle > 0:
                # 随机选择部分索引打乱
                idx = torch.randperm(num_patches, device=x.device)      # 生成随机排列
                shuffled_idx = idx[:num_to_shuffle]                     # 需要打乱的部分
                remain_idx = idx[num_to_shuffle:]                       # 保持原序的部分
                
                # 分离需要打乱和保持原序的位置编码
                shuffle_patches = patch_pos[:, shuffled_idx, :]          # 待打乱区块
                remain_patches = patch_pos[:, remain_idx, :]            # 保持原序区块
                
                # 只打乱选中的部分
                shuffle_idx = torch.randperm(num_to_shuffle, device=x.device)  # 二次排列
                shuffled_patches = shuffle_patches[:, shuffle_idx, :]    # 执行打乱
                
                # 重新组合 (先放打乱部分，再放剩余部分)
                patch_pos = torch.cat([shuffled_patches, remain_patches], dim=1)
            
            # 重新组合class token和patch
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
            torch.cuda.synchronize()  # 确保GPU操作完成

        x = x + pos_embed

        # Apply position dropout and norm
        if hasattr(self.encoder, 'pos_drop'):
            x = self.encoder.pos_drop(x)
        x = self.encoder.norm_pre(x)

        # Add prompt tokens
        x = self.incorporate_prompt(x, self.prompt_embeddings)

        # Process through transformer layers
        if self.prompt_type == 'deep':
            x = self.encoder.blocks[0](x)
            for i in range(1, self.total_d_layer):
                x = self.incorporate_prompt(
                    x, self.deep_prompt_embeddings[i-1], self.prompt_tokens)
                x = self.encoder.blocks[i](x)
        else:
            x = self.encoder.blocks(x)

        x = self.encoder.norm(x)
        return x

    def forward(self, x, shuffle=False, return_features=False):
        x = self.forward_features(x, shuffle=shuffle)
        if return_features:
            return x[:, 0, :]  # Return class token features
        return self.encoder.forward_head(x)
