
from recbole.model.abstract_recommender import SequentialRecommender
from typing import Union, Optional
import math
from functools import partial
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm import Mamba2


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    print("--- 警告: 未找到 Triton RMSNorm 内核。将回退到 PyTorch 实现。 ---")
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def _init_weights(module, n_layer, initializer_range=0.02,  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    """ V2 的权重初始化 """
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class FeedForward(nn.Module):
    """ 
    标准的 FFN 模块
    它包含自己的残差连接和归一化 (Post-Norm)
    """
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.ReLU() # 您也可以换成 nn.GELU()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        # 预归一化 (Pre-Norm)
        norm_input = self.LayerNorm(input_tensor)
        
        # FFN 计算
        hidden_states = self.w_1(norm_input)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        
        # 残差连接
        hidden_states = self.dropout(hidden_states) + input_tensor
        return hidden_states

class SimpleMoELayer(nn.Module):

    def __init__(self, d_model, num_experts=4, inner_size_factor=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # 1. 创建 N 个 FFN 专家
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_model * inner_size_factor, dropout) 
            for _ in range(num_experts)
        ])
        
        # 2. 创建门控网络 (Gating Network)
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x 形状: [B, L, D]
        
        # 1. 计算门控权重
        # [B, L, D] -> [B, L, num_experts]
        gating_logits = self.gating_network(x)
        gating_weights = F.softmax(gating_logits, dim=-1) # [B, L, num_experts]

        # 2. 计算所有专家的输出
        expert_outputs = [expert(x) for expert in self.experts] # List of N * [B, L, D]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=2) # [B, L, N, D]
        
        # 3. 加权平均
        # [B, L, 1, N] @ [B, L, N, D] -> [B, L, 1, D]
        weighted_output = torch.matmul(
            gating_weights.unsqueeze(-2), 
            stacked_expert_outputs
        ).squeeze(-2) # [B, L, D]
        
        return weighted_output


class Block(nn.Module):

    def __init__(self, dim, mixer_cls, channel_mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        # 1. Mamba 模块 (序列混合)
        self.mixer = mixer_cls(dim) 
        self.norm_mixer = norm_cls(dim) # Mamba 之前的 Pre-Norm
        
        # 2. MoE / FFN 模块 (通道混合)
        self.channel_mixer = channel_mixer_cls # 它内部自带归一化和残差
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm_mixer, (nn.LayerNorm, RMSNorm)), "Only LayerNorm and RMSNorm are supported"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        
        # Mamba 部分 
        if not self.fused_add_norm:
            residual_mixer = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states_normed = self.norm_mixer(residual_mixer.to(dtype=self.norm_mixer.weight.dtype))
            if self.residual_in_fp32:
                residual_mixer = residual_mixer.to(torch.float32)
        else:
            # 高性能融合 Add + Norm
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_mixer, RMSNorm) else layer_norm_fn
            hidden_states_normed, residual_mixer = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_mixer.weight,
                self.norm_mixer.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm_mixer.eps,
            )
            
        hidden_states_mamba = self.mixer(hidden_states_normed, inference_params=inference_params)
        
        # FNN
        # Mamba 的输出 + 其输入残差，作为通道混合层的输入
        moe_input = (self.drop_path(hidden_states_mamba) + residual_mixer)
        
        # MoE/FFN 模块 (SimpleMoELayer 或 FeedForward)
        hidden_states_moe = self.channel_mixer(moe_input)
        
        # 最终的输出是 MoE 的输出，残差是 MoE 的输入
        return hidden_states_moe, moe_input

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, 
                 residual_in_fp32=False, fused_add_norm=False, layer_idx=None, 
                 drop_path=0., device=None, dtype=None,
                 use_moe=False, num_experts=4, moe_inner_factor=4, moe_dropout=0.2):

    if ssm_cfg is None: ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    #Mamba 序列混合器
    mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    
    #通道混合器 
    if use_moe:
        print(f"--- Layer {layer_idx}: 使用 MoE (Experts: {num_experts}) ---")
        channel_mixer_cls = SimpleMoELayer(
            d_model, 
            num_experts=num_experts, 
            inner_size_factor=moe_inner_factor,
            dropout=moe_dropout
        )
    else:
        print(f"--- Layer {layer_idx}: 使用标准 FFN ---")
        channel_mixer_cls = FeedForward(
            d_model, 
            d_model * moe_inner_factor, # FFN 内部维度也由 factor 控制
            dropout=moe_dropout
        )
        
    block = Block(
        d_model,
        mixer_cls,
        channel_mixer_cls, # 传入 MoE 或 FFN 模块
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
 
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            shuffle_probs: list = [0.3, 0.5, 0.7], 
            use_moe: bool = False,
            num_experts: int = 4,
            moe_inner_factor: int = 4,
            moe_dropout: float = 0.2,
            # ---
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                # 避免抛出错误，但如果 fused_add_norm=True 仍会失败
                print("--- 严重警告: fused_add_norm=True 但 Triton 内核导入失败! ---")
                print("--- 请在实例化 MixerModel 时设置 fused_add_norm=False ---")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    moe_inner_factor=moe_inner_factor,
                    moe_dropout=moe_dropout,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()
        
        # 可学习的重要性预测器 (MLP)
        self.importance_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid() # 输出 0-1 之间的重要性得分
        )
        # 渐进式Shuffle
        self.shuffle_probs = shuffle_probs


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def _attention_guided_shuffle_forward(
        self, 
        x: Tensor, 
        residual: Optional[Tensor], 
        layer: nn.Module, 
        inference_params=None
    ):

        B, L, D = x.shape
        
        # 计算重要性 O(L)
        importance = self.importance_predictor(x).squeeze(-1) # [B, L]
        
        # 逻辑：生成掩码
        # True = 不重要, 打乱; False = 重要, 保留
        shuffle_mask = (torch.rand(B, L, device=x.device) > importance).unsqueeze(-1) # [B, L, 1]

        # 逻辑：准备完全打乱的索引
        shuffled_indices = torch.randperm(L, device=x.device).unsqueeze(0).repeat(B, 1) # [B, L]
        inverse_indices = torch.argsort(shuffled_indices, dim=1) # [B, L]
        
        shuffled_indices_lookup = shuffled_indices.unsqueeze(-1).expand(-1, -1, D)
        inverse_indices_lookup = inverse_indices.unsqueeze(-1).expand(-1, -1, D)

        #融合输入
        x_shuffled = x.gather(1, shuffled_indices_lookup)
        x_permuted = torch.where(shuffle_mask, x_shuffled, x)

        if residual is not None:
            res_shuffled = residual.gather(1, shuffled_indices_lookup)
            res_permuted = torch.where(shuffle_mask, res_shuffled, residual)
        else:
            res_permuted = None

        # Mamba+fnn Block 处理
        output_permuted, res_out_permuted = layer(
            x_permuted, res_permuted, inference_params=inference_params
        )

        # 准备完全恢复的序列
        output_unshuffled = output_permuted.gather(1, inverse_indices_lookup)
        
        #  融合输出 (恢复)
        output = torch.where(shuffle_mask, output_unshuffled, output_permuted)

        if res_out_permuted is not None:
            res_out_unshuffled = res_out_permuted.gather(1, inverse_indices_lookup)
            residual_out = torch.where(shuffle_mask, res_out_unshuffled, res_out_permuted)
        else:
            residual_out = None
            
        return output, residual_out

    def forward(self, input_ids, pos=None, inference_params=None):
        hidden_states = input_ids  
        residual = None
        if pos is not None:
            hidden_states = hidden_states + pos
            
        for i, layer in enumerate(self.layers):
            
            # --- HYBRID Shuffle 策略 ---
            prob = self.shuffle_probs[min(i, len(self.shuffle_probs)-1)]
            
            if self.training and torch.rand(1).item() < prob:
                hidden_states, residual = self._attention_guided_shuffle_forward(
                    hidden_states, residual, layer, inference_params=inference_params
                )
            else:
                # 执行标准 Block (Mamba + FFN)
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

            hidden_states = self.drop_out_in_block(hidden_states)
            
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class IGTMRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(IGTMRec, self).__init__(config, dataset)
        
        self.hidden_size = config["hidden_size"] 
        self.num_layers = config["num_layers"]   
        self.dropout_prob = config["dropout_prob"] 
        self.beta = config["beta"]
        self.norm_embedding = config['norm_embedding']
        
        # Hyperparameters for SSD Block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]  
        self.expand = config["expand"]  
        self.headdim = config['headdim']
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)  # 0 -> mask_token

        # 加载多模态特征
        dataset_name = config['dataset']
        feature_path = f"./dataset/{dataset_name}"

        img_feat = nn.Embedding.from_pretrained(
            torch.load(feature_path+"/img_feat.pt"), freeze=True
        )
        

        text_feat = nn.Embedding.from_pretrained(
            torch.load(feature_path+"/text_feat.pt"), freeze=True
        )

        
        # 添加padding行
        self.img_feat = nn.Embedding.from_pretrained(
            torch.cat((torch.zeros(1, img_feat.weight.shape[-1]), img_feat.weight), dim=0),
            freeze=True
        )
        self.text_feat = nn.Embedding.from_pretrained(
            torch.cat((torch.zeros(1, text_feat.weight.shape[-1]), text_feat.weight), dim=0),
            freeze=True
        )

        # 多模态特征转换
        self.img_trans = nn.Sequential(
            nn.Linear(self.img_feat.weight.shape[-1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.text_trans = nn.Sequential(
            nn.Linear(self.text_feat.weight.shape[-1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        
        self.IGTMRec_layers = nn.ModuleList([
            IGTMRecLayer(
                beta = self.beta,
                d_model=self.hidden_size,  
                d_state=self.d_state,      
                d_conv=self.d_conv,        
                expand=self.expand,        
                dropout=self.dropout_prob, 
                num_layers=self.num_layers,
                headdim = self.headdim,
                # 新增：MoM 相关超参（可从 config 读）

            ) for _ in range(self.num_layers)
        ])


        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, cum_item_length, item_idx, flip_index,time_diff):

        # 获取多模态特征
        img_emb = self.img_feat(item_seq)
        text_emb = self.text_feat(item_seq)
        img_emb = self.img_trans(img_emb)
        text_emb = self.text_trans(text_emb)

        item_emb = self.item_embedding(item_seq)
        # 是否对初始嵌入归一化
        if self.norm_embedding == True:
            item_emb = self.dropout(item_emb)
            item_emb = self.LayerNorm(item_emb)
            img_emb = self.dropout(self.LayerNorm(img_emb))
            text_emb = self.dropout(self.LayerNorm(text_emb))


        for i in range(self.num_layers):
            item_emb = self.IGTMRec_layers[i](item_emb,img_emb,text_emb, item_idx, flip_index,time_diff)

        # gather_last_token_output
        gather_index = cum_item_length - 1 # [B]
        seq_output = item_emb[0, gather_index, :]

        return seq_output
    
    def calculate_loss(self, item_id, item_id_list, cum_item_length, item_idx, flip_index,time_diff):

        item_seq = item_id_list.unsqueeze(0)     # [1, cat_dim such as 13297]
        item_idx = item_idx.unsqueeze(0)

        seq_output = self.forward(item_seq, cum_item_length, item_idx, flip_index,time_diff) # [B, hidden_size]
        pos_items = item_id                      # [B]

        test_item_emb = self.item_embedding.weight # [item_num, hidden_size]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # [B, item_num]

        loss = self.loss_fct(logits, pos_items)
        return loss


    def full_sort_predict(self, item_id_list, cum_item_length, item_idx, flip_index,positive_i,time_diff):
        # positive_i:目标物品
        item_seq = item_id_list.unsqueeze(0)    # [1, cat_dim such as 13297]
        item_idx = item_idx.unsqueeze(0)
        
        seq_output = self.forward(item_seq, cum_item_length, item_idx, flip_index,time_diff) # [B, hidden_size]
        test_items_emb = self.item_embedding.weight # [item_num, hidden_size]

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        valid_loss = self.loss_fct(scores, positive_i)

        return scores,valid_loss

    
# 双向mamba2+FFN
class IGTMRecLayer(nn.Module):
    def __init__(self, beta, d_model, d_state, d_conv, expand, dropout, num_layers, headdim):
        super().__init__()
        self.beta = beta
        self.num_layers = num_layers
        self.d_model=d_model
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
        self.inner_size=d_model*4

        self.Model = MixerModel( 
            d_model,
            2,
            rms_norm=False,
            fused_add_norm=False,  # <--- 将此参数设置为 False
            shuffle_probs=[0.3, 0.5, 0.7],
            dtype=torch.float32 
        )
        self.gtu_fusion = MultiGTUFusion(d_model=d_model)    

        self.forward_ssd = Mamba2(
                d_model=d_model,  #模型特征维度
                d_state=d_state,  #状态维度
                headdim = headdim,  #注意力维度
                d_conv=d_conv,     # 卷积核尺寸
                expand=expand,     #特征拓展因子
            )   
        
    def forward(self, item_emb,img_emb,text_emb, item_idx, flip_index,time_diff):
        fused=self.gtu_fusion(item_emb,img_emb,text_emb,time_diff)
          
        mamba_out=self.Model(fused)

        return mamba_out




class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.LeakyReLU()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        # Feed-Forward Network
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)

        # residual connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class MultiGTUFusion(nn.Module):
    """
    使用 Multi_GTU 融合频域、Mamba和时间特征
    输入: 三个 [B, L, D] 的特征
    输出: [B, L, D] 的融合特征
    """
    def __init__(self, d_model, kernel_size=[3, 5, 7],max_time=365):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        self.item_proj = nn.Linear(d_model, d_model)
        self.img_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)
        
        self.gtu = GTU(in_channels=d_model, time_strides=1, kernel_size=kernel_size[0])
        self.gtu2 = GTU(in_channels=d_model, time_strides=1, kernel_size=kernel_size[1])
        self.gtu3 = GTU(in_channels=d_model, time_strides=1, kernel_size=kernel_size[2])
        
        self.align_proj1 = nn.Linear(d_model, d_model)
        self.align_proj2 = nn.Linear(d_model, d_model)
        self.align_proj3 = nn.Linear(d_model, d_model)
        
        self.max_time = max_time
        
        # 时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 三个模态的门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 输入：item_emb + time_encoding
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 3),  # 输出：[weight_id, weight_img, weight_text]
            nn.Softmax(dim=-1)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, item_emb, img_emb, text_emb,time_diff):
        """
        Args:
            item_emb, img_emb, text_emb: [B, L, D]
            time_diff: [B, L] - 每个位置到当前时间的间隔（天/小时）
        """
        B, L, D = item_emb.shape
        # 1. 归一化时间（防止数值过大）
        time_norm = (time_diff / self.max_time).unsqueeze(-1)  # [B, L, 1]
        
        # 2. 时间编码
        time_encoding = self.time_encoder(time_norm)  # [B, L, D]
        gate_input = torch.cat([item_emb, time_encoding], dim=-1)  # [B, L, 2D]
        modal_weights = self.gate_network(gate_input)  # [B, L, 3]
        
        # 投影三个特征
        item_out = self.item_proj(item_emb)    # [B, L, D]
        img_out = self.img_proj(img_emb)  # [B, L, D]
        text_out = self.text_proj(text_emb)     # [B, L, D]
        
        # 堆叠三个特征: [B, L, D] -> [B, 3, L, D] -> [B, D, 3, L]
        stacked = torch.stack([item_out, img_out, text_out], dim=1)  # [B, 3, L, D]
        x = stacked.permute(0, 3, 1, 2)  # [B, D, 3, L] 符合GTU的输入格式 (B, C, N, T)
        
        # 应用 GTU (会改变时间维度)
        x1 = self.gtu(x)   # [B, D, 3, L-k1+1]
        x2 = self.gtu2(x)  # [B, D, 3, L-k2+1]
        x3 = self.gtu3(x)  # [B, D, 3, L-k3+1]
        
        # 使用padding对齐时间维度到原始长度L
        pad1 = self.kernel_size[0] - 1  # 需要pad的长度
        pad2 = self.kernel_size[1] - 1
        pad3 = self.kernel_size[2] - 1
        
        # 在时间维度(最后一维)进行padding: (left, right, top, bottom)
        x1_padded = F.pad(x1, (pad1, 0))  # [B, D, 3, L]
        x2_padded = F.pad(x2, (pad2, 0))  # [B, D, 3, L]
        x3_padded = F.pad(x3, (pad3, 0))  # [B, D, 3, L]
        
        # 池化是常见的聚合方式，用于平滑多源特征，避免单一模态主导。
        x1_agg = x1_padded.mean(dim=2)  # [B, D, L]
        x2_agg = x2_padded.mean(dim=2)  # [B, D, L]
        x3_agg = x3_padded.mean(dim=2)  # [B, D, L]
        
        # 转换回 [B, L, D]
        x1_agg = x1_agg.permute(0, 2, 1)  # [B, L, D]
        x2_agg = x2_agg.permute(0, 2, 1)  # [B, L, D]
        x3_agg = x3_agg.permute(0, 2, 1)  # [B, L, D]
        
        # 通过线性层进一步处理
        x1_final = self.align_proj1(x1_agg)  # [B, L, D]
        x2_final = self.align_proj2(x2_agg)  # [B, L, D]
        x3_final = self.align_proj3(x3_agg)  # [B, L, D]

        w_id = modal_weights[..., 0:1]    # [B, L, 1]
        w_img = modal_weights[..., 1:2]
        w_text = modal_weights[..., 2:3]
        fused = w_id * x1_final + w_img * x2_final + w_text * x3_final
        output = self.norm(fused + item_emb)
       
        return output

class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, 
                                 kernel_size=(1, kernel_size), 
                                 stride=(1, time_strides))

    def forward(self, x):

        x_causal_conv = self.con2out(x)  # (B, 2C, N, T-K+1)
        x_p = x_causal_conv[:, :self.in_channels, :, :]   # (B, C, N, T-K+1)
        x_q = x_causal_conv[:, -self.in_channels:, :, :]  # (B, C, N, T-K+1)
        x_gtu = self.tanh(x_p) * self.sigmoid(x_q)        # (B, C, N, T-K+1)
        return x_gtu

