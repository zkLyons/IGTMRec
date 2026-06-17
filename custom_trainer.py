import os
from logging import getLogger
from time import time
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.cuda.amp as amp
# from tensorboardX import SummaryWriter   
# from torch.utils.tensorboard import SummaryWriter

from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
from torch.nn.parallel import DistributedDataParallel
from recbole.trainer import Trainer

class IGTMRecTrainer(Trainer):

    def __init__(self, config, model):
        super(IGTMRecTrainer, self).__init__(config, model)
        self.tensorboard = get_tensorboard(self.logger)
        self.count=0

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        # 清除---------
        # torch.backends.cuda.cufft_plan_cache.clear()
        # 清除---------

        self.model.train()
        # 使用模型自带的损失计算方法。
        loss_func = self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        training_time_per_epoch = 0
        # 创建梯度缩放器（用于混合精度训练）
        scaler = amp.GradScaler('cuda',enabled=self.enable_scaler)

        for batch_idx, (item_id, item_id_list, cum_item_length, item_idx, flip_index,time_diff) in enumerate(iter_data):
            training_time_per_batch = time()
            item_id = item_id.to(self.device)
            item_id_list = item_id_list.to(self.device)
            cum_item_length = cum_item_length.to(self.device)
            item_idx = item_idx.to(self.device)
            flip_index = flip_index.to(self.device)
            time_diff = time_diff.to(self.device)

            self.optimizer.zero_grad()
            sync_loss = 0

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(item_id, item_id_list, cum_item_length, item_idx, flip_index,time_diff)

            loss = losses
            total_loss = ( losses.item() if total_loss is None else total_loss + losses.item())
            self._check_nan(loss)

            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

            training_time_per_epoch += time() - training_time_per_batch
        print(f'training_time_per_epoch: {training_time_per_epoch}')


        return total_loss
    # 禁用梯度计算
    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return
        # 加载最佳模型，最后一轮测试的时候使用。
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()
        # 评估模式？全排序vs负采样
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        inference_time = 0

        num_sample = 0
        for batch_idx, (item_id_list, cum_item_length, item_idx, flip_index, positive_u, positive_i,time_diff) in enumerate(iter_data):
            item_id_list = item_id_list.to(self.device)
            cum_item_length = cum_item_length.to(self.device)
            item_idx = item_idx.to(self.device)
            flip_index = flip_index.to(self.device)
            positive_u = positive_u.to(self.device)
            positive_i = positive_i.to(self.device)
            time_diff = time_diff.to(self.device)


            num_sample += len(cum_item_length)
            inference_time_per_batch = time()
            # 调用模型的评估函数，返回[Batch,total_item_num]
            scores,valid_loss = eval_func(item_id_list, cum_item_length, item_idx, flip_index,positive_i,time_diff)
            inference_time += time() - inference_time_per_batch
            
            # 显示设备状态。
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
                # 完成评估数据收集
            self.eval_collector.eval_batch_collect(scores, None, positive_u, positive_i)

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        # 计算评估指标，hit@k,ndcg,mrr
        result = self.evaluator.evaluate(struct)
        self.tensorboard.add_scalar("Hit@10", result.get('hit@10'),self.count)
        self.tensorboard.add_scalar("Hit@20", result.get('hit@20'),self.count)
        self.tensorboard.add_scalar("ndcg@10", result.get('ndcg@10'),self.count)
        self.tensorboard.add_scalar("ndcg@20", result.get('ndcg@20'),self.count)
        self.tensorboard.add_scalar("mrr@10", result.get('mrr@10'),self.count)
        self.tensorboard.add_scalar("mrr@20", result.get('mrr@20'),self.count)
        self.tensorboard.add_scalars("loss", {"valid_loss": valid_loss}, self.count)
        self.count+=1
            
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")

        print(f'inference_time: {inference_time}')
        return result
    

    # 扰动函数
    import numpy as np # 确保在文件顶部导入 numpy

    import numpy as np # 确保在文件顶部导入 numpy

    @torch.no_grad() # 我们不需要为这个操作计算梯度
    def _perturb_sequences(self, item_id_list, time_diff, cum_item_length, perturb_rate):

        
        # 0% 扰动率 = 什么都不做，直接返回
        if perturb_rate == 0:
            print('扰动率为0')
            return item_id_list, time_diff

        is_shape_mismatch = False
        if item_id_list.dim() == 1 and time_diff.dim() == 2 and \
           time_diff.shape[0] == 1 and item_id_list.shape[0] == time_diff.shape[1]:
            
            is_shape_mismatch = True 
        
            if not hasattr(self, '_warned_shape_mismatch_v3'):
                self.logger.info("error")
                self._warned_shape_mismatch_v3 = True
        
        # 克隆数据
        new_item_id_list_np = item_id_list.cpu().clone().numpy() # 1D array [N_total]
        new_time_diff_np = time_diff.cpu().clone().numpy() # 2D array [1, N_total]

        cum_length_np = cum_item_length.cpu().numpy()
        start_idx = 0
        
        for end_idx in cum_length_np:
            seq_len = end_idx - start_idx
            
            if seq_len < 2:
                start_idx = end_idx
                continue
            
            num_pairs = seq_len - 1
            n_swaps = int(np.ceil(num_pairs * perturb_rate)) 
            
            if n_swaps == 0:
                start_idx = end_idx
                continue
                
            j_indices = np.random.choice(num_pairs, n_swaps, replace=False)
            
            for j in j_indices:
                global_j = start_idx + j
                global_j_plus_1 = global_j + 1
                

                new_item_id_list_np[global_j], new_item_id_list_np[global_j_plus_1] = \
                    new_item_id_list_np[global_j_plus_1], new_item_id_list_np[global_j]
                
                if is_shape_mismatch:

                    new_time_diff_np[0, global_j], new_time_diff_np[0, global_j_plus_1] = \
                        new_time_diff_np[0, global_j_plus_1], new_time_diff_np[0, global_j]
                else:
   
                    new_time_diff_np[global_j], new_time_diff_np[global_j_plus_1] = \
                        new_time_diff_np[global_j_plus_1], new_time_diff_np[global_j]

            start_idx = end_idx
            
        # 5. 将两份扰动过的 numpy 数组传回 GPU
        perturbed_item_id_list = torch.tensor(new_item_id_list_np, dtype=item_id_list.dtype, device=item_id_list.device)
        perturbed_time_diff = torch.tensor(new_time_diff_np, dtype=time_diff.dtype, device=time_diff.device)
        
        return perturbed_item_id_list, perturbed_time_diff


    def _full_sort_batch_eval(self, item_id_list, cum_item_length, item_idx, flip_index,positive_i,time_diff):
        # Note: interaction without item ids
        # 返回的是一个全排序的分数矩阵，形状为 [B, item_num]，其中 B 是批次大小，item_num 是物品总数。对应每一个用户对所有物品的预测分数。
        scores,valid_loss = self.model.full_sort_predict(item_id_list, cum_item_length, item_idx, flip_index,positive_i,time_diff) # [B, item_num]
        # 在此对scores的维度进行处理，使其符合预期形状。
        scores = scores.view(-1, self.tot_item_num)  # [B, item_num]
        # 将第一个物品分数设置为无穷小，我觉得没什么必要
        scores[:, 0] = -np.inf # [B, item_num]
        return scores,valid_loss
    
    def log_topk_rank_histogram(writer, logits, pos_ids, global_step, K=10, tag='TopK_Rank_Hist'):
        """
        将正样本排名分布写入 TensorBoard 直方图
        Args:
            writer: SummaryWriter 实例
            logits: [B, N_item]
            pos_ids: [B]，正样本索引
            global_step: 当前训练步数或 epoch
            K: Top-K
            tag: TensorBoard 日志的名字
        """
        B, N = logits.size()
        ranks = []
        for i in range(B):
            pos_score = logits[i, pos_ids[i]]
            higher_scores = (logits[i] > pos_score).sum().item()
            rank = higher_scores + 1
            ranks.append(rank)

        ranks = np.array(ranks)

        # 计算命中率
        hit_count = np.sum(ranks <= K)
        hit_rate = hit_count / B
        print(f"Step {global_step}: Hit@{K} = {hit_rate * 100:.2f}%")
        # 写直方图
        writer.add_histogram(tag, ranks, global_step)
        # 也可以写标量命中率
        writer.add_scalar(f'{tag}_Hit@{K}', hit_rate, global_step)