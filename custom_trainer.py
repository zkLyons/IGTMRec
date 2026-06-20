import os
from logging import getLogger
from time import time
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.cuda.amp as amp


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

        self.model.train()
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
            scores,valid_loss = eval_func(item_id_list, cum_item_length, item_idx, flip_index,positive_i,time_diff)
            inference_time += time() - inference_time_per_batch
            
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(scores, None, positive_u, positive_i)

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
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
    

    def _full_sort_batch_eval(self, item_id_list, cum_item_length, item_idx, flip_index,positive_i,time_diff):
 
        scores,valid_loss = self.model.full_sort_predict(item_id_list, cum_item_length, item_idx, flip_index,positive_i,time_diff) # [B, item_num]
 
        scores = scores.view(-1, self.tot_item_num)  # [B, item_num]
 
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

        hit_count = np.sum(ranks <= K)
        hit_rate = hit_count / B
        print(f"Step {global_step}: Hit@{K} = {hit_rate * 100:.2f}%")
        writer.add_histogram(tag, ranks, global_step)
        writer.add_scalar(f'{tag}_Hit@{K}', hit_rate, global_step)