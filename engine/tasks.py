from .train_engine import BaseTrainer
from .val_engine import BaseValidator
import torch
from ._taskRegistry import register_task
from tqdm import tqdm
from models.dinov3_det import PostProcess
import torch.distributed as dist
from pycocotools.cocoeval import COCOeval

__all__ = ["ClassificationTrainer", "ClassificationValidator"]


@register_task("train_classification")
class ClassificationTrainer(BaseTrainer):
    def __init__(self, criterion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion # 分类任务特有的 criterion

        # 🌟 核心修改：组合模式！直接复用验证器引擎
        if self.val_loader is not None:
            self.validator = ClassificationValidator(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device
            )
        else:
            self.validator = None

    def train_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        log_dict = {"Loss": loss.item()}
        return loss, log_dict

    def evaluate(self):
        """直接委托给专业的验证器引擎"""
        if self.validator:
            return self.validator.evaluate()
        return 0.0, {}


@register_task("val_classification")
class ClassificationValidator(BaseValidator):
    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0.0

        num_batches = len(self.dataloader)
        if num_batches == 0:
            raise RuntimeError(
                "🚨 致命错误：训练集 DataLoader 的批次数量为 0！\n"
                "可能的原因：\n"
                "1. 数据集目录为空，或读取逻辑导致 0 样本。\n"
                "2. 训练集总样本数小于 Batch Size，且 DataLoader 开启了 drop_last=True。\n"
                "请立即检查数据流配置！"
            )
        
        with torch.no_grad():
            pbar = tqdm(self.dataloader, desc="📉 Validating", leave=False)
            for batch in pbar:
                inputs, targets = self._to_device(batch)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
                
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.dataloader)
        
        print(f"\n📊 [验证结果] Accuracy: {acc*100:.2f}% | Loss: {avg_loss:.4f}")
        return acc, {"Accuracy": acc, "Loss": avg_loss}
    

COCO_80_TO_90_REVERSE = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65,
    60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79,
    70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
}


@register_task("val_detection")
class DetectionValidator(BaseValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 模型输出统一为归一化 [0,1] cxcywh 格式，PostProcess 用标准模式解码即可
        self.postprocessor = PostProcess(topk=100).to(self.device)
    def evaluate(self, is_coco: bool = False):
        self.model.eval()
        results = []


        with torch.no_grad():
            pbar = tqdm(self.dataloader, desc="📉 正在推理验证集", leave=False, disable=not self.is_main_process)
            for batch in pbar:
                inputs, targets = self._to_device(batch)

                outputs = self.model(inputs)

                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

                # PostProcess 将归一化 [0,1] 坐标转为原始图像绝对像素坐标
                batch_res = self.postprocessor(outputs, orig_target_sizes)
                
                for target, res in zip(targets, batch_res):
                    img_id = target["image_id"].item()
                    
                    for box, score, label in zip(res['boxes'], res['scores'], res['labels']):
                        # 过滤掉极低置信度的框，加速后续评估
                        if score.item() < 0.001: 
                            continue
                            
                        # 转换坐标系: xyxy -> xywh
                        x1, y1, x2, y2 = box.tolist()
                        w, h = x2 - x1, y2 - y1
                        
                        # 映射类别 ID
                        if is_coco:
                            coco_cat_id = COCO_80_TO_90_REVERSE.get(label.item(), label.item())
                        else:
                            coco_cat_id = label.item()
                        
                        results.append({
                            "image_id": img_id,
                            "category_id": coco_cat_id,
                            "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                            "score": round(score.item(), 5)
                        })
                
        # ==========================================
        # 🌟 核心点：多卡环境下的结果汇总 (DDP 同步)
        # ==========================================
        if dist.is_available() and dist.is_initialized():
            # 收集所有 GPU 上的 results 列表
            gathered_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_results, results)
            # 把嵌套列表展平
            all_results = []
            for res_list in gathered_results:
                all_results.extend(res_list)
        else:
            all_results = results

        # ==========================================
        # 🌟 核心点：调用你熟悉的 COCO API 进行评估 (仅主进程执行)
        # ==========================================
        map_50_95 = 0.0
        val_logs = {"mAP_50_95": 0.0, "mAP_50": 0.0}
        
        if self.is_main_process:
            if len(all_results) == 0:
                print("⚠️ 警告：当前模型没有预测出任何置信度 > 0.001 的目标。")
                return 0.0, val_logs
                
            print(f"✅ 推理完成，共生成 {len(all_results)} 个有效预测框，准备计算 mAP...")
            
            # 获取原始的 COCO 对象
            coco_gt = self.dataloader.dataset.coco
            
            # 🔥 架构师优化：不要写 JSON 文件！直接在内存中加载字典列表，速度快 10 倍！
            coco_dt = coco_gt.loadRes(all_results)
            
            # 执行你的评测逻辑
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # 提取指标
            map_50_95 = coco_eval.stats[0] # AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
            map_50 = coco_eval.stats[1]    # AP @[ IoU=0.50      | area=   all | maxDets=100 ]
            
            val_logs = {
                "mAP_50_95": map_50_95,
                "mAP_50": map_50
            }

        # 如果是多卡训练，主进程的 mAP 计算完了，需要广播给其他显卡，确保大家保存模型的进度一致
        if dist.is_available() and dist.is_initialized():
            metric_tensor = torch.tensor([map_50_95], dtype=torch.float32, device=self.device)
            dist.broadcast(metric_tensor, src=0)
            map_50_95 = metric_tensor.item()

        return map_50_95, val_logs
    

@register_task("train_detection")
class DetectionTrainer(BaseTrainer):
    def __init__(self, criterion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion # DETR 专属的 SetCriterion

        # 🌟 组合模式：直接复用刚才写的验证器引擎
        if self.val_loader is not None:
            self.validator = DetectionValidator(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                is_main_process=self.is_main_process # 传递主进程标志
            )
        else:
            self.validator = None

    def train_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)
        
        # DETR 返回的是装满损失项的字典
        loss_dict = self.criterion(outputs, targets)
        
        # 将所有的 loss (包含主干 loss 和所有 Decoder 层的 aux_loss) 加和
        total_loss = sum(loss_dict.values())
        
        # 整理日志：不仅记录总 Loss，还把每一项具体 Loss 展开，方便在 TensorBoard 里排查问题
        log_dict = {k: v.item() for k, v in loss_dict.items()}
        log_dict["Total_Loss"] = total_loss.item()
        
        return total_loss, log_dict

    def evaluate(self):
        """直接委托给专业的检测验证器引擎"""
        if self.validator:
            return self.validator.evaluate(is_coco=True)
        return float('-inf'), {}