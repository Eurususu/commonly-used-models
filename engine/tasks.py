from .train_engine import BaseTrainer
from .val_engine import BaseValidator
import torch
from ._taskRegistry import register_task
import tqdm

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
    

@register_task("val_detection")
class DetectionValidator(BaseValidator):
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        
        # 💡 [进阶预留] 
        # 如果未来接入 COCO 评估，这里需要初始化 coco_evaluator 对象
        # coco_evaluator = CocoEvaluator(base_ds, iou_types=["bbox"])

        with torch.no_grad():
            pbar = tqdm(self.dataloader, desc="📉 Validating DETR", leave=False)
            for batch in pbar:
                inputs, targets = self._to_device(batch)
                
                outputs = self.model(inputs)
                loss_dict = self.criterion(outputs, targets)
                
                # DETR 返回的是 Loss 字典，把它们全部加起来
                loss = sum(loss_dict.values())
                total_loss += loss.item()
                
                # 💡 [进阶预留] 
                # 这里会调用 PostProcess 把 outputs 转成真实坐标，然后塞给 coco_evaluator
                # results = postprocessors['bbox'](outputs, target_sizes)
                # coco_evaluator.update(res)

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
        avg_loss = total_loss / len(self.dataloader)
        
        # 💡 [进阶预留] 
        # coco_evaluator.synchronize_between_processes()
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        # map_50_95 = coco_evaluator.coco_eval['bbox'].stats[0]
        
        print(f"\n📊 [验证结果] Val Loss: {avg_loss:.4f} (注意：目标检测最终应以 mAP 为准)")
        
        # ⚠️ 极其关键的 Trick：
        # 因为 BaseTrainer 中保存模型的逻辑是：主指标越大越好 (main_metric > best_metric)
        # 既然我们现在暂时用 Loss 评估，Loss 是越小越好。
        # 所以我们返回 -avg_loss 作为主指标，这样 Loss 越低，负值就越大，就能正确触发保存模型！
        # 如果以后换成了 mAP，这里就直接返回 mAP 即可。
        return -avg_loss, {"Val_Loss": avg_loss}
    

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
                device=self.device
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
            return self.validator.evaluate()
        return float('-inf'), {}