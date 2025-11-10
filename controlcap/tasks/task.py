import copy
import logging
import json
import tqdm
import shutil
import os
import sys
import cv2
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import lavis.common.dist_utils as dist_utils
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.common.dist_utils import (
    get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized, main_process
)
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks.base_task import BaseTask
from controlcap.common.evaluation.eval_densecap import DenseCapEvaluator


@registry.register_task("controlcap")
class ControlCapTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.evaluate = kwargs.get("evaluate", False)
        self.eval_dataset_name = kwargs.get("eval_dataset_name", None)
        self.report_metric = kwargs.get("report_metric", True)
        self.visualize = kwargs.get("visualize", True)

        # Optional cap on evaluation images (env var wins if set)
        self.max_eval_images = kwargs.get("max_eval_images", None)
        _env = os.environ.get("EVAL_MAX_IMAGES", None)
        if _env is not None:
            try:
                self.max_eval_images = int(_env)
                print(f"[INFO] Limiting evaluation metrics to first {self.max_eval_images} images")
            except ValueError:
                pass

    @classmethod
    def setup_task(cls, cfg):
        return cls(**dict(cfg.run_cfg))

    def train_step(self, model, samples):
        return model(samples)

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """Inner loop that supports both epoch- and iter-based training."""
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_llm", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_tag", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")
        header = f"Train: data epoch: [{epoch}]"
        inner_epoch = epoch if start_iters is None else (start_iters // iters_per_epoch)
        if start_iters is not None:
            header = header + f"; inner epoch [{inner_epoch}]"

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update({"epoch": inner_epoch, "num_iters_per_epoch": iters_per_epoch, "iters": i})

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)
                loss_all = loss["loss"]
                loss_llm = loss.get("loss_llm", torch.tensor(0.0, device=loss_all.device))
                loss_tag = loss.get("loss_tag", torch.tensor(0.0, device=loss_all.device))

            if use_amp:
                scaler.scale(loss_all).backward()
            else:
                loss_all.backward()

            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss_all)
            metric_logger.update(loss_llm=loss_llm)
            metric_logger.update(loss_tag=loss_tag)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if i % log_freq == 0:
                state = dict(
                    epoch=inner_epoch,
                    iter="%.4d" % i,
                    lr="%.6f" % optimizer.param_groups[0]["lr"],
                    loss="%.4f" % loss_all.detach().cpu().item(),
                    loss_llm="%.4f" % loss_llm.detach().cpu().item(),
                    loss_tag="%.4f" % loss_tag.detach().cpu().item(),
                )
                self.log_stats(state)

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

    # Toggle topical path during eval via USE_TOPICS=1
    def valid_step(self, model, samples):
        use_topics = os.environ.get("USE_TOPICS", "0") == "1"
        return model.predict_answers_with_topics(samples=samples) if use_topics else model.predict_answers(samples=samples)

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)
        load_ckpt_path = cfg.run_cfg.load_ckpt_path
        if load_ckpt_path is not None:
            model.load_checkpoint(url_or_filename=load_ckpt_path)
        return model

    def build_datasets(self, cfg):
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        eval_dataset_name = self.eval_dataset_name or list(datasets_config)[0]
        if eval_dataset_name not in datasets_config:
            raise ValueError("Eval dataset name not found.")
        self.eval_dataset_ann_path = datasets_config[eval_dataset_name].build_info.annotations.val[0]

        for name in datasets_config:
            dataset_config = datasets_config[name]

            # If evaluating, drop train split build
            if self.evaluate and (
                (self.eval_dataset_name is None and name == list(datasets_config)[0])
                or (self.eval_dataset_name is not None and name == self.eval_dataset_name)
            ):
                dcopy = copy.deepcopy(dataset_config)
                anns = dcopy.build_info.annotations
                if "train" in anns:
                    print("NEW: Skipped training dataset build")
                    anns.pop("train")
                dataset_config = dcopy

            builder = registry.get_builder_class("controlcap")(dataset_config)
            dataset = builder.build_datasets()

            if name != eval_dataset_name:
                dataset.pop("val", None)
                dataset.pop("test", None)
            if self.evaluate:
                dataset.pop("train", None)

            datasets[name] = dataset

        return datasets

    def save_result(self, result, result_dir, filename, remove_duplicate=""):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        result_file = os.path.join(result_dir, f"{filename}_rank{get_rank()}.json")
        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("Merging results.")
            result = []
            for rank in tqdm.tqdm(range(get_world_size())):
                rf = os.path.join(result_dir, f"{filename}_rank{rank}.json")
                tmp = json.load(open(rf, "r"))
                result.extend(tmp)

            id2pred = dict()
            for pred in result:
                id = pred.pop("id")
                id2pred[id] = pred

            gt = json.load(open(self.eval_dataset_ann_path, "r"))
            annotations = gt["annotations"]

            # NEW: accumulate per-image topics sidecar
            topics_by_image = {}
            num_result = 0

            # If requested, limit the metrics/images we keep
            max_keep = self.max_eval_images
            kept = 0

            for annotation in annotations:
                if max_keep is not None and kept >= max_keep:
                    break
                id = annotation["id"]
                if id in id2pred:
                    kept += 1
                    pred = id2pred[id]
                    annotation["extra_info"]["pred_result"] = copy.deepcopy(pred)

                    topics = pred.get("topics", {})
                    if topics:
                        kws = topics.get("main_topic_keywords", [])
                        annotation["topics"] = kws
                        annotation["extra_info"]["scene_topics"] = ", ".join([w for w in kws if isinstance(w, str)])
                        img_id = int(annotation.get("image_id", -1))
                        if img_id >= 0 and img_id not in topics_by_image:
                            topics_by_image[img_id] = topics

            merged_path = os.path.join(result_dir, filename + ".json")
            with open(merged_path, "w") as fw:
                json.dump(gt, fw)

            sidecar = os.path.join(result_dir, "topics_by_image.json")
            with open(sidecar, "w") as fside:
                json.dump(topics_by_image, fside, indent=2)
            logging.info(f":Wrote topics sidecar to ({sidecar}).")
            logging.info(f":Save result to ({merged_path}).")

        return os.path.join(result_dir, filename + ".json")

    @dist_utils.main_process
    def visualize_result(self, result_file, result_dir="./"):
        logging.info(f":Begin visualization ({result_file}).")
        save_dir = os.path.join(result_dir, "viz")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        os.mkdir(save_dir)
        file = COCO(result_file)
        image_root = file.dataset["dataset"]["image_root"]
        imgs = list(file.imgs.items())
        expand_ratio = 5

        max_num = 100
        vis_num = 0

        for image_id, img in tqdm.tqdm(imgs):
            img_path = os.path.join(image_root, img["file_name"])
            anns = file.imgToAnns.get(image_id, [])
            if len(anns) == 0:
                continue
            else:
                vis_num += 1
                if vis_num > max_num:
                    break
            image = cv2.imread(img_path)
            h, w, _ = image.shape
            captions_to_draw = []
            for ann in anns:
                extra_info = ann.get('extra_info', dict())
                pred_result = extra_info.get('pred_result', None)
                if pred_result is None:
                    continue
                caption = pred_result['caption']
                stags = pred_result.get('tag_set1', [])
                otags = pred_result.get('tag_set2', [])
                vis_caption = '[' + ','.join(stags) + '][' + ','.join(otags) + '][' + caption + ']'
                seg = ann["segmentation"]
                if isinstance(seg, list):
                    mask = np.zeros((h, w), np.uint8)
                    for seg_ in seg:
                        mask = cv2.fillPoly(mask, np.array(seg_).reshape(1, -1, 2).astype(np.int64), 1)
                else:
                    if isinstance(seg["counts"], list):
                        seg = mask_util.frPyObjects(seg, *seg["size"])
                    elif not isinstance(seg["counts"], bytes):
                        seg["counts"] = seg["counts"].encode()
                    mask = mask_util.decode(seg)

                x, y, wb, hb = cv2.boundingRect(mask)
                pos = (x*expand_ratio, (y + int(hb/2))*expand_ratio)
                bbox = (x, y, x+wb, y+hb)
                rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)[0].tolist()
                cv2.rectangle(image, [bbox[0], bbox[1]], [bbox[2], bbox[3]], color=rgb, thickness=1)

                captions_to_draw.append((vis_caption, pos, rgb))

            dsize = (w*expand_ratio, h*expand_ratio)
            image = cv2.resize(image, dsize)

            # draw one-line scene topics header, if present
            scene_kw = None
            for ann in anns:
                pr = ann.get("extra_info", {}).get("pred_result", None)
                if pr and "topics" in pr and pr["topics"]:
                    scene_kw = pr["topics"].get("main_topic_keywords", [])
                    break
            if scene_kw:
                header = "scene: " + ", ".join([str(w) for w in scene_kw])[:120]
                cv2.putText(image, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            for caption, pos, rgb in captions_to_draw:
                cv2.putText(image, caption, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=rgb, thickness=3)

            save_path = os.path.join(save_dir, os.path.basename(img["file_name"]))
            cv2.imwrite(save_path, image)

        logging.info(f":Save to ({save_dir}).")
        return

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_dir = registry.get_path("result_dir")
        result_file = self.save_result(
            val_result,
            result_dir=result_dir,
            filename=f"{split_name}",
            remove_duplicate="question_id",
        )

        metrics = {"agg_metrics": 0}
        if self.report_metric:
            if ("reg" in self.eval_dataset_name) or ("refcoco" in self.eval_dataset_name):
                metrics = self.report_metrics_reg(result_file)
            else:
                # Only support vg evaluation now
                supported = ["vg", "grit"]
                if any(ds in self.eval_dataset_name for ds in supported):
                    metrics = self.report_metrics_densecap(result_file=result_file)
                else:
                    logging.info(f":Not support evaluation for dataset ({self.eval_dataset_name}).")

        if self.visualize:
            self.visualize_result(result_file, result_dir)

        return metrics

    @dist_utils.main_process
    def report_metrics_densecap(self, result_file, gt_file_override=None):
        logging.info(f":Begin evaluation ({result_file}).")

        def seg2bbox(seg):
            if isinstance(seg, list):
                seq = []
                for seg_ in seg:
                    seq.extend(seg_)
                x1, y1 = np.array(seq).reshape(-1, 2).min(0)
                x2, y2 = np.array(seq).reshape(-1, 2).max(0)
                bbox = [x1, y1, x2, y2]
            else:
                if isinstance(seg["counts"], list):
                    seg = mask_util.frPyObjects(seg, *seg["size"])
                elif not isinstance(seg["counts"], bytes):
                    seg["counts"] = seg["counts"].encode()
                mask = mask_util.decode(seg)
                x1, x2 = np.nonzero(mask.sum(0) != 0)[0][0], np.nonzero(mask.sum(0) != 0)[0][-1]
                y1, y2 = np.nonzero(mask.sum(1) != 0)[0][0], np.nonzero(mask.sum(1) != 0)[0][-1]
                bbox = [x1, y1, x2, y2]
            return bbox

        # prediction COCO
        result = COCO(result_file)

        # ground truth selection
        gt_dict = {
            "vg1.2": "data/vg/controlcap/vg1.2/test.json",
            "vg1.0": "data/vg/controlcap/vg1.0/test.json",
            "vgcoco": "data/vg/controlcap/vgcoco/test.json"
        }
        gt_file = gt_file_override or gt_dict.get(self.eval_dataset_name, None)
        if gt_file is None or not os.path.exists(gt_file):
            raise ValueError(f"Ground Truth file for [{self.eval_dataset_name}] not found (got {gt_file}).")
        gt = COCO(gt_file)

        empty_pred_num = 0
        ev = DenseCapEvaluator()
        recs = []
        for image_id, _ in tqdm.tqdm(list(gt.imgs.items())):
            anns = gt.imgToAnns[image_id]
            rec = dict()
            target_boxes = []
            target_text = []
            for ann in anns:
                box = seg2bbox(ann['segmentation'])
                target_boxes.append(box)
                target_text.append(ann['caption'])
            rec['target_boxes'] = target_boxes
            rec['target_text'] = target_text

            preds = result.imgToAnns.get(image_id, [])
            if len(preds) == 0:
                empty_pred_num += 1
                continue
            scores, boxes, text = [], [], []
            for pred in preds:
                box = seg2bbox(pred['segmentation'])
                pred_result = pred['extra_info'].get('pred_result', None)
                if pred_result is None:
                    continue
                score = pred_result.get('score', 1)
                caption = pred_result.get('caption', "")
                scores.append(score)
                boxes.append(box)
                text.append(caption)

            if len(boxes) == 0:
                empty_pred_num += 1
                continue

            rec['scores'] = scores
            rec['boxes'] = boxes
            rec['text'] = text
            rec['img_info'] = image_id
            recs.append(rec)

        for rec in tqdm.tqdm(recs):
            try:
                ev.add_result(
                    scores=torch.tensor(rec['scores']),
                    boxes=torch.tensor(rec['boxes']),
                    text=rec['text'],
                    target_boxes=torch.tensor(rec['target_boxes']),
                    target_text=rec['target_text'],
                    img_info=rec['img_info'],
                )
            except Exception as e:
                print(f"sample error: {e}")

        if empty_pred_num != 0:
            logging.info(f":Image numbers with empty prediction ({empty_pred_num}).")

        metrics = ev.evaluate()
        logging.info(f":Metrics ({str(metrics)}).")
        metrics["agg_metrics"] = metrics["map"]
        return metrics

    @dist_utils.main_process
    def report_metrics_reg(self, result_file):
        logging.info(f":Begin evaluation ({result_file}).")
        sys.stdout = None

        # prediction
        result = COCO(result_file)
        for id, ann in result.anns.items():
            pred_result = ann["extra_info"].get("pred_result", None)
            if pred_result is None:
                raise ValueError(f"Pred result for [{self.eval_dataset_name}] is not found")
            ann['caption'] = pred_result["caption"]

        # ground truth
        gt_dict = {
            "vg_reg": "data/vg/controlcap/vg_reg/test.json",
            "refcocog": "data/refcoco/controlcap/refcocog_val.json"
        }
        gt_file = gt_dict.get(self.eval_dataset_name, None)
        if gt_file is None:
            raise ValueError(f"Ground Truth file for [{self.eval_dataset_name}] is not found")
        gt = COCO(gt_file)

        # Evaluate
        coco_eval = COCOEvalCap(gt, result)
        coco_eval.params['image_id'] = result.getImgIds()
        coco_eval.evaluate()

        metrics = copy.deepcopy(coco_eval.eval)
        metrics["METEOR"] = metrics["METEOR"] * 100
        metrics["CIDEr"] = metrics["CIDEr"] * 100

        sys.stdout = sys.__stdout__
        logging.info(f":Metrics ({str(metrics)}).")
        metrics["agg_metrics"] = metrics["METEOR"]
        return metrics

    @main_process
    def log_stats(self, stats, split_name='train'):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass
