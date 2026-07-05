# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from torchmetrics import MeanMetric, MetricCollection
from .models.infonce_loss import multi_positive_infonce_loss

from . import logger
from .models import get_model


class AverageKeyMeter(MeanMetric):
    def __init__(self, key, *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, dict):
        value = dict[self.key]
        value = value[torch.isfinite(value)]
        return super().update(value)


class GenericModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        name = cfg.model.get("name")
        name = "orienternet" if name in ("localizer_bev_depth", None) else name
        self.model = get_model(name)(cfg.model)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.metrics_val = MetricCollection(self.model.metrics(), prefix="val/")
        self.losses_val = None  
        self._losses_val_keys_known = False # we do not know the loss keys in advance

    def _monitor_embeddings(self, batch, pred, batch_idx):
        """Monitor embeddings during training."""
        try:
            with torch.no_grad():
                # Try to extract embeddings from different possible sources
                embeddings = self._extract_embeddings(batch, pred)
                
                if embeddings is not None:
                    step_name = f"epoch_{self.current_epoch}_batch_{batch_idx}"
                    diagnosis = EmbeddingDiagnostics.diagnose_embeddings(
                        embeddings, 
                        step_name=step_name,
                        verbose=True
                    )
                    self.last_embedding_diagnosis = diagnosis
                    
                    # Log key metrics to tensorboard
                    if diagnosis:
                        self.log("debug/embedding_std", diagnosis['stats']['std'])
                        self.log("debug/embedding_max_diff", diagnosis['stats']['max_diff'])
                        self.log("debug/embedding_collapsed", float(diagnosis['is_collapsed']))
                        if diagnosis['batch_size'] > 1:
                            self.log("debug/embedding_sim_std", diagnosis['similarity']['off_diag_std'])
        except Exception as e:
            print(f"⚠️  Error monitoring embeddings: {e}")

    def _extract_embeddings(self, batch, pred):
        """Extract embeddings from the model - customize this for your model."""
        try:
            # Try to get global descriptors if available
            if hasattr(self.model.image_encoder, 'global_descriptor_head'):
                features = self.model.image_encoder(batch)
                global_desc = features.get("global_descriptor")
                if isinstance(global_desc, torch.Tensor) and global_desc.dim() == 2 and global_desc is not None:
                    return global_desc
                else:
                    print(f"global_desc dimension: {global_desc.dim()}")
                    print(f"global_desc is not None:{global_desc is not None}")
                    return global_desc
                
            
            return None
            
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            return None

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        losses = pred['losses']
        '''
        # EMBEDDING MONITORING - Add this section
        if batch_idx % self.cfg.model.embedding_monitor_interval == 0:
            self._monitor_embeddings(batch, pred, batch_idx)
        '''
        self.log_dict(
            {f"loss/{k}/train": v.mean() for k, v in losses.items()},
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True, # Set to True to allow sync of losses between GPUs
        )
        return losses["total"].mean()

    def on_train_epoch_end(self):
      """Record lr after each train epoch"""
      current_lr = self.optimizers().param_groups[0]['lr']
      self.log('lr/train', current_lr)
      
      # Learning Rate too low
      if current_lr < 1e-6:
          print(f"⚠️ Learning rate is very low: {current_lr:.2e}")
          print("Consider early stopping or adjusting min_lr")


    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        losses = pred['losses']
        if self.losses_val is None:
            self.losses_val = MetricCollection(
                {k: AverageKeyMeter(k).to(self.device) for k in losses},
                prefix="loss/",
                postfix="/val",
            )
        self.metrics_val(pred, batch)
        self.log_dict(self.metrics_val, sync_dist=True)
        self.losses_val.update(losses)
        self.log_dict(self.losses_val, sync_dist=True)

    def validation_epoch_start(self, batch):
        self.losses_val = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.get('weight_decay', 1e-5))   
        ret = {"optimizer": optimizer}
        cfg_scheduler = self.cfg.training.get("lr_scheduler")
        if cfg_scheduler is not None:
            
            scheduler_class = getattr(torch.optim.lr_scheduler, cfg_scheduler.name)
            
            if cfg_scheduler.name == "CosineAnnealingLR":
                scheduler_kwargs = {
                    "T_max" : cfg_scheduler.get("T_max",10),
                    "eta_min": cfg_scheduler.get("eta_min",0)
                }

            elif cfg_scheduler.name == "StepLR":
                scheduler_kwargs = {
                    'step_size': cfg_scheduler.get('step_size', 50),
                    'gamma': cfg_scheduler.get('gamma', 0.1)
                }
            elif cfg_scheduler.name == "ExponentialLR":
                scheduler_kwargs = {
                    'gamma': cfg_scheduler.get('gamma', 0.95)
                }

            elif cfg_scheduler.name == "ReduceLROnPlateau":
                    # ReduceLROnPlateau
                    scheduler_kwargs = {
                        'mode': cfg_scheduler.get('mode', 'min'),
                        'factor': cfg_scheduler.get('factor', 0.7),
                        'patience': cfg_scheduler.get('patience', 2),
                        'threshold': cfg_scheduler.get('threshold', 0.01),
                        'threshold_mode': cfg_scheduler.get('threshold_mode', 'rel'),
                        'cooldown': cfg_scheduler.get('cooldown', 0),
                        'min_lr': cfg_scheduler.get('min_lr', 5e-7),
                        'eps': cfg_scheduler.get('eps', 1e-8),
                    }
                    
                    lr_scheduler_config = {
                        "scheduler": scheduler_class(optimizer=optimizer, **scheduler_kwargs),
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": cfg_scheduler.get("monitor", "loss/total/val"),  # 🔧 监控的指标
                        "strict": cfg_scheduler.get("strict", True),
                        "name": "learning_rate",
                    }

            elif cfg_scheduler.name == "CosineAnnealingWarmRestarts":
                # CosineAnnealingWarmRestarts
                scheduler_kwargs = {
                    'T_0': cfg_scheduler.get('T_0', 10),
                    'T_mult': cfg_scheduler.get('T_mult', 2),
                    'eta_min': cfg_scheduler.get('eta_min', 1e-6),
                }
                lr_scheduler_config = {
                    "scheduler": scheduler_class(optimizer=optimizer, **scheduler_kwargs),
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "learning_rate",
                }

            else:
                excluded_keys={"name", "max_epochs", "warmup_steps"}
                scheduler_kwargs = {
                    k: v for k, v in cfg_scheduler.items() 
                    if k not in excluded_keys
                }
            
            scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)

            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss/total/val",
                "strict": True,
                "name": "learning_rate",
            }
        return ret

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        cfg=None,
        find_best=False,
    ):
        assert hparams_file is None, "hparams are not supported."

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or (lambda storage, loc: storage), weights_only=False
        )
        if find_best:
            best_score, best_name = None, None
            modes = {"min": torch.lt, "max": torch.gt}
            for key, state in checkpoint["callbacks"].items():
                if not key.startswith("ModelCheckpoint"):
                    continue
                mode = eval(key.replace("ModelCheckpoint", ""))["mode"]
                if best_score is None or modes[mode](
                    state["best_model_score"], best_score
                ):
                    best_score = state["best_model_score"]
                    best_name = Path(state["best_model_path"]).name
            logger.info("Loading best checkpoint %s", best_name)
            if best_name != checkpoint_path:
                return cls.load_from_checkpoint(
                    Path(checkpoint_path).parent / best_name,
                    map_location,
                    hparams_file,
                    strict,
                    cfg,
                    find_best=False,
                )

        logger.info(
            "Using checkpoint %s from epoch %d and step %d.",
            checkpoint_path.name,
            checkpoint["epoch"],
            checkpoint["global_step"],
        )
        cfg_ckpt = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
        if list(cfg_ckpt.keys()) == ["cfg"]:  # backward compatibility
            cfg_ckpt = cfg_ckpt["cfg"]
        cfg_ckpt = OmegaConf.create(cfg_ckpt)

        if cfg is None:
            cfg = {}
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        with open_dict(cfg_ckpt):
            cfg = OmegaConf.merge(cfg_ckpt, cfg)

        return pl.core.saving._load_state(cls, checkpoint, strict=strict, cfg=cfg)

class EmbeddingDiagnostics:
    """Monitor and diagnose embedding collapse during training."""
    
    @staticmethod
    def diagnose_embeddings(embeddings, step_name="", verbose=True):
        """
        Diagnose if embeddings are collapsing.
        
        Args:
            embeddings: torch.Tensor of shape [batch_size, embed_dim]
            step_name: str, name of the training step
            verbose: bool, whether to print detailed info
        """
        if embeddings is None:
            if verbose:
                print(f"⚠️  No embeddings provided for {step_name}")
            return None
            
        if embeddings.dim() != 2:
            if verbose:
                print(f"⚠️  Expected 2D embeddings, got {embeddings.shape}")
            return None
        
        batch_size, embed_dim = embeddings.shape
        
        # Basic statistics
        mean_val = embeddings.mean().item()
        std_val = embeddings.std().item()
        min_val = embeddings.min().item()
        max_val = embeddings.max().item()
        
        # Check for collapse: all embeddings are too similar
        first_embed = embeddings[0:1]  # [1, embed_dim]
        differences = torch.norm(embeddings - first_embed, dim=1)  # [batch_size]
        max_diff = differences.max().item()
        mean_diff = differences.mean().item()
        
        # Check for NaN or Inf
        has_nan = torch.isnan(embeddings).any().item()
        has_inf = torch.isinf(embeddings).any().item()
        
        # Similarity analysis (only if no NaN/Inf)
        off_diag_mean = off_diag_std = off_diag_max = 0.0
        if not (has_nan or has_inf) and batch_size > 1:
            try:
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
                similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
                
                # Off-diagonal elements (similarities between different samples)
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
                off_diagonal = similarity_matrix[mask]
                off_diag_mean = off_diagonal.mean().item()
                off_diag_std = off_diagonal.std().item()
                off_diag_max = off_diagonal.max().item()
            except:
                pass  # Skip similarity analysis if it fails
        
        # Determine if collapsed
        is_collapsed = (
            max_diff < 1e-6 or  # All embeddings are identical
            std_val < 1e-6 or   # No variation in embeddings
            (off_diag_std < 1e-3 and batch_size > 1)  # All similarities are too similar
        )
        
        diagnosis = {
            'step_name': step_name,
            'batch_size': batch_size,
            'embed_dim': embed_dim,
            'is_collapsed': is_collapsed,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'stats': {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
            },
            'similarity': {
                'off_diag_mean': off_diag_mean,
                'off_diag_std': off_diag_std,
                'off_diag_max': off_diag_max,
            }
        }
        
        if verbose:
            EmbeddingDiagnostics.print_diagnosis(diagnosis)
        
        return diagnosis
    
    @staticmethod
    def print_diagnosis(diagnosis):
        """Print detailed diagnosis."""
        step_name = diagnosis['step_name']
        print(f"\n🔍 EMBEDDING DIAGNOSIS [{step_name}]")
        print("=" * 50)
        
        # Basic info
        print(f"Shape: [{diagnosis['batch_size']}, {diagnosis['embed_dim']}]")
        
        # Health check
        if diagnosis['has_nan']:
            print("🚨 NaN detected in embeddings!")
        if diagnosis['has_inf']:
            print("🚨 Inf detected in embeddings!")
        if diagnosis['is_collapsed']:
            print("🚨 EMBEDDINGS COLLAPSED!")
        else:
            print("✅ Embeddings look healthy")
        
        # Statistics
        stats = diagnosis['stats']
        print(f"Mean: {stats['mean']:.6f}")
        print(f"Std:  {stats['std']:.6f}")
        print(f"Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"Max difference between embeddings: {stats['max_diff']:.6f}")
        
        # Similarity analysis
        sim = diagnosis['similarity']
        if diagnosis['batch_size'] > 1:
            print(f"Off-diagonal similarity mean: {sim['off_diag_mean']:.6f}")
            print(f"Off-diagonal similarity std:  {sim['off_diag_std']:.6f}")
            print(f"Max similarity: {sim['off_diag_max']:.6f}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if diagnosis['is_collapsed']:
            print("- Reduce learning rate")
            print("- Check temperature parameter in loss function")
            print("- Remove or reduce normalization")
            print("- Add gradient clipping")
            print("- Check for bugs in loss computation")
        elif stats['std'] > 10:
            print("- Embeddings have large variance, consider gradient clipping")


class LayerWiseOptimizer(GenericModule):
    def configure_optimizers(self):
        sam2_finetune_params = []
        adapter_params = []
        orienternet_head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # 跳过所有被冻结的参数
            
            if 'image_encoder.' in name:
                sam2_finetune_params.append(param)
            elif 'encoder_adaptation' in name:
                adapter_params.append(param)
            else:
                orienternet_head_params.append(param)
        
        main_lr = self.cfg.training.lr
        
        if self.cfg.training.finetune_from_checkpoint is not None:
            param_groups = [
            {'params': sam2_finetune_params, 'lr': main_lr / 100, 'name': 'sam2_backbone'},
            {'params': adapter_params, 'lr': main_lr, 'name': 'adapter'},
            {'params': orienternet_head_params, 'lr': main_lr / 10, 'name': 'orienternet_head'}
            ]
        else:
            param_groups = [
            {'params': sam2_finetune_params, 'lr': main_lr / 100, 'name': 'sam2_backbone'},
            {'params': adapter_params, 'lr': main_lr, 'name': 'adapter'},
            {'params': orienternet_head_params, 'lr': main_lr, 'name': 'orienternet_head'}
            ]
        for group in param_groups:
            print(f"  - Group: {group['name']}, Params: {len(group['params'])}, LR: {group['lr']:.1e}")

        optimizer = torch.optim.Adam(param_groups, lr=self.cfg.training.lr, weight_decay=self.cfg.training.get('weight_decay', 1e-5))   
        ret = {"optimizer": optimizer}
        cfg_scheduler = self.cfg.training.get("lr_scheduler")
        if cfg_scheduler is not None:
            
            scheduler_class = getattr(torch.optim.lr_scheduler, cfg_scheduler.name)
            
            if cfg_scheduler.name == "CosineAnnealingLR":
                scheduler_kwargs = {
                    "T_max" : cfg_scheduler.get("T_max",10),
                    "eta_min": cfg_scheduler.get("eta_min",0)
                }

            elif cfg_scheduler.name == "StepLR":
                scheduler_kwargs = {
                    'step_size': cfg_scheduler.get('step_size', 50),
                    'gamma': cfg_scheduler.get('gamma', 0.1)
                }
            elif cfg_scheduler.name == "ExponentialLR":
                scheduler_kwargs = {
                    'gamma': cfg_scheduler.get('gamma', 0.95)
                }
            else:
                excluded_keys={"name", "max_epochs", "warmup_steps"}
                scheduler_kwargs = {
                    k: v for k, v in cfg_scheduler.items() 
                    if k not in excluded_keys
                }
            
            scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)

            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss/total/val",
                "strict": True,
                "name": "learning_rate",
            }
        return ret