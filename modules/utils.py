import os
import logging
import time
import random
import numpy as np
import torch
from types import SimpleNamespace
from typing import Dict, Any, Optional
from DINOv2ForRadiology.dinov2.train.ssl_meta_arch import SSLMetaArch


class Utils:
    """유틸리티 클래스"""
    
    @staticmethod
    def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str, filename: str = 'checkpoint.pth.tar') -> None:
        """
        모델 체크포인트 저장 메서드
        
        Args:
            state: 저장할 상태 딕셔너리
            is_best: 최고 모델 여부
            checkpoint_dir: 체크포인트 저장 디렉토리
            filename: 파일 이름
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(state, filepath)
        
        if is_best:
            best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
            import shutil
            shutil.copyfile(filepath, best_filepath)
    
    @staticmethod
    def setup_logging(log_dir: str) -> logging.Logger:
        """
        로깅 설정 메서드
        
        Args:
            log_dir: 로그 저장 디렉토리
            
        Returns:
            logger: 설정된 로거
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d-%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    @staticmethod
    def set_seed(seed: int) -> None:
        """
        재현성을 위한 시드 설정 메서드
        
        Args:
            seed: 랜덤 시드 값
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def setup_fusion_model(args: Any, pretrained_checkpoint: Optional[Dict[str, torch.Tensor]] = None, out_dim: int = 1) -> tuple:
        """
        융합 모델 설정 메서드
        
        Args:
            args: 명령줄 인수
            pretrained_checkpoint: 사전 학습된 체크포인트
            out_dim: 출력 차원
            
        Returns:
            model: 설정된 모델
            cfg: 모델 설정
            device: 사용 디바이스
        """
        # DINOv2 config
        cfg_dict = {
            "compute_precision": {"grad_scaler": False, "teacher": {}, "student": {}},
            "student": {
                "pretrained_weights": None,
                "arch": "vit_base",
                "patch_size": 16,
                "ffn_bias": False,
                "ffn_layer": "mlp",
                "layerscale": 1.0,
                "proj_bias": True,
                "drop_path_rate": 0.1,
                "drop_path_uniform": False,
                "block_chunks": 0,
                "qkv_bias": True,
            },
            "dino": {
                "loss_weight": 1.0,
                "koleo_loss_weight": 0.0,
                "head_n_prototypes": 65536,
                "head_bottleneck_dim": 256,
                "head_hidden_dim": 2048,
                "head_nlayers": 3,
            },
            "ibot": {
                "loss_weight": 0.0,
                "mask_ratio_min_max": (0.0, 0.0),
                "mask_sample_probability": 0.0,
                "separate_head": False,
                "head_n_prototypes": 0,
                "head_hidden_dim": 0,
                "head_bottleneck_dim": 0,
                "head_nlayers": 0,
            },
            "crops": {"local_crops_number": 0, "global_crops_size": 224},
            "optim": {"layerwise_decay": 1.0, "patch_embed_lr_mult": 1.0},
            "train": {"centering": "sinkhorn_knopp"},
        }

        # Config를 SimpleNamespace로 변환
        cfg = SimpleNamespace(**cfg_dict)
        cfg.compute_precision = SimpleNamespace(**cfg_dict["compute_precision"])
        cfg.student = SimpleNamespace(**cfg_dict["student"])
        cfg.dino = SimpleNamespace(**cfg_dict["dino"])
        cfg.ibot = SimpleNamespace(**cfg_dict["ibot"])
        cfg.crops = SimpleNamespace(**cfg_dict["crops"])
        cfg.optim = SimpleNamespace(**cfg_dict["optim"])
        cfg.train = SimpleNamespace(**cfg_dict["train"])

        print(f"DINOv2 모델 아키텍처: {cfg.student.arch}")

        # GPU 사용 가능 여부 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 디바이스: {device}")

        # 기본 모델 생성
        base_model = SSLMetaArch(cfg).to(device)

        # 사전 학습된 체크포인트 로드 (제공된 경우)
        if pretrained_checkpoint:
            base_model.student.backbone.load_state_dict(pretrained_checkpoint)
            print("Pretrained weights를 성공적으로 로드했습니다.")

        # FusionModel 모듈 직접 임포트
        from .models import FusionModel

        # 융합 모델 생성
        model = FusionModel(
            base_model=base_model,
            feature_dim=768,  # ViT-Base 출력 차원
            tabpfn_dim=args.tabpfn_dim if hasattr(args, 'tabpfn_dim') else 64,
            num_slices=args.batch_size,  # 배치 크기 (환자당 슬라이스 수)
            hidden_dim=1024,
            out_dim=out_dim
        ).to(device)

        return model, cfg, device