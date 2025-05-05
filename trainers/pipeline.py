import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Tuple, Any, Optional

# 모듈 임포트
from modules.utils import Utils
from modules.losses import RMSELoss
from modules.data_processor import DataProcessor
from trainers.trainer import Trainer


class TrainingPipeline:
    """전체 학습 파이프라인 클래스"""
    def __init__(self, args: Any):
        """
        초기화 메서드
        
        Args:
            args: 명령줄 인수
        """
        self.args = args
        Utils.set_seed(args.seed)
        self.logger = Utils.setup_logging(args.log_dir)
        self.logger.info(f"Args: {args}")
        
        # 디렉토리 생성
        for directory in [args.checkpoint_dir, args.log_dir, args.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_data(self):
        """
        데이터 로더 생성 메서드
        
        Returns:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            test_inha_loader: INHA 테스트 데이터 로더
            test_kmuh_loader: KMUH 테스트 데이터 로더
        """
        self.logger.info("데이터 로더 생성 중...")
        from dataloaders import create_four_dataloaders
        
        train_loader, val_loader, test_inha_loader, test_kmuh_loader = create_four_dataloaders(
            inha_csv_path=self.args.train_csv,
            kmuh_csv_path=self.args.test_csv,
            batch_size=self.args.batch_size,
            resize_dim=224,
            target_slices=self.args.target_slices
        )
        return train_loader, val_loader, test_inha_loader, test_kmuh_loader
    
    def load_checkpoint(self):
        """
        사전 학습된 체크포인트 로드 메서드
        
        Returns:
            filtered_checkpoint: 필터링된 체크포인트
        """
        self.logger.info(f"{self.args.pretrained_weights}에서 사전 학습된 가중치 로드 중...")
        checkpoint = torch.load(self.args.pretrained_weights, map_location=self.device)
        
        # 백본 가중치만 필터링
        filtered_checkpoint = {
            k.replace("student.backbone.", ""): v
            for k, v in checkpoint["model"].items()
            if k.startswith("student.backbone.")
        }
        return filtered_checkpoint
    
    def build_model(self, filtered_checkpoint):
        """
        모델 구축 메서드
        
        Args:
            filtered_checkpoint: 필터링된 체크포인트
            
        Returns:
            model: 구축된 모델
            cfg: 모델 설정
        """
        self.logger.info("융합 모델 설정 중...")
        model, cfg, _ = Utils.setup_fusion_model(
            args=self.args, 
            pretrained_checkpoint=filtered_checkpoint, 
            out_dim=1
            
        )
        return model, cfg
    
    def train_tabpfn(self, model, train_loader):
        """
        TabPFN 학습 메서드
        
        Args:
            model: 모델
            train_loader: 학습 데이터 로더
            
        Returns:
            model: TabPFN이 학습된 모델
        """
        self.logger.info("TabPFN 학습을 위한 데이터 수집 중...")
        X_train, y_train = DataProcessor.collect_tabpfn_training_data(train_loader)
        
        self.logger.info("TabPFN 모델 학습 중...")
        model.fit_tabpfn(X_train, y_train)
        return model
    
    def setup_training(self, model, train_loader_len):
        """
        학습 설정 메서드
        
        Args:
            model: 모델
            train_loader_len: 학습 데이터 로더 길이
            
        Returns:
            trainer: 학습기
        """
        criterion = RMSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, eps=1e-8)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=5e-4,
            total_steps=self.args.epochs*train_loader_len,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            logger=self.logger
        )
        
        return trainer
    
    def run(self):
        """
        전체 파이프라인 실행 메서드
        """
        # 데이터 로드
        self.train_loader, self.val_loader, self.test_inha_loader, self.test_kmuh_loader = self.load_data()
        
        # 체크포인트 로드
        filtered_checkpoint = self.load_checkpoint()
        
        # 모델 구축
        model, _ = self.build_model(filtered_checkpoint)
        
        # TabPFN 학습
        model = self.train_tabpfn(model, self.train_loader)
        
        # 학습 설정
        trainer = self.setup_training(model, len(self.train_loader))
        
        # 학습 및 검증
        self.logger.info("학습 시작...")
        best_epoch = trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.args.epochs,
            early_stopping=self.args.early_stopping,
            checkpoint_dir=self.args.checkpoint_dir,
            log_interval=self.args.log_interval
        )
        
        # 최고 모델 로드
        self.logger.info(f"{best_epoch}번째 에포크에서 최고 모델 로드 중...")
        checkpoint = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        
        # INHA 테스트
        self.logger.info("INHA 데이터에서 모델 테스트 중...")
        test_metrics_inha, results_df_inha = trainer.test(self.test_inha_loader)
        
        self.logger.info(
            f"INHA 테스트 결과 | "
            f"MSE: {test_metrics_inha['mse']:.4f} | "
            f"RMSE: {test_metrics_inha['rmse']:.4f} | "
            f"MAE: {test_metrics_inha['mae']:.4f} | "
            f"R²: {test_metrics_inha['r2']:.4f}"
        )
        
        results_path_inha = os.path.join(self.args.results_dir, f'test_results_Inha_{time.strftime("%Y%m%d-%H%M%S")}.csv')
        results_df_inha.to_csv(results_path_inha, index=False)
        self.logger.info(f"테스트 결과 저장됨: {results_path_inha}")
        
        # KMUH 테스트
        self.logger.info("KMUH 데이터에서 모델 테스트 중...")
        test_metrics_kmuh, results_df_kmuh = trainer.test(self.test_kmuh_loader)
        
        self.logger.info(
            f"KMUH 테스트 결과 | "
            f"MSE: {test_metrics_kmuh['mse']:.4f} | "
            f"RMSE: {test_metrics_kmuh['rmse']:.4f} | "
            f"MAE: {test_metrics_kmuh['mae']:.4f} | "
            f"R²: {test_metrics_kmuh['r2']:.4f}"
        )
        
        results_path_kmuh = os.path.join(self.args.results_dir, f'test_results_Kmuh_{time.strftime("%Y%m%d-%H%M%S")}.csv')
        results_df_kmuh.to_csv(results_path_kmuh, index=False)
        self.logger.info(f"테스트 결과 저장됨: {results_path_kmuh}")