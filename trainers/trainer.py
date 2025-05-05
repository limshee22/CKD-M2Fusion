import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# 모듈 임포트
from modules.utils import Utils


class Trainer:
    """모델 학습 클래스"""
    def __init__(
        self, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: Any, 
        device: torch.device, 
        logger: logging.Logger
    ):
        """
        초기화 메서드
        
        Args:
            model: 학습할 모델
            criterion: 손실 함수
            optimizer: 최적화기
            scheduler: 학습률 스케줄러
            device: 학습 디바이스
            logger: 로거
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, log_interval: int = 10) -> float:
        """
        한 에포크 동안 모델 학습 메서드
        
        Args:
            train_loader: 학습 데이터 로더
            epoch: 현재 에포크
            log_interval: 로깅 간격
            
        Returns:
            평균 학습 손실
        """
        self.model.train()
        running_loss = 0.0
        total_loss = 0.0
        start_time = time.time()
        patient_count = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for batch_idx, batch in progress_bar:
            images = batch['image'].to(self.device)
            lab_values = batch['lab_values'].to(self.device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx'].to(self.device)
            
            baseline_Cr_log = lab_values[:, 0].view(-1, 1)
            
            outputs = self.model(images, lab_values, patient_ids, slice_indices)
            loss = self.criterion(outputs, baseline_Cr_log[0].unsqueeze(0))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            total_loss += loss.item()
            patient_count += 1
            
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed_time = time.time() - start_time
                progress_bar.set_description(
                    f'Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | '
                    f'Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s'
                )
                running_loss = 0.0
        
        return total_loss / patient_count
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        검증 세트에서 모델 평가 메서드
        
        Args:
            val_loader: 검증 데이터 로더
            
        Returns:
            검증 지표 딕셔너리
        """
        self.model.eval()
        total_loss = 0.0
        patient_count = 0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="검증 중"):
                images = batch['image'].to(self.device)
                lab_values = batch['lab_values'].to(self.device)
                patient_ids = batch['patient_id']
                slice_indices = batch['slice_idx'].to(self.device)
                
                baseline_Cr_log = lab_values[:, 0].view(-1, 1)
                
                outputs = self.model(images, lab_values, patient_ids, slice_indices)
                loss = self.criterion(outputs, baseline_Cr_log[0].unsqueeze(0))
                
                total_loss += loss.item()
                patient_count += 1
                
                all_targets.append(np.exp(baseline_Cr_log[0].item()))
                all_preds.append(np.exp(outputs.item()))
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        return {
            'loss': total_loss / patient_count if patient_count > 0 else float('inf'),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        }

    def test(self, test_loader: DataLoader) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        테스트 세트에서 모델 평가 메서드
        
        Args:
            test_loader: 테스트 데이터 로더
            
        Returns:
            metrics: 테스트 지표 딕셔너리
            results_df: 결과 데이터프레임
        """
        self.model.eval()
        
        patient_predictions = {}
        patient_targets = {}
        patient_lab_values = {}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="테스트 중"):
                images = batch['image'].to(self.device)
                lab_values = batch['lab_values'].to(self.device)
                patient_ids = batch['patient_id']
                slice_indices = batch['slice_idx'].to(self.device)
                
                baseline_Cr_log = lab_values[:, 0].view(-1, 1)
                
                outputs = self.model(images, lab_values, patient_ids, slice_indices)
                
                patient_id = patient_ids[0]
                
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = []
                    patient_targets[patient_id] = baseline_Cr_log[0].item()
                    patient_lab_values[patient_id] = lab_values[0].cpu().numpy()
                
                patient_predictions[patient_id].append(outputs.item())
        
        final_predictions = {}
        for patient_id, preds in patient_predictions.items():
            final_predictions[patient_id] = np.mean(preds)
        
        final_predictions = {pid: np.exp(pred) for pid, pred in final_predictions.items()}
        patient_targets = {pid: np.exp(target) for pid, target in patient_targets.items()}

        all_targets_original = np.array([target for target in patient_targets.values()])
        all_preds_original = np.array([pred for pred in final_predictions.values()])

        mse = mean_squared_error(all_targets_original, all_preds_original)
        mae = mean_absolute_error(all_targets_original, all_preds_original)
        r2 = r2_score(all_targets_original, all_preds_original) 
        
        results_df = pd.DataFrame({
            'patient_id': list(final_predictions.keys()),
            'true_baseline_Cr': [patient_targets[pid] for pid in final_predictions.keys()],
            'predicted_baseline_Cr': [final_predictions[pid] for pid in final_predictions.keys()]
        })
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        }
        
        return metrics, results_df
    
    def fit(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int, 
        early_stopping: int, 
        checkpoint_dir: str, 
        log_interval: int = 10
    ) -> int:
        """
        모델 학습 및 검증 수행 메서드
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            epochs: 학습 에포크 수
            early_stopping: 조기 종료 인내 값
            checkpoint_dir: 체크포인트 저장 디렉토리
            log_interval: 로깅 간격
            
        Returns:
            best_epoch: 최고 성능 에포크
        """
        for epoch in range(1, epochs + 1):
            # 학습
            train_loss = self.train_epoch(train_loader, epoch, log_interval)
            
            # 검증
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            
            # 학습률 업데이트
            self.scheduler.step()
            
            # 지표 기록
            self.logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Val R²: {val_metrics['r2']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # 최고 모델 확인
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # 체크포인트 저장
                Utils.save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'best_val_loss': self.best_val_loss,
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    },
                    is_best,
                    checkpoint_dir,
                    filename=f'checkpoint_epoch{epoch}.pth.tar'
                )
            else:
                self.patience_counter += 1
            
            # 조기 종료
            if self.patience_counter >= early_stopping:
                self.logger.info(f"{epoch}번째 에포크 후 조기 종료")
                break
        
        return self.best_epoch