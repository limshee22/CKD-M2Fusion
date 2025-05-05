import numpy as np
import torch
from typing import Tuple, List, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm


class DataProcessor:
    """데이터 처리 클래스"""
    
    @staticmethod
    def collect_tabpfn_training_data(train_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        TabPFN 학습을 위한 데이터 수집 메서드
        
        Args:
            train_loader: 학습 데이터 로더
            
        Returns:
            X_train: 수집된 특징 데이터
            y_train: 수집된 타겟 데이터
        """
        X_train = []
        y_train = []
        
        print("TabPFN 학습을 위한 고유한 환자 데이터 수집 중...")
        for batch in tqdm(train_loader, desc="수집 중"):
            lab_values = batch['lab_values'].cpu().numpy()
            
            # 배치의 첫 번째 행이 한 환자의 데이터 (동일한 환자의 여러 슬라이스가 배치에 있음)
            unique_lab = lab_values[0:1]
            
            # 대상 값과 특징 분리
            # baseline_Cr_log를 대상으로 설정 (인덱스 0)
            features = np.column_stack([
                unique_lab[:, 1],  # cr_48h
                unique_lab[:, 2],  # age
                unique_lab[:, 3],  # sex
                unique_lab[:, 4],  # MeshVolume
                unique_lab[:, 5],  # Sphericity
                unique_lab[:, 6],  # Thickness_Mean
                unique_lab[:, 7],  # BUN
                unique_lab[:, 8],  # Albumin
                unique_lab[:, 9],  # HTN
            ])
            
            targets = unique_lab[:, 0]  # baseline_Cr_log
            
            X_train.append(features)
            y_train.append(targets)
        
        # 모든 배치 결합
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        return X_train, y_train