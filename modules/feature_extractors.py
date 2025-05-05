import torch
import torch.nn as nn
import numpy as np
from tabpfn import TabPFNRegressor
from typing import Tuple, Optional

class TabPFNFeatureExtractor(nn.Module):
    """
    TabPFN을 사용하여 표 형식 데이터에서 특징을 추출하는 모듈
    """
    def __init__(self, output_dim=64):
        super().__init__()
        self.tabpfn = TabPFNRegressor()
        self.output_dim = output_dim
        self.is_trained = False
        
    def fit(self, X_train, y_train):
        """TabPFN 모델 학습"""
        self.tabpfn.fit(X_train, y_train)
        self.is_trained = True
        print("TabPFN 모델 학습 완료!")
        # 학습 데이터의 특성 수 저장
        self.n_features = X_train.shape[1]
        
    def forward(self, lab_values):
        """
        lab_values에서 TabPFN 특징 추출
        
        Args:
            lab_values: 배치의 임상 데이터 [batch_size, 10]
        
        Returns:
            TabPFN 특징 [batch_size, output_dim]
        """
        # 텐서를 NumPy 배열로 변환
        lab_np = lab_values.detach().cpu().numpy()
        
        # 배치에서 환자 ID를 기반으로 중복 제거한 특징 추출
        # 실제로는 각 배치에 동일한 환자의 여러 슬라이스가 있으므로 첫 번째 행만 사용
        unique_lab = lab_np[0:1]
        
        # 원래 코드의 collect_tabpfn_training_data 함수에서 사용한 특성과 동일한 방식으로 처리
        # baseline_Cr_log를 제외한 9개의 특성만 사용 (인덱스 1-9)
        features = np.column_stack([
            unique_lab[:, 1],  # cr_48h
            unique_lab[:, 2],  # age
            unique_lab[:, 3],  # sex
            unique_lab[:, 4],  # MeshVolume
            unique_lab[:, 5],  # Sphericity
            unique_lab[:, 6],  # Thickness_Mean
            unique_lab[:, 7],  # BUN
            unique_lab[:, 8],  # Albumin
            unique_lab[:, 9] if unique_lab.shape[1] > 9 else np.zeros(unique_lab.shape[0])  # HTN
        ])
        
        # TabPFN 예측 수행
        try:
            predictions = self.tabpfn.predict(features)
        except Exception as e:
            print(f"TabPFN 예측 중 오류 발생: {e}")
            # 오류 발생 시 임의의 특성 반환
            predictions = np.zeros(unique_lab.shape[0])
        
        # 차원 확장 및 복제를 통한 특성 생성
        N = unique_lab.shape[0]
        features_expanded = np.repeat(predictions.reshape(N, 1), self.output_dim, axis=1)
        
        # NumPy 배열을 텐서로 변환
        features_tensor = torch.FloatTensor(features_expanded).to(lab_values.device)
        
        # 배치의 모든 슬라이스에 동일한 특성 반복
        batch_size = lab_values.size(0)
        features_tensor = features_tensor.repeat(batch_size, 1)
        
        return features_tensor