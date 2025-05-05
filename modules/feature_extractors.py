import torch
import torch.nn as nn
import numpy as np
from tabpfn import TabPFNRegressor
from typing import Tuple, Optional


class TabPFNFeatureExtractor(nn.Module):
    """TabPFN을 사용하여 표 형식 데이터에서 특징을 추출하는 모듈"""
    def __init__(self, output_dim: int = 64):
        """
        초기화 메서드
        
        Args:
            output_dim: 출력 특징 차원
        """
        super().__init__()
        self.tabpfn = TabPFNRegressor()
        self.output_dim = output_dim
        self.is_trained = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        TabPFN 모델 학습 메서드
        
        Args:
            X_train: 학습 특징 데이터
            y_train: 학습 타겟 데이터
        """
        self.tabpfn.fit(X_train, y_train)
        self.is_trained = True
        print("TabPFN 모델 학습 완료!")
        
    def forward(self, lab_values: torch.Tensor) -> torch.Tensor:
        """
        lab_values에서 TabPFN 특징 추출 메서드
        
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
        
        # TabPFN에서 특징 추출 (내부 표현을 특징으로 사용)
        N, D = unique_lab.shape
        
        # TabPFN의 내부 예측 과정을 사용하여 특성 추출
        _, all_preds = self.tabpfn.predict(unique_lab, return_all_preds=True, return_features=True)
        
        # 앙상블 예측과 모든 다른 예측을 결합하여 풍부한 특성 표현 생성
        features = all_preds.reshape(N, -1)
        
        # 출력 차원에 맞게 특성 조정
        if features.shape[1] >= self.output_dim:
            features = features[:, :self.output_dim]
        else:
            # 특성 차원이 부족한 경우 패딩
            pad_width = ((0, 0), (0, self.output_dim - features.shape[1]))
            features = np.pad(features, pad_width, mode='constant')
        
        # NumPy 배열을 텐서로 변환
        features_tensor = torch.FloatTensor(features).to(lab_values.device)
        
        # 배치의 모든 슬라이스에 동일한 특성 반복
        batch_size = lab_values.size(0)
        features_tensor = features_tensor.repeat(batch_size, 1)
        
        return features_tensor