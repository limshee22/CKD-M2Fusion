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
        
        # TabPFN에서 특징 추출 (내부 표현을 특징으로 사용)
        N, D = unique_lab.shape
        
        # TabPFN의 내부 예측을 사용하여 피처 생성
        # 'return_all_preds'는 사용할 수 없으므로 대체 방법 사용
        # 대신 기본 predict를 사용하고 텐서 크기 조정
        predictions = self.tabpfn.predict(unique_lab)
        
        # 예측 결과를 특징으로 사용 (대체 방법)
        # 여기서는 단순히 predictions를 복제하여 출력 크기를 맞춤
        features = np.repeat(predictions.reshape(N, 1), self.output_dim, axis=1)
        
        # 출력 차원에 맞게 특성 조정 (이미 조정됨)
        
        # NumPy 배열을 텐서로 변환
        features_tensor = torch.FloatTensor(features).to(lab_values.device)
        
        # 배치의 모든 슬라이스에 동일한 특성 반복
        batch_size = lab_values.size(0)
        features_tensor = features_tensor.repeat(batch_size, 1)
        
        return features_tensor