import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional, Union, Any
import numpy as np


class RegressionHead(nn.Module):
    """회귀 헤드 모듈"""
    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 1, dropout_rate: float = 0.3):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.Tanh()
        )
        
        # 가중치 초기화 개선
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화 메서드"""
        for i in range(0, len(self.regression_head) - 1, 3):  # Linear 레이어마다
            if isinstance(self.regression_head[i], nn.Linear):
                nn.init.xavier_normal_(self.regression_head[i].weight, gain=0.01)
                nn.init.zeros_(self.regression_head[i].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 메서드
        
        Args:
            x: 입력 텐서
            
        Returns:
            회귀 출력 텐서
        """
        output = self.regression_head(x)
        # [-1, 1] -> [-0.7, 1.2] 범위로 제한
        output = torch.clamp(output, -0.7, 1.2)
        return output


class FusionModel(nn.Module):
    """
    DINOv2와 TabPFN 특징을 융합하는 모델
    """
    def __init__(self, base_model, feature_dim=768, tabpfn_dim=64, num_slices=16, hidden_dim=1024, out_dim=1):
        super(FusionModel, self).__init__()
        self.base_model = base_model  # DINOv2 backbone
        self.feature_dim = feature_dim
        self.num_slices = num_slices
        self.is_training = True  # DINOv2 학습 상태 플래그
        
        # TabPFN 특징 추출기
        from .feature_extractors import TabPFNFeatureExtractor
        self.tabpfn_extractor = TabPFNFeatureExtractor(output_dim=tabpfn_dim)
        
        # 이미지 특징 차원 축소를 위한 프로젝션 레이어
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU()
        )
        
        # 축소된 특징 차원
        reduced_feature_dim = feature_dim // 4
        
        # 융합된 특징을 처리하기 위한 회귀 헤드
        self.regression_head = RegressionHead(
            in_dim=(reduced_feature_dim * num_slices) + tabpfn_dim,  # 이미지 특징 + TabPFN 특징
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout_rate=0.5
        )

    # 수정된 train 메서드: mode 파라미터를 받도록 변경
    def train(self, mode: bool = True):
        """DINOv2( base_model )는 is_train 플래그만,
        나머지 서브모듈은 표준 Module.train(mode) 사용"""
        self.is_training = mode                # 우리 자체 플래그
        self.base_model.is_train = mode        # ← DINOv2 방식

        # 나머지 서브모듈만 정상적으로 recursion
        self.feature_projection.train(mode)
        self.regression_head.train(mode)
        self.tabpfn_extractor.train(mode)      # 특별히 학습 안 하지만 일관성 유지
        return self

    def eval(self):
        """eval() 역시 base_model 은 is_train 플래그만 조정"""
        return self.train(False)
    
    def fit_tabpfn(self, X_train, y_train):
        """TabPFN 특징 추출기 학습"""
        self.tabpfn_extractor.fit(X_train, y_train)
        
    def forward(self, images, lab_values=None, patient_ids=None, slice_indices=None):
        batch_size = images.size(0)
        features_list = []
        
        # 각 슬라이스 처리
        for i in range(batch_size):
            img = images[i].unsqueeze(0)  # [1, 3, 224, 224]
            
            # 모든 슬라이스를 동일하게 처리
            img_input = {"samples": img, "masks": None}
            with torch.no_grad():
                slice_features = self.base_model.student.backbone(img_input)
                if isinstance(slice_features, (list, tuple)):
                    slice_features = slice_features[0]  # [CLS] 토큰 가져오기
            
            # 특징 차원 축소 적용
            reduced_features = self.feature_projection(slice_features.squeeze(0))
            features_list.append(reduced_features)  # [feature_dim//4]
        
        # 모든 이미지 특징 연결 (concatenate)
        concatenated_img_features = torch.cat(features_list, dim=0).unsqueeze(0)  # [1, batch_size * (feature_dim//4)]
        
        # 실험실 값이 제공된 경우 TabPFN 특징 추출
        if lab_values is not None:
            # TabPFN 특징 추출
            tabpfn_features = self.tabpfn_extractor(lab_values)  # [batch_size, tabpfn_dim]
            
            # 첫 번째 행의 TabPFN 특징만 사용 (모든 슬라이스에서 동일한 환자)
            patient_tabpfn_features = tabpfn_features[0].unsqueeze(0)  # [1, tabpfn_dim]
            
            # 이미지 특징과 TabPFN 특징 융합
            combined_features = torch.cat([concatenated_img_features, patient_tabpfn_features], dim=1)
        else:
            combined_features = concatenated_img_features
        
        # 최종 회귀
        output = self.regression_head(combined_features)
        
        return output