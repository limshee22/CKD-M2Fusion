import os
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import sys

# TabPFN imports
from tabpfn import TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor

# Import dataset and dataloaders
from M2Fusion_Loader import KidneySliceDataset, TestKidneySliceDataset, PatientBatchSampler, create_four_dataloaders


class RegressionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=1, dropout_rate=0.3):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, out_dim),
            # 출력값을 제한하기 위한 Tanh 추가
            nn.Tanh()
        )
        
        # 가중치 초기화 개선
        self._init_weights()
    
    def _init_weights(self):
        # 각 레이어에 작은 초기 가중치를 사용하여 안정성 개선
        for i in range(0, len(self.regression_head) - 1, 3):  # Linear 레이어마다
            if isinstance(self.regression_head[i], nn.Linear):
                nn.init.xavier_normal_(self.regression_head[i].weight, gain=0.01)
                nn.init.zeros_(self.regression_head[i].bias)
    
    def forward(self, x):
        # Tanh 출력을 log 스케일의 적절한 범위로 스케일링
        output = self.regression_head(x)
        # [-1, 1] -> [0, 1] -> [-2, 2] 범위로 변환
        output = torch.clamp(output, -0.7, 1.2)
        return output


class TabPFNFeatureExtractor(nn.Module):
    """
    TabPFNEmbedding을 사용하여 표 형식 데이터에서 특징을 추출하는 모듈
    학습 가능한 어텐션 기반 특징 집계 포함
    """
    def __init__(self, embedding_dim=192, aggregation_method='attention'):
        super().__init__()
        self.tabpfn = TabPFNRegressor(device='cuda')
        self.embedding_extractor = TabPFNEmbedding(tabpfn_reg=self.tabpfn, n_fold=0)
        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method
        self.is_fitted = False
        
        # 학습 데이터 저장
        self.X_train = None
        self.y_train = None
        
        # 학습 가능한 어텐션 메커니즘
        if aggregation_method == 'attention':
            self.attention_net = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 4),
                nn.ReLU(),
                nn.Linear(embedding_dim // 4, 1),
                nn.Softmax(dim=0)
            )
        elif aggregation_method == 'learnable_pooling':
            # 학습 가능한 가중치 파라미터
            self.feature_weights = nn.Parameter(torch.ones(23))  # 23개 특징에 대한 가중치
        elif aggregation_method == 'transformer':
            # Mini Transformer for feature aggregation
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
    def fit(self, X_train, y_train):
        """TabPFN 모델 학습 및 데이터 저장"""
        self.tabpfn.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.is_fitted = True
        print("TabPFN 모델 학습 완료!")
        print(f"학습 데이터 크기: X_train {X_train.shape}, y_train {y_train.shape}")
        
    def aggregate_embeddings(self, embeddings_2d):
        """
        다양한 방법으로 (23, 192) 임베딩을 (1, 192)로 집계
        
        Args:
            embeddings_2d: (num_features, embedding_dim) = (23, 192)
            
        Returns:
            pooled_embeddings: (1, embedding_dim) = (1, 192)
        """
        if self.aggregation_method == 'mean':
            # 기본 평균
            return np.mean(embeddings_2d, axis=0, keepdims=True)
            
        elif self.aggregation_method == 'attention':
            # 학습 가능한 어텐션 기반 집계
            embeddings_tensor = torch.FloatTensor(embeddings_2d).to(next(self.attention_net.parameters()).device)
            
            # 각 특징에 대한 어텐션 가중치 계산
            attention_weights = []
            for i in range(embeddings_tensor.size(0)):
                weight = self.attention_net(embeddings_tensor[i])
                attention_weights.append(weight)
            
            attention_weights = torch.cat(attention_weights, dim=0)  # (23, 1)
            attention_weights = torch.softmax(attention_weights.squeeze(), dim=0)  # (23,)
            
            # 가중 평균 계산
            weighted_embeddings = torch.sum(
                embeddings_tensor * attention_weights.unsqueeze(1), 
                dim=0, keepdim=True
            )
            
            return weighted_embeddings.detach().cpu().numpy()
            
        elif self.aggregation_method == 'learnable_pooling':
            # 학습 가능한 가중치 기반 집계
            embeddings_tensor = torch.FloatTensor(embeddings_2d).to(self.feature_weights.device)
            
            # 소프트맥스를 통한 가중치 정규화
            normalized_weights = torch.softmax(self.feature_weights, dim=0)
            
            # 가중 평균
            weighted_embeddings = torch.sum(
                embeddings_tensor * normalized_weights.unsqueeze(1),
                dim=0, keepdim=True
            )
            
            return weighted_embeddings.detach().cpu().numpy()
            
        elif self.aggregation_method == 'transformer':
            # 미니 트랜스포머 기반 집계
            embeddings_tensor = torch.FloatTensor(embeddings_2d).unsqueeze(0).to(self.cls_token.device)  # (1, 23, 192)
            
            # CLS 토큰 추가
            cls_token = self.cls_token.expand(1, -1, -1)  # (1, 1, 192)
            embeddings_with_cls = torch.cat([cls_token, embeddings_tensor], dim=1)  # (1, 24, 192)
            
            # 트랜스포머 인코더 통과
            encoded = self.transformer_encoder(embeddings_with_cls)  # (1, 24, 192)
            
            # CLS 토큰만 사용 (첫 번째 토큰)
            cls_output = encoded[:, 0, :]  # (1, 192)
            
            return cls_output.detach().cpu().numpy()
            
        elif self.aggregation_method == 'max_mean':
            # Max pooling과 Mean pooling 결합
            max_pooled = np.max(embeddings_2d, axis=0)  # (192,)
            mean_pooled = np.mean(embeddings_2d, axis=0)  # (192,)
            
            # 두 결과를 결합 (element-wise 평균)
            combined = (max_pooled + mean_pooled) / 2
            return combined.reshape(1, -1)
            
        elif self.aggregation_method == 'weighted_variance':
            # 분산 기반 가중 평균
            variances = np.var(embeddings_2d, axis=1)  # (23,) 각 특징의 분산
            weights = variances / np.sum(variances)  # 정규화
            
            weighted_avg = np.dot(weights, embeddings_2d)  # (192,)
            return weighted_avg.reshape(1, -1)
            
        else:
            # 기본값: 평균
            return np.mean(embeddings_2d, axis=0, keepdims=True)
        
    def forward(self, lab_values, data_source="train"):
        """
        lab_values에서 TabPFNEmbedding 특징 추출
        
        Args:
            lab_values: 배치의 임상 데이터 [batch_size, n_features]
            data_source: "train", "val", 또는 "test"
        
        Returns:
            TabPFN embedding 특징 [batch_size, embedding_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("TabPFN 모델이 아직 학습되지 않았습니다. fit() 메소드를 먼저 호출하세요.")
        
        # 텐서를 NumPy 배열로 변환
        lab_np = lab_values.detach().cpu().numpy()
        
        # 배치에서 환자 ID를 기반으로 중복 제거한 특징 추출
        # 실제로는 각 배치에 동일한 환자의 여러 슬라이스가 있으므로 첫 번째 행만 사용
        unique_lab = lab_np[0:1]
        
        # 특징 추출 (baseline_Cr_log 제외)
        features = np.column_stack([
            unique_lab[:, 1],  # SEX
            unique_lab[:, 2],  # AGE
            unique_lab[:, 3],  # BMI
            unique_lab[:, 4],  # BUN
            unique_lab[:, 5],  # Hb
            unique_lab[:, 6],  # Albumin
            unique_lab[:, 7],  # P
            unique_lab[:, 8],  # Ca
            unique_lab[:, 9],  # tCO2
            unique_lab[:, 10],  # K
            unique_lab[:, 11],  # Cr_48_log
            unique_lab[:, 12],  # original_shape_SurfaceVolumeRatio
            unique_lab[:, 13],  # original_shape_Sphericity
            unique_lab[:, 14],  # original_shape_Maximum3DDiameter
            unique_lab[:, 15],  # original_shape_Maximum2DDiameterColumn
            unique_lab[:, 16],  # original_shape_MinorAxisLength
            unique_lab[:, 17],  # original_shape_LeastAxisLength
            unique_lab[:, 18],  # original_shape_Elongation
            unique_lab[:, 19],  # original_shape_Flatness
            unique_lab[:, 20],  # local_thickness_mean
            unique_lab[:, 21],  # convexity_ratio_area
            unique_lab[:, 22],  # convexity_ratio_vol
            unique_lab[:, 23],  # original_shape_MeshVolume
        ])
        
        # TabPFNEmbedding을 사용하여 embedding 추출
        try:
            # Fix: data_source 매핑을 더 명확하게 처리
            # TabPFNEmbedding은 "train"과 "test"만 지원하는 것 같음
            mapped_data_source = "train" if data_source == "train" else "test"
            
            embeddings = self.embedding_extractor.get_embeddings(
                self.X_train, self.y_train, features, data_source=mapped_data_source
            )
            # embeddings shape: (n_samples, n_features, embedding_dim)
            # 우리의 경우: (1, 23, 192)
            
            # 고급 특징 집계 방법 사용
            if len(embeddings.shape) == 3:
                embeddings_2d = embeddings[0]  # (23, 192)
                pooled_embeddings = self.aggregate_embeddings(embeddings_2d)
            else:
                pooled_embeddings = embeddings
                
        except Exception as e:
            print(f"TabPFNEmbedding 추출 중 오류 발생: {e}")
            print(f"data_source: {data_source}, mapped to: {mapped_data_source}")
            print(f"Features shape: {features.shape}")
            print(f"X_train shape: {self.X_train.shape if self.X_train is not None else 'None'}")
            
            # 오류 발생 시 평균 특성을 기반으로 한 fallback 특성 반환
            # 간단한 선형 변환을 사용하여 embedding 차원으로 매핑
            fallback_features = np.mean(features, axis=1, keepdims=True)  # (1, 1)
            # 특성을 embedding 차원으로 확장
            pooled_embeddings = np.tile(fallback_features, (1, self.embedding_dim))  # (1, 192)
            print(f"Fallback embeddings shape: {pooled_embeddings.shape}")
        
        # NumPy 배열을 텐서로 변환
        features_tensor = torch.FloatTensor(pooled_embeddings).to(lab_values.device)
        
        # 배치의 모든 슬라이스에 동일한 특성 반복
        batch_size = lab_values.size(0)
        features_tensor = features_tensor.repeat(batch_size, 1)
        
        return features_tensor


class TabPFNOnlyModel(nn.Module):
    """
    TabPFNEmbedding만을 사용하는 순수 표 형식 데이터 회귀 모델
    이미지 처리 없이 임상 데이터만 활용
    """
    def __init__(self, tabpfn_dim=192, hidden_dim=512, out_dim=1,
                 dropout_rate=0.3, tabpfn_aggregation='attention'):
        super(TabPFNOnlyModel, self).__init__()
        
        # TabPFNEmbedding 특징 추출기 (고급 집계 방법 포함)
        self.tabpfn_extractor = TabPFNFeatureExtractor(
            embedding_dim=tabpfn_dim,
            aggregation_method=tabpfn_aggregation
        )
        
        # TabPFN 특징을 처리하기 위한 회귀 헤드
        self.regression_head = RegressionHead(
            in_dim=tabpfn_dim,  # TabPFN 특징만 사용
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout_rate=dropout_rate
        )

    def train(self, mode: bool = True):
        """모든 서브모듈을 학습 모드로 설정"""
        super().train(mode)
        self.regression_head.train(mode)
        self.tabpfn_extractor.train(mode)  # TabPFN은 실제로 학습하지 않지만 일관성 유지
        return self

    def eval(self):
        """모든 서브모듈을 평가 모드로 설정"""
        return self.train(False)
    
    def fit_tabpfn(self, X_train, y_train):
        """TabPFNEmbedding 특징 추출기 학습"""
        self.tabpfn_extractor.fit(X_train, y_train)
        
    def forward(self, lab_values=None, patient_ids=None, slice_indices=None, data_source="train"):
        """
        순전파: 임상 데이터만 사용하여 예측 수행
        
        Args:
            lab_values: 임상 데이터 [batch_size, n_features]
            patient_ids: 환자 ID (사용하지 않음)
            slice_indices: 슬라이스 인덱스 (사용하지 않음) 
            data_source: "train", "val", 또는 "test"
            
        Returns:
            output: 예측값 [1, 1]
        """
        if lab_values is None:
            raise ValueError("lab_values는 필수 입력입니다.")
        
        # TabPFNEmbedding 특징 추출
        tabpfn_features = self.tabpfn_extractor(lab_values, data_source=data_source)  # [batch_size, tabpfn_dim]
        
        # 첫 번째 행의 TabPFN 특징만 사용 (모든 슬라이스에서 동일한 환자)
        patient_tabpfn_features = tabpfn_features[0].unsqueeze(0)  # [1, tabpfn_dim]
        
        # 최종 회귀 (학습 가능)
        output = self.regression_head(patient_tabpfn_features)
        
        return output


def setup_tabpfn_model(args=None, out_dim=1):
    """
    TabPFNEmbedding만을 사용하는 모델 설정
    """
    # GPU 가용성 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # TabPFN 전용 모델 생성
    model = TabPFNOnlyModel(
        tabpfn_dim=args.tabpfn_dim,    # TabPFNEmbedding feature dimension
        hidden_dim=args.hidden_dim,    # Hidden layer dimension
        out_dim=out_dim,
        dropout_rate=args.dropout_rate,
        tabpfn_aggregation=args.tabpfn_aggregation
    ).to(device)

    print("TabPFN 전용 모델이 성공적으로 생성되었습니다.")

    return model, None, device

def collect_tabpfn_training_data(train_loader):
    """
    TabPFNEmbedding 학습을 위한 데이터 수집
    """
    X_train = []
    y_train = []
    
    print("TabPFNEmbedding 학습을 위한 고유한 환자 데이터 수집 중...")
    for batch in tqdm(train_loader, desc="수집 중"):
        lab_values = batch['lab_values'].cpu().numpy()
        
        # 배치의 첫 번째 행이 한 환자의 데이터 (동일한 환자의 여러 슬라이스가 배치에 있음)
        unique_lab = lab_values[0:1]
        
        # 대상 값과 특징 분리
        # baseline_Cr_log를 대상으로 설정 (인덱스 0)
        features = np.column_stack([
            unique_lab[:, 1],  # SEX
            unique_lab[:, 2],  # AGE
            unique_lab[:, 3],  # BMI
            unique_lab[:, 4],  # BUN
            unique_lab[:, 5],  # Hb
            unique_lab[:, 6],  # Albumin
            unique_lab[:, 7],  # P
            unique_lab[:, 8],  # Ca
            unique_lab[:, 9],  # tCO2
            unique_lab[:, 10],  # K
            unique_lab[:, 11],  # Cr_48_log
            unique_lab[:, 12],  # original_shape_SurfaceVolumeRatio
            unique_lab[:, 13],  # original_shape_Sphericity
            unique_lab[:, 14],  # original_shape_Maximum3DDiameter
            unique_lab[:, 15],  # original_shape_Maximum2DDiameterColumn
            unique_lab[:, 16],  # original_shape_MinorAxisLength
            unique_lab[:, 17],  # original_shape_LeastAxisLength
            unique_lab[:, 18],  # original_shape_Elongation
            unique_lab[:, 19],  # original_shape_Flatness
            unique_lab[:, 20],  # local_thickness_mean
            unique_lab[:, 21],  # convexity_ratio_area
            unique_lab[:, 22],  # convexity_ratio_vol
            unique_lab[:, 23],  # original_shape_MeshVolume
        ])
        
        log_targets = unique_lab[:, 0]  # baseline_Cr_log
        y_train.append(log_targets)
        X_train.append(features)
    
    # 모든 배치 결합
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    return X_train, y_train

def collect_validation_data(val_loader):
    """
    검증 데이터 수집
    """
    X_val = []
    y_val = []
    
    print("검증 데이터 수집 중...")
    for batch in tqdm(val_loader, desc="수집 중"):
        lab_values = batch['lab_values'].cpu().numpy()
        
        # 배치의 첫 번째 행이 한 환자의 데이터
        unique_lab = lab_values[0:1]
        
        # 특징 추출
        features = np.column_stack([
            unique_lab[:, 1],  # SEX
            unique_lab[:, 2],  # AGE
            unique_lab[:, 3],  # BMI
            unique_lab[:, 4],  # BUN
            unique_lab[:, 5],  # Hb
            unique_lab[:, 6],  # Albumin
            unique_lab[:, 7],  # P
            unique_lab[:, 8],  # Ca
            unique_lab[:, 9],  # tCO2
            unique_lab[:, 10],  # K
            unique_lab[:, 11],  # Cr_48_log
            unique_lab[:, 12],  # original_shape_SurfaceVolumeRatio
            unique_lab[:, 13],  # original_shape_Sphericity
            unique_lab[:, 14],  # original_shape_Maximum3DDiameter
            unique_lab[:, 15],  # original_shape_Maximum2DDiameterColumn
            unique_lab[:, 16],  # original_shape_MinorAxisLength
            unique_lab[:, 17],  # original_shape_LeastAxisLength
            unique_lab[:, 18],  # original_shape_Elongation
            unique_lab[:, 19],  # original_shape_Flatness
            unique_lab[:, 20],  # local_thickness_mean
            unique_lab[:, 21],  # convexity_ratio_area
            unique_lab[:, 22],  # convexity_ratio_vol
            unique_lab[:, 23],  # original_shape_MeshVolume
        ])
        
        log_targets = unique_lab[:, 0]  # baseline_Cr_log
        y_val.append(log_targets)
        X_val.append(features)
    
    # 모든 배치 결합
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    
    return X_val, y_val

def collect_test_data(test_loader):
    """
    테스트 데이터 수집
    """
    X_test = []
    y_test = []
    
    print("테스트 데이터 수집 중...")
    for batch in tqdm(test_loader, desc="수집 중"):
        lab_values = batch['lab_values'].cpu().numpy()
        
        # 배치의 첫 번째 행이 한 환자의 데이터
        unique_lab = lab_values[0:1]
        
        # 특징 추출
        features = np.column_stack([
            unique_lab[:, 1],  # SEX
            unique_lab[:, 2],  # AGE
            unique_lab[:, 3],  # BMI
            unique_lab[:, 4],  # BUN
            unique_lab[:, 5],  # Hb
            unique_lab[:, 6],  # Albumin
            unique_lab[:, 7],  # P
            unique_lab[:, 8],  # Ca
            unique_lab[:, 9],  # tCO2
            unique_lab[:, 10],  # K
            unique_lab[:, 11],  # Cr_48_log
            unique_lab[:, 12],  # original_shape_SurfaceVolumeRatio
            unique_lab[:, 13],  # original_shape_Sphericity
            unique_lab[:, 14],  # original_shape_Maximum3DDiameter
            unique_lab[:, 15],  # original_shape_Maximum2DDiameterColumn
            unique_lab[:, 16],  # original_shape_MinorAxisLength
            unique_lab[:, 17],  # original_shape_LeastAxisLength
            unique_lab[:, 18],  # original_shape_Elongation
            unique_lab[:, 19],  # original_shape_Flatness
            unique_lab[:, 20],  # local_thickness_mean
            unique_lab[:, 21],  # convexity_ratio_area
            unique_lab[:, 22],  # convexity_ratio_vol
            unique_lab[:, 23],  # original_shape_MeshVolume
        ])
        
        log_targets = unique_lab[:, 0]  # baseline_Cr_log
        y_test.append(log_targets)
        X_test.append(features)
    
    # 모든 배치 결합
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    
    return X_test, y_test

def train_tabpfn_model(model, train_loader, criterion, optimizer, device, epoch, X_train, y_train, X_val, y_val, log_interval=10):
    """
    TabPFN 모델을 한 에포크 동안 학습 (이미지 없이 임상 데이터만 사용)
    """
    model.train()  # 전체 모델을 학습 모드로 설정
    running_loss = 0.0
    total_loss = 0.0
    start_time = time.time()
    patient_count = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for batch_idx, batch in progress_bar:
        # 임상 데이터만 가져오기 (이미지는 무시)
        lab_values = batch['lab_values'].to(device)   # [batch_size, 24]
        patient_ids = batch['patient_id']             # 환자 ID 리스트
        slice_indices = batch['slice_idx'].to(device) # [batch_size]
        
        # 대상값은 baseline_Cr_log (lab_values의 인덱스 0)
        baseline_Cr_log = lab_values[:, 0].view(-1, 1)         # [batch_size, 1]
        
        # 모델 순전파 (임상 데이터만 사용) - 학습 시에는 data_source="train"
        outputs = model(lab_values=lab_values, patient_ids=patient_ids, 
                       slice_indices=slice_indices, data_source="train")

        # 배치의 모든 슬라이스가 동일한 환자이므로 첫 번째 슬라이스의 대상값만 비교
        loss = criterion(outputs, baseline_Cr_log[0].unsqueeze(0))
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑 (안정적인 학습을 위해)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        total_loss += loss.item()
        patient_count += 1
        
        # 진행 상황 기록
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed_time = time.time() - start_time
            progress_bar.set_description(
                f'Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | '
                f'Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s'
            )
            running_loss = 0.0
    
    # 에포크의 평균 손실 반환
    return total_loss / patient_count


def validate_tabpfn_model(model, val_loader, criterion, device, X_train, y_train, X_val, y_val):
    """
    TabPFN 모델을 검증 세트에서 평가 (개선된 오류 처리 포함)
    """
    model.eval()  # 평가 모드 설정
    total_loss = 0.0
    patient_count = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="검증 중")):
            try:
                # 임상 데이터만 가져오기 (이미지는 무시)
                lab_values = batch['lab_values'].to(device)
                patient_ids = batch['patient_id']
                slice_indices = batch['slice_idx'].to(device)
                
                # 대상값은 baseline_Cr_log (lab_values의 인덱스 0)
                baseline_Cr_log = lab_values[:, 0].view(-1, 1)
                
                # 모델 순전파 (임상 데이터만 사용) - 검증 시에는 data_source="val"
                outputs = model(lab_values=lab_values, patient_ids=patient_ids, 
                               slice_indices=slice_indices, data_source="val")
                
                # 손실 계산 (모든 슬라이스가 동일한 대상을 가지므로 첫 번째만 사용)
                loss = criterion(outputs, baseline_Cr_log[0].unsqueeze(0))
                
                # 통계 업데이트
                total_loss += loss.item()
                patient_count += 1
                
                # 예측 및 대상 저장 (지표 계산용)
                all_targets.append(np.exp(baseline_Cr_log[0].item()))
                all_preds.append(np.exp(outputs.item()))
                
            except Exception as e:
                print(f"검증 배치 {batch_idx}에서 오류 발생: {e}")
                continue
    
    # 지표 계산
    if len(all_targets) > 0:
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
    else:
        # 모든 배치가 실패한 경우 기본값 반환
        mse = float('inf')
        mae = float('inf')
        r2 = -float('inf')
    
    # 지표 반환
    return {
        'loss': total_loss / patient_count if patient_count > 0 else float('inf'),
        'mse': mse,
        'rmse': np.sqrt(mse) if mse != float('inf') else float('inf'),
        'mae': mae,
        'r2': r2
    }


def test_tabpfn_model(model, test_loader, device, X_train, y_train, X_test, y_test):
    """
    TabPFN 모델을 테스트하고 환자 수준 예측 생성 (개선된 오류 처리 포함)
    """
    model.eval()  # 평가 모드 설정
    
    # 환자 예측 저장
    patient_predictions = {}
    patient_targets = {}
    patient_lab_values = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="테스트 중")):
            try:
                # 임상 데이터만 가져오기 (이미지는 무시)
                lab_values = batch['lab_values'].to(device)
                patient_ids = batch['patient_id']
                slice_indices = batch['slice_idx'].to(device)
                
                # 대상값은 baseline_Cr_log (lab_values의 인덱스 0)
                baseline_Cr_log = lab_values[:, 0].view(-1, 1)
                
                # 모델 순전파 - 테스트 시에는 data_source="test"
                outputs = model(lab_values=lab_values, patient_ids=patient_ids, 
                               slice_indices=slice_indices, data_source="test")
                
                # 환자 ID 및 예측 저장
                patient_id = patient_ids[0]  # 배치의 모든 ID가 동일
                
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = []
                    patient_targets[patient_id] = baseline_Cr_log[0].item()
                    patient_lab_values[patient_id] = lab_values[0].cpu().numpy()
                
                patient_predictions[patient_id].append(outputs.item())
                
            except Exception as e:
                print(f"테스트 배치 {batch_idx}에서 오류 발생: {e}")
                print(f"환자 ID: {patient_ids[0] if 'patient_ids' in locals() else 'Unknown'}")
                continue
    
    # 환자별 예측 집계 (평균 사용)
    final_predictions = {}
    for patient_id, preds in patient_predictions.items():
        if len(preds) > 0:
            final_predictions[patient_id] = np.mean(preds)
    
    if len(final_predictions) == 0:
        print("경고: 성공적으로 처리된 테스트 배치가 없습니다.")
        return {
            'mse': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': -float('inf')
        }, pd.DataFrame()
    
    # 원래 스케일로 변환하여 지표 계산
    final_predictions = {pid: np.exp(pred) for pid, pred in final_predictions.items()}
    patient_targets = {pid: np.exp(target) for pid, target in patient_targets.items() if pid in final_predictions}

    # 원래 스케일에서 지표 계산
    all_targets_original = np.array([target for target in patient_targets.values()])
    all_preds_original = np.array([pred for pred in final_predictions.values()])

    mse = mean_squared_error(all_targets_original, all_preds_original)
    mae = mean_absolute_error(all_targets_original, all_preds_original)
    r2 = r2_score(all_targets_original, all_preds_original) 
    
    # 결과 데이터프레임 생성
    results_df = pd.DataFrame({
        'patient_id': list(final_predictions.keys()),
        'true_baseline_Cr': [patient_targets[pid] for pid in final_predictions.keys()],
        'predicted_baseline_Cr': [final_predictions[pid] for pid in final_predictions.keys()]
    })
    
    # 지표 및 결과 반환
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }
    
    return metrics, results_df

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        import shutil
        shutil.copyfile(filepath, best_filepath)

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(
        log_dir, f"training_{time.strftime('%Y%m%d-%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,          # ← 이미 설정돼 있어도 덮어쓴다
    )

    return logging.getLogger(__name__)

def main():
    """
    TabPFN Embedding 전용 모델 학습 파이프라인 메인 함수 (이미지 없음)
    """
    parser = argparse.ArgumentParser(description='Kidney TabPFN-Only Model Training (Lab Values Only)')
    parser.add_argument('--train-csv', type=str, default='./Data/INHA_HoldOut2.csv', help='Path to training CSV')
    parser.add_argument('--test-csv', type=str, default='./Data/KMUH.csv', help='Path to test CSV')
#     parser.add_argument('--train-csv', type=str, default='./Data/example_train.csv', help='Path to training CSV')
#     parser.add_argument('--test-csv', type=str, default='./Data/example_test.csv', help='Path to test CSV')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--target-slices', type=int, default=16, help='Target number of kidney slices')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--early-stopping', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/TabPFN_Only', help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs/TabPFN_Only', help='Directory to save logs')
    parser.add_argument('--results-dir', type=str, default='./results/TabPFN_Only', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tabpfn-dim', type=int, default=192, help='TabPFNEmbedding feature dimension')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden layer dimension for regression head')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--tabpfn-aggregation', type=str, default='attention', 
                       choices=['mean', 'attention', 'learnable_pooling', 'transformer', 'max_mean', 'weighted_variance'],
                       help='TabPFN embedding aggregation method')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    args = parser.parse_args()
    
    # 재현성을 위한 랜덤 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 로깅 설정
    logger = setup_logging(args.log_dir)
    logger.info(f"Args: {args}")
    
    # 디렉토리 확인 및 생성
    for directory in [args.checkpoint_dir, args.log_dir, args.results_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 데이터 로더 생성
    logger.info("데이터 로더 생성 중...")
    train_loader, val_loader, test_inha_loader, test_kmuh_loader = create_four_dataloaders(
        inha_csv_path=args.train_csv,
        kmuh_csv_path=args.test_csv,
        batch_size=args.batch_size,
        resize_dim=224,  # 이미지는 사용하지 않지만 호환성을 위해 유지
        target_slices=args.target_slices
    )
    
    # 모델 설정
    logger.info("TabPFN 전용 모델 설정 중...")
    model, _, device = setup_tabpfn_model(args=args, out_dim=1)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"총 파라미터 수: {total_params:,}")
    logger.info(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # TabPFNEmbedding 학습을 위한 데이터 수집
    logger.info("TabPFNEmbedding 학습을 위한 데이터 수집 중...")
    X_train, y_train = collect_tabpfn_training_data(train_loader)
    
    # 검증 및 테스트 데이터 수집
    logger.info("검증 데이터 수집 중...")
    X_val, y_val = collect_validation_data(val_loader)
    
    logger.info("INHA 테스트 데이터 수집 중...")
    X_test_inha, y_test_inha = collect_test_data(test_inha_loader)
    
    logger.info("KMUH 테스트 데이터 수집 중...")
    X_test_kmuh, y_test_kmuh = collect_test_data(test_kmuh_loader)
    
    # TabPFNEmbedding 모델 학습
    logger.info("TabPFNEmbedding 모델 학습 중...")
    model.fit_tabpfn(X_train, y_train)
    
    # 손실 함수 및 최적화기 정의
    class RMSELoss(nn.Module):
        """
        로그 스케일 값에 대한 지수 변환이 포함된 RMSE 손실
        """
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()
            
        def forward(self, predictions, targets):
            # 예측 및 대상에 지수 변환 적용
            predictions = torch.exp(predictions)
            targets = torch.exp(targets)
            # RMSE 계산
            return torch.sqrt(self.mse(predictions, targets))
    
    # 손실 함수 및 최적화기 설정
    criterion = RMSELoss()
    
    # 최적화기 설정 (단일 학습률, TabPFN은 학습하지 않으므로)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        eps=1e-8
    )
    
    # 학습률 스케줄러 설정
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # 학습 변수 초기화
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # 학습 루프
    logger.info("학습 시작...")
    logger.info(f"총 학습 가능한 파라미터: {trainable_params:,}개")
    logger.info(f"TabPFN 집계 방법: {args.tabpfn_aggregation}")
    
    for epoch in range(1, args.epochs + 1):
        # 학습
        train_loss = train_tabpfn_model(
            model, train_loader, criterion, optimizer, device, epoch, 
            X_train, y_train, X_val, y_val, args.log_interval
        )
        
        # 검증
        val_metrics = validate_tabpfn_model(
            model, val_loader, criterion, device, X_train, y_train, X_val, y_val
        )
        val_loss = val_metrics['rmse']
        
        # 학습률 업데이트
        scheduler.step()
        
        # 현재 학습률 가져오기
        current_lr = optimizer.param_groups[0]['lr']
        
        # 지표 기록
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R²: {val_metrics['r2']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # 최고 모델 확인
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # 체크포인트 저장
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': args,
                },
                is_best,
                args.checkpoint_dir,
                filename=f'checkpoint_epoch{epoch}.pth.tar'
            )
        else:
            patience_counter += 1
        
        # 조기 종료
        if patience_counter >= args.early_stopping:
            logger.info(f"{epoch}번째 에포크 후 조기 종료")
            break
    
    # 테스트를 위해 최고 모델 로드
    logger.info(f"{best_epoch}번째 에포크에서 최고 모델 로드 중...")
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'model_best.pth.tar'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    # INHA 테스트 데이터로 모델 테스트
    logger.info("INHA 데이터에서 모델 테스트 중...")
    test_metrics_inha, results_df_inha = test_tabpfn_model(
        model, test_inha_loader, device, X_train, y_train, X_test_inha, y_test_inha
    )
    
    # 테스트 지표 기록
    logger.info(
        f"INHA 테스트 결과 | "
        f"MSE: {test_metrics_inha['mse']:.4f} | "
        f"RMSE: {test_metrics_inha['rmse']:.4f} | "
        f"MAE: {test_metrics_inha['mae']:.4f} | "
        f"R²: {test_metrics_inha['r2']:.4f}"
    )
    
    # 테스트 결과 저장
    results_path_inha = os.path.join(args.results_dir, f'test_results_Inha_{time.strftime("%Y%m%d-%H%M%S")}.csv')
    results_df_inha.to_csv(results_path_inha, index=False)
    logger.info(f"테스트 결과 저장됨: {results_path_inha}")
    
    # KMUH 테스트 데이터로 모델 테스트
    logger.info("KMUH 데이터에서 모델 테스트 중...")
    test_metrics_kmuh, results_df_kmuh = test_tabpfn_model(
        model, test_kmuh_loader, device, X_train, y_train, X_test_kmuh, y_test_kmuh
    )
    
    # 테스트 지표 기록
    logger.info(
        f"KMUH 테스트 결과 | "
        f"MSE: {test_metrics_kmuh['mse']:.4f} | "
        f"RMSE: {test_metrics_kmuh['rmse']:.4f} | "
        f"MAE: {test_metrics_kmuh['mae']:.4f} | "
        f"R²: {test_metrics_kmuh['r2']:.4f}"
    )
    
    # 테스트 결과 저장
    results_path_kmuh = os.path.join(args.results_dir, f'test_results_Kmuh_{time.strftime("%Y%m%d-%H%M%S")}.csv')
    results_df_kmuh.to_csv(results_path_kmuh, index=False)
    logger.info(f"테스트 결과 저장됨: {results_path_kmuh}")
    
    logger.info("TabPFN 전용 모델 학습 완료!")


if __name__ == "__main__":
    main()
