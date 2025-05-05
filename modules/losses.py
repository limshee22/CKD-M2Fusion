import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """로그 스케일 값에 대한 지수 변환이 포함된 RMSE 손실"""
    def __init__(self):
        """초기화 메서드"""
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        순전파 메서드
        
        Args:
            predictions: 예측값 텐서
            targets: 실제값 텐서
            
        Returns:
            손실값 텐서
        """
        # 예측 및 대상에 지수 변환 적용
        predictions = torch.exp(predictions)
        targets = torch.exp(targets)
        # RMSE 계산
        return torch.sqrt(self.mse(predictions, targets))