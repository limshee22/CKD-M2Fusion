from torch.utils.data import DataLoader
from typing import Tuple

from .datasets import KidneySliceDataset, TestKidneySliceDataset
from .samplers import PatientBatchSampler


def create_four_dataloaders(
    inha_csv_path: str = "./Data/INHA.csv", 
    kmuh_csv_path: str = "./Data/KMUH.csv", 
    batch_size: int = 16, 
    resize_dim: int = 224, 
    target_slices: int = 48
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    4개의 데이터로더 생성 함수
    
    INHA.csv 파일을 활용하여 train, valid, test_inha 데이터로더를 생성하고,
    KMUH.csv 파일로 test_kmuh 데이터로더를 생성
    
    Split=1은 train으로 사용
    Split=2는 valid로 사용
    Split=3은 test_inha로 사용
    KMUH는 전체를 test_kmuh로 사용
    
    Args:
        inha_csv_path: INHA.csv 파일 경로
        kmuh_csv_path: KMUH.csv 파일 경로
        batch_size: 배치 크기
        resize_dim: 이미지 리사이즈 차원
        target_slices: 목표 슬라이스 수
        
    Returns:
        train_loader, val_loader, test_inha_loader, test_kmuh_loader: 데이터로더 4개 튜플
    """
    try:
        # 훈련용 데이터셋 생성 (Split==1)
        train_dataset = KidneySliceDataset(inha_csv_path, split=1, resize_dim=resize_dim, target_slices=target_slices)
        
        # 검증용 데이터셋 생성 (Split==2)
        val_dataset = KidneySliceDataset(inha_csv_path, split=2, resize_dim=resize_dim, target_slices=target_slices)
        
        # INHA 테스트용 데이터셋 생성 (Split==3)
        test_inha_dataset = KidneySliceDataset(inha_csv_path, split=3, resize_dim=resize_dim, target_slices=target_slices)
        
        # KMUH 테스트용 데이터셋 생성
        test_kmuh_dataset = TestKidneySliceDataset(kmuh_csv_path, resize_dim=resize_dim, target_slices=target_slices)
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=PatientBatchSampler(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            ),
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=PatientBatchSampler(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            ),
            num_workers=0,
            pin_memory=False
        )
        
        test_inha_loader = DataLoader(
            test_inha_dataset,
            batch_sampler=PatientBatchSampler(
                dataset=test_inha_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            ),
            num_workers=0,
            pin_memory=False
        )
        
        test_kmuh_loader = DataLoader(
            test_kmuh_dataset,
            batch_sampler=PatientBatchSampler(
                dataset=test_kmuh_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            ),
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, val_loader, test_inha_loader, test_kmuh_loader
        
    except Exception as e:
        print(f"데이터로더 생성 오류: {str(e)}")
        raise e