import random
from collections import defaultdict
from torch.utils.data import Sampler
from typing import List, Dict, Iterator, Optional


class PatientBatchSampler(Sampler):
    """한 환자 슬라이스들만 모아 한 배치를 만드는 커스텀 Sampler"""
    
    def __init__(self, dataset, batch_size: int = 16, shuffle: bool = True, drop_last: bool = False):
        """
        초기화 메서드
        
        Args:
            dataset: 데이터셋 인스턴스
            batch_size: 배치 크기
            shuffle: 셔플 여부
            drop_last: 마지막 배치 버림 여부
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 환자별로 샘플 인덱스를 모아둠
        self.patient_to_indices = self._group_indices_by_patient()
        
        # 환자 목록
        self.patients = list(self.patient_to_indices.keys())
        
        # 환자 순서/슬라이스 순서를 셔플할지
        if self.shuffle:
            random.shuffle(self.patients)
            for p in self.patients:
                self.patient_to_indices[p].sort()

    def _group_indices_by_patient(self) -> Dict[int, List[int]]:
        """
        환자별로 샘플 인덱스를 그룹화
        
        Returns:
            환자별 샘플 인덱스 딕셔너리
        """
        patient_to_indices = defaultdict(list)
        
        for i, sample in enumerate(self.dataset.samples):
            p_idx = sample['patient_idx']  # dataset.df의 row index
            patient_to_indices[p_idx].append(i)
            
        return patient_to_indices

    def __iter__(self) -> Iterator[List[int]]:
        """
        환자 하나씩 꺼내면서, batch_size 만큼 슬라이스를 끊어 yield
        
        Returns:
            배치 인덱스 리스트 반복자
        """
        for p_idx in self.patients:
            indices = self.patient_to_indices[p_idx]
            # batch_size 단위로 슬라이스를 끊어서 배치 생성
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        """
        총 배치 개수 반환
        
        Returns:
            총 배치 개수
        """
        total_batches = 0
        for p_idx in self.patients:
            count = len(self.patient_to_indices[p_idx])
            if self.drop_last:
                total_batches += (count // self.batch_size)
            else:
                total_batches += (count + self.batch_size - 1) // self.batch_size
        return total_batches