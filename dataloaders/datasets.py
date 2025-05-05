import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from typing import Dict, Any, List, Tuple


class KidneySliceDataset(Dataset):
    """환자 별로 모든 신장 슬라이스를 개별 샘플로 처리하는 데이터셋"""

    def __init__(self, csv_path: str, split: int = 1, resize_dim: int = 224, target_slices: int = 50):
        """
        초기화 메서드
        
        Args:
            csv_path: CSV 파일 경로
            split: 1 for train, 2 for validation, 3 for test
            resize_dim: 이미지 리사이즈 크기
            target_slices: mask target slices. if over,under => cutting, zero padding
        """
        self.resize_dim = resize_dim
        self.target_slices = target_slices
        
        # CSV 파일 읽기
        self.df = pd.read_csv(csv_path)
        
        # Split에 따라 데이터 필터링
        self.df = self.df[self.df['Split'] == split].reset_index(drop=True)
        split_name = "훈련" if split == 1 else "검증" if split == 2 else "테스트"
        print(f"{split_name} 환자 수: {len(self.df)}")
        
        # 환자 ID와 각 슬라이스 인덱스를 저장할 리스트
        self.samples = []

        # 모든 환자의 데이터를 미리 처리하여 슬라이스 인덱스를 저장
        self._process_patient_data()
        
        print(f"총 슬라이스 샘플 수: {len(self.samples)}")

    def _process_patient_data(self) -> None:
        """모든 환자의 데이터를 처리하여 슬라이스 인덱스 저장"""
        for idx in range(len(self.df)):
            patient_id = str(self.df.loc[idx, 'A_NUM'])
            
            try:
                # 마스크 로드
                mask_path = f"./NPY/INHA_MASK/{patient_id}_mask.npy"
                mask_volume = np.load(mask_path)
                
                # 신장이 있는 슬라이스 찾기
                kidney_slices = np.where(np.any(mask_volume == 1, axis=(1, 2)))[0]
                
                if len(kidney_slices) == 0:
                    print(f"환자 {patient_id}의 마스크에서 신장 슬라이스를 찾을 수 없습니다.")
                    continue

                # 슬라이스 개수를 target_slices에 맞춰줌
                kidney_slices = self._adjust_slice_count(kidney_slices)

                # 이제 kidney_slices는 target_slices 길이를 가짐
                for slice_idx in kidney_slices:
                    self.samples.append({
                        'patient_idx': idx,
                        'slice_idx': slice_idx
                    })
                
                print(f"환자 {patient_id}에서 {len(kidney_slices)}개의 신장 슬라이스를 최종 추가했습니다.")
                
            except Exception as e:
                print(f"환자 {patient_id} 처리 중 오류 발생: {str(e)}")

    def _adjust_slice_count(self, kidney_slices: np.ndarray) -> np.ndarray:
        """
        슬라이스 개수를 target_slices에 맞게 조정
        
        Args:
            kidney_slices: 신장이 있는 슬라이스 인덱스 배열
            
        Returns:
            조정된 슬라이스 인덱스 배열
        """
        num_kidney_slices = len(kidney_slices)
        
        if num_kidney_slices < self.target_slices:
            # 부족하면 앞뒤로 패딩
            pad_total = self.target_slices - num_kidney_slices
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front  # 홀수 차이면 뒤쪽을 +1 더 패딩
            kidney_slices = np.pad(
                kidney_slices,
                pad_width=(pad_front, pad_back),
                mode='edge'  # 가장자리 값으로 패딩
            )
            
        elif num_kidney_slices > self.target_slices:
            # 많으면 앞뒤로 잘라서 제한
            cut_total = num_kidney_slices - self.target_slices
            cut_front = cut_total // 2
            cut_back = cut_total - cut_front  # 홀수 차이면 뒤쪽을 +1 더 제거
            kidney_slices = kidney_slices[cut_front : -cut_back if cut_back != 0 else None]
            
        return kidney_slices

    def __len__(self) -> int:
        """데이터셋 길이 반환"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        인덱스에 해당하는 샘플 반환
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            샘플 딕셔너리 (patient_id, slice_idx, image, lab_values)
        """
        try:
            # 샘플 정보 가져오기
            sample = self.samples[idx]
            patient_idx = sample['patient_idx']
            target_slice_idx = sample['slice_idx']
            
            # 환자 ID 가져오기
            patient_id = str(self.df.loc[patient_idx, 'A_NUM'])
            
            # 데이터 로드
            dicom_path = f"./NPY/INHA/{patient_id}_DICOM.npy"
            mask_path = f"./NPY/INHA_MASK/{patient_id}_mask.npy"
            
            # numpy 배열 로드
            ct_volume = np.load(dicom_path).copy()
            mask_volume = np.load(mask_path)
            
            # 3채널 이미지 생성
            image_3ch = self._create_3channel_image(ct_volume, mask_volume, target_slice_idx)
            
            # 실험값 가져오기
            lab_values = self._get_lab_values(patient_idx)

            return {
                'patient_id': patient_id,
                'slice_idx': target_slice_idx,
                'image': torch.FloatTensor(image_3ch),  # [3, H, W] 형태
                'lab_values': torch.FloatTensor(lab_values)
            }
            
        except Exception as e:
            print(f"인덱스 {idx}에 대한 __getitem__에서 오류 발생: {str(e)}")
            raise e

    def _create_3channel_image(self, ct_volume: np.ndarray, mask_volume: np.ndarray, 
                               target_slice_idx: int) -> np.ndarray:
        """
        3채널 이미지 생성
        
        Args:
            ct_volume: CT 볼륨 데이터
            mask_volume: 마스크 볼륨 데이터
            target_slice_idx: 대상 슬라이스 인덱스
            
        Returns:
            3채널 이미지 배열 [3, H, W]
        """
        channels = []

        # 동일한 슬라이스 3번 복사 (= 3 channel)
        if target_slice_idx == -1:
            masked_slice = np.zeros_like(ct_volume[0])
        else:
            # 실제 슬라이스
            masked_slice = ct_volume[target_slice_idx] * mask_volume[target_slice_idx]

        channels = [masked_slice, masked_slice, masked_slice]
        
        # 리사이즈
        channels = [cv2.resize(slice, (self.resize_dim, self.resize_dim)) for slice in channels]
        
        # 3채널 이미지로 스택
        return np.stack(channels, axis=0)

    def _get_lab_values(self, patient_idx: int) -> List[float]:
        """
        환자의 실험실 값 가져오기
        
        Args:
            patient_idx: 환자 인덱스
            
        Returns:
            실험실 값 리스트
        """
        baseline_cr = float(self.df.loc[patient_idx, 'Baseline_Cr_log'])

        SEX = float(self.df.loc[patient_idx, 'SEX'])
        AGE = float(self.df.loc[patient_idx, 'AGE'])
        BMI = float(self.df.loc[patient_idx, 'BMI'])
        BUN = float(self.df.loc[patient_idx, 'BUN'])
        Hb = float(self.df.loc[patient_idx, 'Hb'])
        Albumin = float(self.df.loc[patient_idx, 'Albumin'])
        P = float(self.df.loc[patient_idx, 'P'])
        Ca = float(self.df.loc[patient_idx, 'Ca'])
        tCO2 = float(self.df.loc[patient_idx, 'tCO2'])
        K = float(self.df.loc[patient_idx, 'K'])
        Cr_48h_log = float(self.df.loc[patient_idx, 'Cr_48h_log'])

        original_shape_SurfaceVolumeRatio = float(self.df.loc[patient_idx, 'original_shape_SurfaceVolumeRatio'])
        original_shape_Sphericity = float(self.df.loc[patient_idx, 'original_shape_Sphericity'])
        original_shape_Maximum3DDiameter = float(self.df.loc[patient_idx, 'original_shape_Maximum3DDiameter'])
        original_shape_Maximum2DDiameterColumn = float(self.df.loc[patient_idx, 'original_shape_Maximum2DDiameterColumn'])
        original_shape_MinorAxisLength = float(self.df.loc[patient_idx, 'original_shape_MinorAxisLength'])
        original_shape_LeastAxisLength = float(self.df.loc[patient_idx, 'original_shape_LeastAxisLength'])
        original_shape_Elongation = float(self.df.loc[patient_idx, 'original_shape_Elongation'])
        original_shape_Flatness = float(self.df.loc[patient_idx, 'original_shape_Flatness'])
        local_thickness_mean = float(self.df.loc[patient_idx, 'local_thickness_mean'])
        convexity_ratio_area = float(self.df.loc[patient_idx, 'convexity_ratio_area'])
        convexity_ratio_vol = float(self.df.loc[patient_idx, 'convexity_ratio_vol'])
        original_shape_MeshVolume = float(self.df.loc[patient_idx, 'original_shape_MeshVolume'])

        
        return [baseline_cr, SEX, AGE, BMI, BUN, Hb, Albumin, P, Ca, tCO2, K,
                Cr_48h_log, original_shape_SurfaceVolumeRatio, original_shape_Sphericity,
                original_shape_Maximum3DDiameter, original_shape_Maximum2DDiameterColumn,
                original_shape_MinorAxisLength, original_shape_LeastAxisLength,
                original_shape_Elongation, original_shape_Flatness, local_thickness_mean,
                convexity_ratio_area, convexity_ratio_vol, original_shape_MeshVolume]


class TestKidneySliceDataset(Dataset):
    """테스트 환자 별로 모든 신장 슬라이스를 개별 샘플로 처리하는 데이터셋"""

    def __init__(self, csv_path: str, resize_dim: int = 224, target_slices: int = 50):
        """
        초기화 메서드
        
        Args:
            csv_path: CSV 파일 경로
            resize_dim: 이미지 리사이즈 크기
            target_slices: 목표 슬라이스 수
        """
        self.resize_dim = resize_dim
        self.target_slices = target_slices

        # CSV 파일 읽기
        self.df = pd.read_csv(csv_path)
        print(f"테스트 환자 수: {len(self.df)}")
        
        # 환자 ID와 각 슬라이스 인덱스를 저장할 리스트
        self.samples = []

        # 모든 환자의 데이터를 미리 처리하여 슬라이스 인덱스를 저장
        self._process_patient_data()
        
        print(f"총 테스트 슬라이스 샘플 수: {len(self.samples)}")

    def format_patient_id(self, patient_id: Any) -> str:
        """
        ID가 7자리가 되도록 앞에 0을 채움
        
        Args:
            patient_id: 환자 ID
            
        Returns:
            형식화된 환자 ID
        """
        return str(patient_id).zfill(7)

    def _process_patient_data(self) -> None:
        """모든 환자의 데이터를 처리하여 슬라이스 인덱스 저장"""
        for idx in range(len(self.df)):
            patient_id = self.format_patient_id(self.df.loc[idx, 'A_NUM'])
            
            try:
                # 마스크 로드
                mask_path = f"./NPY/KMUH_MASK/{patient_id}_mask.npy"
                mask_volume = np.load(mask_path)
                
                # 신장이 있는 슬라이스 찾기
                kidney_slices = np.where(np.any(mask_volume == 1, axis=(1,2)))[0]
                
                if len(kidney_slices) == 0:
                    print(f"환자 {patient_id}의 마스크에서 신장 슬라이스를 찾을 수 없습니다.")
                    continue
                
                # 슬라이스 개수를 target_slices에 맞춤
                kidney_slices = self._adjust_slice_count(kidney_slices)

                # 이제 kidney_slices는 target_slices 길이를 가짐
                for slice_idx in kidney_slices:
                    self.samples.append({
                        'patient_idx': idx,
                        'slice_idx': slice_idx
                    })
                
                print(f"환자 {patient_id}에서 {len(kidney_slices)}개의 신장 슬라이스를 최종 추가했습니다.")
                
            except Exception as e:
                print(f"환자 {patient_id} 처리 중 오류 발생: {str(e)}")

    def _adjust_slice_count(self, kidney_slices: np.ndarray) -> np.ndarray:
        """
        슬라이스 개수를 target_slices에 맞게 조정
        
        Args:
            kidney_slices: 신장이 있는 슬라이스 인덱스 배열
            
        Returns:
            조정된 슬라이스 인덱스 배열
        """
        num_kidney_slices = len(kidney_slices)
        
        if num_kidney_slices < self.target_slices:
            # 부족 -> 패딩
            pad_total = self.target_slices - num_kidney_slices
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front
            kidney_slices = np.pad(
                kidney_slices,
                pad_width=(pad_front, pad_back),
                mode='edge'
            )
        elif num_kidney_slices > self.target_slices:
            # 많음 -> 앞뒤 자르기
            cut_total = num_kidney_slices - self.target_slices
            cut_front = cut_total // 2
            cut_back = cut_total - cut_front
            kidney_slices = kidney_slices[cut_front : -cut_back if cut_back != 0 else None]
            
        return kidney_slices

    def __len__(self) -> int:
        """데이터셋 길이 반환"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        인덱스에 해당하는 샘플 반환
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            샘플 딕셔너리 (patient_id, slice_idx, image, lab_values)
        """
        try:
            # 샘플 정보 가져오기
            sample = self.samples[idx]
            patient_idx = sample['patient_idx']
            target_slice_idx = sample['slice_idx']
            
            # 환자 ID 가져오기 및 형식 지정
            patient_id = self.format_patient_id(self.df.loc[patient_idx, 'A_NUM'])
            
            # 데이터 로드
            dicom_path = f"./NPY/KMUH/{patient_id}_DICOM.npy"
            mask_path = f"./NPY/KMUH_MASK/{patient_id}_mask.npy"
            
            # numpy 배열 로드
            ct_volume = np.load(dicom_path).copy()
            mask_volume = np.load(mask_path)
            
            # 3채널 이미지 생성
            image_3ch = self._create_3channel_image(ct_volume, mask_volume, target_slice_idx)
            
            # 실험값 가져오기
            lab_values = self._get_lab_values(patient_idx)

            return {
                'patient_id': patient_id,
                'slice_idx': target_slice_idx,
                'image': torch.FloatTensor(image_3ch),  # [3, H, W] 형태
                'lab_values': torch.FloatTensor(lab_values)
            }
            
        except Exception as e:
            print(f"인덱스 {idx}에 대한 __getitem__에서 오류 발생: {str(e)}")
            raise e

    def _create_3channel_image(self, ct_volume: np.ndarray, mask_volume: np.ndarray,
                               target_slice_idx: int) -> np.ndarray:
        """
        3채널 이미지 생성
        
        Args:
            ct_volume: CT 볼륨 데이터
            mask_volume: 마스크 볼륨 데이터
            target_slice_idx: 대상 슬라이스 인덱스
            
        Returns:
            3채널 이미지 배열 [3, H, W]
        """
        channels = []

        # 동일한 슬라이스 3번 복사 (= 3 channel)
        if target_slice_idx == -1:
            masked_slice = np.zeros_like(ct_volume[0])
        else:
            # 실제 슬라이스
            masked_slice = ct_volume[target_slice_idx] * mask_volume[target_slice_idx]

        channels = [masked_slice, masked_slice, masked_slice]
        
        # 리사이즈
        channels = [cv2.resize(slice, (self.resize_dim, self.resize_dim)) for slice in channels]
        
        # 3채널 이미지로 스택
        return np.stack(channels, axis=0)

    def _get_lab_values(self, patient_idx: int) -> List[float]:
        """
        환자의 실험실 값 가져오기
        
        Args:
            patient_idx: 환자 인덱스
            
        Returns:
            실험실 값 리스트
        """
        baseline_cr = float(self.df.loc[patient_idx, 'Baseline_Cr_log'])

        SEX = float(self.df.loc[patient_idx, 'SEX'])
        AGE = float(self.df.loc[patient_idx, 'AGE'])
        BMI = float(self.df.loc[patient_idx, 'BMI'])
        BUN = float(self.df.loc[patient_idx, 'BUN'])
        Hb = float(self.df.loc[patient_idx, 'Hb'])
        Albumin = float(self.df.loc[patient_idx, 'Albumin'])
        P = float(self.df.loc[patient_idx, 'P'])
        Ca = float(self.df.loc[patient_idx, 'Ca'])
        tCO2 = float(self.df.loc[patient_idx, 'tCO2'])
        K = float(self.df.loc[patient_idx, 'K'])
        Cr_48h_log = float(self.df.loc[patient_idx, 'Cr_48h_log'])

        original_shape_SurfaceVolumeRatio = float(self.df.loc[patient_idx, 'original_shape_SurfaceVolumeRatio'])
        original_shape_Sphericity = float(self.df.loc[patient_idx, 'original_shape_Sphericity'])
        original_shape_Maximum3DDiameter = float(self.df.loc[patient_idx, 'original_shape_Maximum3DDiameter'])
        original_shape_Maximum2DDiameterColumn = float(self.df.loc[patient_idx, 'original_shape_Maximum2DDiameterColumn'])
        original_shape_MinorAxisLength = float(self.df.loc[patient_idx, 'original_shape_MinorAxisLength'])
        original_shape_LeastAxisLength = float(self.df.loc[patient_idx, 'original_shape_LeastAxisLength'])
        original_shape_Elongation = float(self.df.loc[patient_idx, 'original_shape_Elongation'])
        original_shape_Flatness = float(self.df.loc[patient_idx, 'original_shape_Flatness'])
        local_thickness_mean = float(self.df.loc[patient_idx, 'local_thickness_mean'])
        convexity_ratio_area = float(self.df.loc[patient_idx, 'convexity_ratio_area'])
        convexity_ratio_vol = float(self.df.loc[patient_idx, 'convexity_ratio_vol'])
        original_shape_MeshVolume = float(self.df.loc[patient_idx, 'original_shape_MeshVolume'])

        
        return [baseline_cr, SEX, AGE, BMI, BUN, Hb, Albumin, P, Ca, tCO2, K,
                Cr_48h_log, original_shape_SurfaceVolumeRatio, original_shape_Sphericity,
                original_shape_Maximum3DDiameter, original_shape_Maximum2DDiameterColumn,
                original_shape_MinorAxisLength, original_shape_LeastAxisLength,
                original_shape_Elongation, original_shape_Flatness, local_thickness_mean,
                convexity_ratio_area, convexity_ratio_vol, original_shape_MeshVolume]