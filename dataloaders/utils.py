import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional


def prepare_split_data(inha_csv_path: str) -> str:
    """
    INHA.csv 파일을 Split 열을 기준으로 분할하고 새 CSV 파일 생성
    (현재 이 함수는 사용되지 않고 있지만, 향후 필요할 수 있으므로 구현)
    
    Args:
        inha_csv_path: INHA.csv 파일 경로
        
    Returns:
        분할된 CSV 파일 경로
    """
    # INHA.csv 파일 읽기
    df_inha = pd.read_csv(inha_csv_path)
    
    # Split 열이 있는지 확인
    if 'Split' not in df_inha.columns:
        raise ValueError("INHA.csv 파일에 Split 열이 없습니다.")
    
    # Split 값 확인
    split_values = df_inha['Split'].unique()
    print(f"INHA.csv의 Split 값: {split_values}")
    
    # Split=1인 환자들은 그대로 train으로 사용
    train_df = df_inha[df_inha['Split'] == 1].copy()
    
    # Split=2인 환자들만 추출
    val_test_df = df_inha[df_inha['Split'] == 2].copy()
    
    # Split=2인 환자 ID 목록
    val_test_patient_ids = val_test_df['A_NUM'].unique()
    
    # 환자 ID 기준으로 1:1 분할 (검증:테스트)
    val_ids, test_inha_ids = train_test_split(val_test_patient_ids, test_size=0.5, random_state=42)
    
    # 검증용 데이터프레임 생성
    val_df = val_test_df[val_test_df['A_NUM'].isin(val_ids)].copy()
    val_df['Split'] = 2  # Split 값 유지
    
    # 테스트용 데이터프레임 생성
    test_inha_df = val_test_df[val_test_df['A_NUM'].isin(test_inha_ids)].copy()
    test_inha_df['Split'] = 3  # Split 값을 3으로 변경
    
    # 분할 정보 저장
    split_train_df = train_df.copy()
    split_val_df = val_df.copy()
    split_test_inha_df = test_inha_df.copy()
    
    # 결합된 데이터프레임 생성
    combined_df = pd.concat([split_train_df, split_val_df, split_test_inha_df])
    
    # 분할된 CSV 파일 저장
    split_csv_path = os.path.splitext(inha_csv_path)[0] + "_split.csv"
    combined_df.to_csv(split_csv_path, index=False)
    print(f"분할된 CSV 파일 저장: {split_csv_path}")
    
    # 환자 수 통계
    train_count = len(split_train_df['A_NUM'].unique())
    val_count = len(split_val_df['A_NUM'].unique())
    test_inha_count = len(split_test_inha_df['A_NUM'].unique())
    total_count = train_count + val_count + test_inha_count
    
    print(f"훈련 환자 수(Split=1): {train_count} ({train_count/total_count:.1%})")
    print(f"검증 환자 수(Split=2): {val_count} ({val_count/total_count:.1%})")
    print(f"테스트(INHA) 환자 수(Split=3): {test_inha_count} ({test_inha_count/total_count:.1%})")
    
    return split_csv_path