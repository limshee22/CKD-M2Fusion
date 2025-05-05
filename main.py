import argparse
from trainers import TrainingPipeline


def parse_args():
    """
    명령줄 인수 파싱 함수
    
    Returns:
        args: 파싱된 인수
    """
    parser = argparse.ArgumentParser(description='Kidney CT Slice Fusion Model Training')
    parser.add_argument('--train-csv', type=str, default='./Data/INHA_split_new.csv', help='학습 CSV 경로')
    parser.add_argument('--test-csv', type=str, default='./Data/KMUH.csv', help='테스트 CSV 경로')
    parser.add_argument('--pretrained-weights', type=str, default='model_final.pth', help='사전 학습된 DINOv2 가중치 경로')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    parser.add_argument('--target-slices', type=int, default=32, help='목표 신장 슬라이스 수')
    parser.add_argument('--epochs', type=int, default=50, help='학습 에포크 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='가중치 감쇠')
    parser.add_argument('--early-stopping', type=int, default=10, help='조기 종료 인내심')
    parser.add_argument('--log-interval', type=int, default=10, help='로깅 간격')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/fusion/32', help='체크포인트 저장 디렉토리')
    parser.add_argument('--log-dir', type=str, default='./logs', help='로그 저장 디렉토리')
    parser.add_argument('--results-dir', type=str, default='./results/fusion/32', help='결과 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--tabpfn-dim', type=int, default=64, help='TabPFN 특징 차원')
    
    return parser.parse_args()


def main():
    """
    메인 함수
    """
    # 인수 파싱
    args = parse_args()
    
    # 학습 파이프라인 생성 및 실행
    pipeline = TrainingPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()