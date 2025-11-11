"""
생육지표 예측 Flask 웹 애플리케이션 (2단계 버전)
1단계: 환경 데이터 → 생육지표 예측
2단계: 환경+생육 데이터 → 주당누적수확수 예측
"""

from flask import Flask, render_template, request, send_file, jsonify
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 제한
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# 예측할 생육지표 목록
TRAIN_TARGETS = [
    "개화마디", "착과마디", "수확마디", "착과수", "열매수", "초장",
    "생장길이", "엽수", "엽장", "엽폭", "줄기굵기", "화방높이"
]

# 수확수 예측 타겟
HARVEST_TARGET = "주당누적수확수"

KEY_FARM = "농가명"
KEY_DATE = "일시"

# ==================== 유틸리티 클래스 ====================
class SkewAwareScaler:
    """왜도 기반 스케일러"""
    def __init__(self, low_thr=0.3, high_thr=1.0, fillna="median", eps=1e-6):
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.fillna = fillna
        self.eps = eps
        self.meta = {}
        self._scalers = {}

    def transform(self, df, cols):
        """데이터 변환"""
        df = df.copy()
        if self.fillna == "median":
            for c in cols:
                df[c] = df[c].fillna(df[c].median())
        elif self.fillna == "mean":
            for c in cols:
                df[c] = df[c].fillna(df[c].mean())

        out = df.copy()
        for c in cols:
            info = self.meta[c]
            r = info["rule"]
            shift = info["shift"]
            sc = self._scalers[c]
            x = out[c].astype(float).values.copy()

            if r == "constant":
                out[c] = 0.0
                continue
            if r == "log_robust":
                x = np.log1p(x + shift)
            if sc is not None:
                x = sc.transform(x.reshape(-1, 1)).ravel()
            out[c] = x
        return out

    def inverse_transform(self, df_scaled, cols):
        """역변환"""
        out = df_scaled.copy()
        for c in cols:
            info = self.meta[c]
            r = info["rule"]
            shift = info["shift"]
            sc = self._scalers[c]
            x = out[c].astype(float).values.copy()

            if r != "constant" and sc is not None:
                x = sc.inverse_transform(x.reshape(-1, 1)).ravel()
            if r == "log_robust":
                x = np.expm1(x) - shift
            out[c] = x
        return out

# ==================== 모델 정의 ====================
class GRURegVar(nn.Module):
    """가변 길이 시퀀스를 처리하는 GRU 회귀 모델"""
    def __init__(self, in_dim, hidden=96, layers=2, out_dim=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x, lens):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        packed = pack_padded_sequence(
            x,
            lengths=lens.cpu(),
            batch_first=True,
            enforce_sorted=True
        )
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)

        # 마지막 타임스텝 추출
        idx = (lens - 1).view(-1, 1, 1).expand(out.size(0), 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)

        return self.fc(last)

# ==================== 핵심 함수 ====================
def sanitize_datetime_col(df, date_col):
    """날짜 컬럼 정제"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[date_col])
    after = len(df)
    if before != after:
        print(f"[sanitize] dropped {before-after} rows with NaT in '{date_col}'")
    return df

def load_model_and_scalers(target, model_root):
    """특정 타겟에 대한 모델, 스케일러, 메타데이터 로드"""
    target_dir = os.path.join(model_root, target)
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {target_dir}")

    # 메타데이터 로드
    meta_path = os.path.join(target_dir, "model_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 스케일러 로드
    scaler_x = joblib.load(os.path.join(target_dir, "scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(target_dir, "scaler_y.pkl"))

    # 하이퍼파라미터
    hp = meta["hparams_best"]
    feature_cols = meta["features"]

    # 모델 초기화 및 가중치 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = os.path.join(target_dir, "best_model.pt")

    # 먼저 저장된 모델 파일을 로드해서 구조 파악
    checkpoint = torch.load(weight_path, map_location=device)

    # GRU 가중치에서 hidden size와 layer 수 추출
    try:
        # gru.weight_hh_l0의 shape에서 hidden size 추출
        hidden_size = checkpoint['gru.weight_hh_l0'].shape[1]

        # 레이어 수 계산 (gru.weight_ih_l{N} 중 가장 큰 N + 1)
        num_layers = max([int(k.split('_l')[1].split('.')[0]) for k in checkpoint.keys()
                         if 'gru.weight_ih_l' in k]) + 1

        print(f"[{target}] 모델에서 추출한 구조: hidden={hidden_size}, layers={num_layers}")

    except Exception as e:
        # 추출 실패 시 meta 파일의 값 사용
        print(f"[{target}] 모델 구조 추출 실패, meta 파일 사용: {str(e)}")
        hidden_size = hp.get("HIDDEN", 96)
        num_layers = hp.get("LAYERS", 2)

    # dropout 값은 meta 파일에서
    dropout = hp.get("DROPOUT", 0.2)

    model = GRURegVar(
        in_dim=len(feature_cols),
        hidden=hidden_size,
        layers=num_layers,
        out_dim=1,
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint)
    model.eval()

    return model, scaler_x, scaler_y, hp, feature_cols, device

def predict_daily_values(df_env, target, model, scaler_x, scaler_y, hp, feature_cols, device):
    """일별 생육지표 예측"""
    # 환경 데이터 스케일링
    df_env_scaled = df_env.copy()
    df_env_scaled[feature_cols] = scaler_x.transform(df_env[feature_cols], feature_cols)

    min_days = hp["MIN_DAYS"]
    max_days = hp["MAX_DAYS"]

    predictions = []

    # 농가별로 그룹화
    for farm, group in df_env_scaled.groupby(KEY_FARM):
        group = group.sort_values(KEY_DATE).set_index(KEY_DATE)

        # 각 날짜에 대해 예측
        for current_date in group.index:
            # 윈도우 설정 (과거 max_days ~ 1일 전까지)
            start_date = current_date - pd.Timedelta(days=max_days)
            end_date = current_date - pd.Timedelta(days=1)

            if end_date < start_date:
                continue

            # 날짜 범위에 해당하는 데이터 추출
            date_range = pd.date_range(start_date, end_date, freq="D")
            available_dates = group.index.intersection(date_range)

            # 최소 일수 체크
            if len(available_dates) < min_days:
                continue

            # 시퀀스 데이터 추출
            sequence = group.loc[available_dates, feature_cols]

            # 결측치 체크
            if sequence.isna().any().any():
                continue

            # 최대 길이 제한
            if len(sequence) > max_days:
                sequence = sequence.iloc[-max_days:]

            # 텐서 변환
            x = torch.tensor(sequence.values, dtype=torch.float32).unsqueeze(0).to(device)
            lens = torch.tensor([len(sequence)], dtype=torch.long).to(device)

            # 예측
            with torch.no_grad():
                y_hat = model(x, lens).cpu().numpy().reshape(-1)

            # 역스케일링
            pred_value = scaler_y.inverse_transform(
                pd.DataFrame({target: [y_hat[0]]}),
                [target]
            )[target].iloc[0]

            predictions.append({
                KEY_FARM: farm,
                KEY_DATE: current_date,
                target: float(pred_value)
            })

    return pd.DataFrame(predictions)

def process_prediction(env_file_path, growth_file_path, model_root):
    """1단계: 환경 데이터로 생육지표 예측"""
    results = {
        'status': 'processing',
        'message': '',
        'progress': 0
    }

    try:
        # 환경 데이터 로드 (필수)
        df_env = pd.read_excel(env_file_path)
        df_env = sanitize_datetime_col(df_env, KEY_DATE)

        results['message'] = f'환경 데이터 로드: {df_env.shape[0]}행'
        results['progress'] = 5

        # 생육지표 실측값 로드 (선택)
        df_measured = None
        has_measured = False
        if growth_file_path and os.path.exists(growth_file_path):
            df_measured = pd.read_excel(growth_file_path)
            df_measured = sanitize_datetime_col(df_measured, KEY_DATE)
            has_measured = True
            results['message'] = f'실측값 로드: {df_measured.shape[0]}행'
            results['progress'] = 10

        # 각 타겟별 예측
        all_predictions = []
        total_targets = len(TRAIN_TARGETS)

        for idx, target in enumerate(TRAIN_TARGETS):
            try:
                # 모델 로드
                model, scaler_x, scaler_y, hp, feature_cols, device = load_model_and_scalers(
                    target, model_root
                )

                # 예측 수행
                df_pred = predict_daily_values(
                    df_env, target, model, scaler_x, scaler_y, hp, feature_cols, device
                )

                all_predictions.append(df_pred)

                base_progress = 10 if has_measured else 5
                progress_range = 70 if has_measured else 85
                progress = base_progress + int((idx + 1) / total_targets * progress_range)
                results['progress'] = progress
                results['message'] = f'{target} 예측 완료 ({idx+1}/{total_targets})'

            except Exception as e:
                print(f"타겟 {target} 예측 중 오류: {str(e)}")
                continue

        if len(all_predictions) == 0:
            raise Exception("예측 결과가 없습니다. 모델 파일을 확인해주세요.")

        results['message'] = '예측 결과 병합 중...'
        results['progress'] = 85

        # 모든 예측 결과 병합
        merged_pred = all_predictions[0]
        for df in all_predictions[1:]:
            merged_pred = merged_pred.merge(df, on=[KEY_FARM, KEY_DATE], how="outer")

        # 환경 데이터와 예측 결과 병합
        final_result = df_env.merge(merged_pred, on=[KEY_FARM, KEY_DATE], how="left")

        # 실측값이 있는 경우 병합
        filled_count = {}
        if has_measured and df_measured is not None:
            results['message'] = '실측값과 병합 중...'
            results['progress'] = 90

            for target in TRAIN_TARGETS:
                if target in df_measured.columns:
                    measured_data = df_measured[[KEY_FARM, KEY_DATE, target]].copy()
                    final_result = final_result.merge(measured_data, on=[KEY_FARM, KEY_DATE], how="left", suffixes=('_pred', ''))

                    # 실측값이 있으면 실측값 사용, 없으면 예측값 사용
                    if f'{target}_pred' in final_result.columns:
                        final_result[target] = final_result[target].fillna(final_result[f'{target}_pred'])
                        final_result.drop(columns=[f'{target}_pred'], inplace=True)

                    measured_cnt = final_result[target].notna().sum()
                    filled_count[target] = {'total': measured_cnt}

        # 컬럼 정리
        base_cols = [KEY_FARM, KEY_DATE]
        env_cols = [col for col in df_env.columns if col not in base_cols]
        target_cols = [col for col in TRAIN_TARGETS if col in final_result.columns]

        ordered_cols = base_cols + env_cols + target_cols
        ordered_cols = [col for col in ordered_cols if col in final_result.columns]
        final_result = final_result[ordered_cols]

        # 정렬
        final_result = final_result.sort_values([KEY_FARM, KEY_DATE])

        results['message'] = '결과 파일 생성 중...'
        results['progress'] = 95

        # 엑셀 파일로 저장
        output_filename = f'step1_growth_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        final_result.to_excel(output_path, index=False, engine='openpyxl')

        results['status'] = 'success'
        results['message'] = '예측 완료!'
        results['progress'] = 100
        results['output_file'] = output_filename
        results['row_count'] = len(final_result)
        results['has_measured'] = has_measured

    except Exception as e:
        results['status'] = 'error'
        results['message'] = f'오류 발생: {str(e)}'
        results['progress'] = 0

    return results

def process_harvest_prediction(file_path, model_root):
    """2단계: 환경+생육 데이터로 주당누적수확수 예측"""
    results = {
        'status': 'processing',
        'message': '',
        'progress': 0
    }

    try:
        # 데이터 로드
        df = pd.read_excel(file_path)
        df = sanitize_datetime_col(df, KEY_DATE)

        results['message'] = f'데이터 로드: {df.shape[0]}행, {df.shape[1]}열'
        results['progress'] = 10

        original_rows = len(df)

        # 결측치 제거
        df_clean = df.dropna()
        removed_rows = original_rows - len(df_clean)

        results['message'] = f'결측치 제거: {removed_rows}행 제거됨'
        results['progress'] = 20

        if len(df_clean) == 0:
            raise Exception("결측치 제거 후 데이터가 없습니다.")

        # 주당누적수확수 모델 로드
        results['message'] = '수확수 예측 모델 로드 중...'
        results['progress'] = 30

        model, scaler_x, scaler_y, hp, feature_cols, device = load_model_and_scalers(
            HARVEST_TARGET, model_root
        )

        results['message'] = '수확수 예측 수행 중...'
        results['progress'] = 50

        # 예측 수행
        df_harvest_pred = predict_daily_values(
            df_clean, HARVEST_TARGET, model, scaler_x, scaler_y, hp, feature_cols, device
        )

        # 결과 병합
        results['message'] = '결과 병합 중...'
        results['progress'] = 80

        final_result = df_clean.merge(
            df_harvest_pred,
            on=[KEY_FARM, KEY_DATE],
            how="left"
        )

        # 정렬
        final_result = final_result.sort_values([KEY_FARM, KEY_DATE])

        results['message'] = '결과 파일 생성 중...'
        results['progress'] = 90

        # 엑셀 파일로 저장
        output_filename = f'step2_harvest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        final_result.to_excel(output_path, index=False, engine='openpyxl')

        results['status'] = 'success'
        results['message'] = '예측 완료!'
        results['progress'] = 100
        results['output_file'] = output_filename
        results['row_count'] = len(final_result)
        results['removed_rows'] = removed_rows
        results['predicted_rows'] = len(df_harvest_pred)

    except Exception as e:
        results['status'] = 'error'
        results['message'] = f'오류 발생: {str(e)}'
        results['progress'] = 0

    return results

# ==================== 원클릭 수확수 예측 ====================
def process_direct_harvest(env_file_path, growth_file_path, model_root):
    """원클릭: 환경 데이터 → 생육지표 예측 → 수확수 예측"""
    results = {
        'status': 'processing',
        'message': '',
        'progress': 0
    }

    try:
        # Step 1: 환경 데이터 로드
        df_env = pd.read_excel(env_file_path)
        df_env = sanitize_datetime_col(df_env, KEY_DATE)

        results['message'] = f'환경 데이터 로드: {df_env.shape[0]}행'
        results['progress'] = 5

        # Step 2: 생육지표 실측값 로드 (선택)
        df_measured = None
        if growth_file_path and os.path.exists(growth_file_path):
            df_measured = pd.read_excel(growth_file_path)
            df_measured = sanitize_datetime_col(df_measured, KEY_DATE)
            results['message'] = f'실측값 로드: {df_measured.shape[0]}행'
            results['progress'] = 10

        # Step 3: 생육지표 예측
        all_predictions = []
        total_targets = len(TRAIN_TARGETS)

        for idx, target in enumerate(TRAIN_TARGETS):
            try:
                model, scaler_x, scaler_y, hp, feature_cols, device = load_model_and_scalers(
                    target, model_root
                )

                df_pred = predict_daily_values(
                    df_env, target, model, scaler_x, scaler_y, hp, feature_cols, device
                )

                all_predictions.append(df_pred)

                progress = 10 + int((idx + 1) / total_targets * 40)
                results['progress'] = progress
                results['message'] = f'생육지표 예측 중... {target} ({idx+1}/{total_targets})'

            except Exception as e:
                print(f"타겟 {target} 예측 중 오류: {str(e)}")
                continue

        if len(all_predictions) == 0:
            raise Exception("생육지표 예측 실패. 모델 파일을 확인해주세요.")

        results['message'] = '생육지표 예측 결과 병합 중...'
        results['progress'] = 50

        # 모든 예측 결과 병합
        merged_pred = all_predictions[0]
        for df in all_predictions[1:]:
            merged_pred = merged_pred.merge(df, on=[KEY_FARM, KEY_DATE], how="outer")

        # 환경 데이터와 예측 결과 병합
        df_combined = df_env.merge(merged_pred, on=[KEY_FARM, KEY_DATE], how="left")

        # 실측값이 있는 경우 병합
        if df_measured is not None:
            results['message'] = '실측값과 병합 중...'
            results['progress'] = 55

            for target in TRAIN_TARGETS:
                if target in df_measured.columns:
                    measured_data = df_measured[[KEY_FARM, KEY_DATE, target]].copy()
                    df_combined = df_combined.merge(measured_data, on=[KEY_FARM, KEY_DATE], how="left", suffixes=('_pred', ''))

                    if f'{target}_pred' in df_combined.columns:
                        df_combined[target] = df_combined[target].fillna(df_combined[f'{target}_pred'])
                        df_combined.drop(columns=[f'{target}_pred'], inplace=True)

        results['message'] = '결측치 제거 중...'
        results['progress'] = 60

        # 결측치 제거
        original_rows = len(df_combined)
        df_clean = df_combined.dropna()
        removed_rows = original_rows - len(df_clean)

        if len(df_clean) == 0:
            raise Exception(f"결측치 제거 후 데이터가 없습니다. (제거된 행: {removed_rows}개)")

        results['message'] = f'결측치 {removed_rows}행 제거 완료'
        results['progress'] = 70

        # Step 4: 주당누적수확수 모델 로드
        results['message'] = '수확수 예측 모델 로드 중...'
        results['progress'] = 75

        model, scaler_x, scaler_y, hp, feature_cols, device = load_model_and_scalers(
            HARVEST_TARGET, model_root
        )

        results['message'] = '수확수 예측 수행 중...'
        results['progress'] = 80

        # Step 5: 수확수 예측
        df_harvest_pred = predict_daily_values(
            df_clean, HARVEST_TARGET, model, scaler_x, scaler_y, hp, feature_cols, device
        )

        results['message'] = '최종 결과 병합 중...'
        results['progress'] = 90

        # 최종 결과 병합
        final_result = df_clean.merge(
            df_harvest_pred,
            on=[KEY_FARM, KEY_DATE],
            how="left"
        )

        # 정렬
        final_result = final_result.sort_values([KEY_FARM, KEY_DATE])

        results['message'] = '주별 집계 중...'
        results['progress'] = 92

        # 주별로 집계 (월요일 기준)
        weekly_results = []

        for farm in final_result[KEY_FARM].unique():
            farm_data = final_result[final_result[KEY_FARM] == farm].copy()

            # 주차 계산 (ISO 주차 기준)
            farm_data['년도'] = farm_data[KEY_DATE].dt.isocalendar().year
            farm_data['주차'] = farm_data[KEY_DATE].dt.isocalendar().week

            # 주별 그룹화
            for (year, week), group in farm_data.groupby(['년도', '주차']):
                # 주당누적수확수의 최대/최소
                if HARVEST_TARGET in group.columns:
                    max_harvest = group[HARVEST_TARGET].max()*2
                    min_harvest = group[HARVEST_TARGET].min()*2
                    mean_harvest = group[HARVEST_TARGET].mean()*2

                    weekly_results.append({
                        KEY_FARM: farm,
                        '년도': int(year),
                        '주차': int(week),
                        #f'{HARVEST_TARGET}_최대': max_harvest,
                        #f'{HARVEST_TARGET}_최소': min_harvest,
                        f'{HARVEST_TARGET}' : mean_harvest
                    })

        results['message'] = '최종 결과 생성 중...'
        results['progress'] = 95

        # 주별 결과 데이터프레임 생성
        df_weekly = pd.DataFrame(weekly_results)
        df_weekly = df_weekly.sort_values([KEY_FARM, '년도', '주차'])

        # 엑셀 파일로 저장
        output_filename = f'harvest_direct_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        df_weekly.to_excel(output_path, index=False, engine='openpyxl')

        results['status'] = 'success'
        results['message'] = '예측 완료!'
        results['progress'] = 100
        results['output_file'] = output_filename
        results['row_count'] = len(df_weekly)
        results['removed_rows'] = removed_rows
        results['predicted_rows'] = len(df_harvest_pred)
        results['weekly_count'] = len(df_weekly)

    except Exception as e:
        results['status'] = 'error'
        results['message'] = f'오류 발생: {str(e)}'
        results['progress'] = 0

    return results

# ==================== Flask 라우트 ====================
@app.route('/')
def index():
    """메인 페이지: 원클릭 수확수 예측"""
    return render_template('direct.html')

@app.route('/step1')
def step1_index():
    """1단계: 생육지표 예측 페이지"""
    return render_template('index.html')

@app.route('/predict_direct', methods=['POST'])
def predict_direct():
    """원클릭: 환경 데이터로 바로 수확수 예측"""
    if 'env_file' not in request.files:
        return jsonify({'status': 'error', 'message': '환경 데이터 파일이 없습니다.'})

    env_file = request.files['env_file']

    if env_file.filename == '':
        return jsonify({'status': 'error', 'message': '환경 데이터 파일이 선택되지 않았습니다.'})

    if not env_file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'status': 'error', 'message': 'Excel 파일만 업로드 가능합니다.'})

    try:
        # 환경 데이터 파일 저장
        env_filename = secure_filename(env_file.filename)
        env_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'env_{env_filename}')
        env_file.save(env_filepath)

        # 생육지표 파일 처리 (선택사항)
        growth_filepath = None
        if 'growth_file' in request.files:
            growth_file = request.files['growth_file']
            if growth_file.filename != '':
                if not growth_file.filename.endswith(('.xlsx', '.xls')):
                    if os.path.exists(env_filepath):
                        os.remove(env_filepath)
                    return jsonify({'status': 'error', 'message': '생육지표 파일도 Excel 형식이어야 합니다.'})

                growth_filename = secure_filename(growth_file.filename)
                growth_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'growth_{growth_filename}')
                growth_file.save(growth_filepath)

        # 원클릭 예측 수행
        results = process_direct_harvest(env_filepath, growth_filepath, app.config['MODEL_FOLDER'])

        # 업로드 파일 삭제
        if os.path.exists(env_filepath):
            os.remove(env_filepath)
        if growth_filepath and os.path.exists(growth_filepath):
            os.remove(growth_filepath)

        return jsonify(results)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'처리 중 오류: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """1단계: 파일 업로드 및 생육지표 예측"""
    if 'env_file' not in request.files:
        return jsonify({'status': 'error', 'message': '환경 데이터 파일이 없습니다.'})

    env_file = request.files['env_file']

    if env_file.filename == '':
        return jsonify({'status': 'error', 'message': '환경 데이터 파일이 선택되지 않았습니다.'})

    if not env_file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'status': 'error', 'message': 'Excel 파일만 업로드 가능합니다.'})

    try:
        # 환경 데이터 파일 저장
        env_filename = secure_filename(env_file.filename)
        env_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'env_{env_filename}')
        env_file.save(env_filepath)

        # 생육지표 파일 처리 (선택사항)
        growth_filepath = None
        if 'growth_file' in request.files:
            growth_file = request.files['growth_file']
            if growth_file.filename != '':
                if not growth_file.filename.endswith(('.xlsx', '.xls')):
                    if os.path.exists(env_filepath):
                        os.remove(env_filepath)
                    return jsonify({'status': 'error', 'message': '생육지표 파일도 Excel 형식이어야 합니다.'})

                growth_filename = secure_filename(growth_file.filename)
                growth_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'growth_{growth_filename}')
                growth_file.save(growth_filepath)

        # 예측 수행
        results = process_prediction(env_filepath, growth_filepath, app.config['MODEL_FOLDER'])

        # 업로드 파일 삭제
        if os.path.exists(env_filepath):
            os.remove(env_filepath)
        if growth_filepath and os.path.exists(growth_filepath):
            os.remove(growth_filepath)

        return jsonify(results)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'처리 중 오류: {str(e)}'})

@app.route('/harvest')
@app.route('/step2')
def harvest_index():
    """2단계: 수확수 예측 페이지"""
    return render_template('harvest.html')

@app.route('/predict_harvest', methods=['POST'])
def predict_harvest():
    """2단계: 환경+생육 데이터로 주당누적수확수 예측"""
    if 'combined_file' not in request.files:
        return jsonify({'status': 'error', 'message': '파일이 없습니다.'})

    file = request.files['combined_file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': '파일이 선택되지 않았습니다.'})

    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'status': 'error', 'message': 'Excel 파일만 업로드 가능합니다.'})

    try:
        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'combined_{filename}')
        file.save(filepath)

        # 예측 수행
        results = process_harvest_prediction(filepath, app.config['MODEL_FOLDER'])

        # 업로드 파일 삭제
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(results)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'처리 중 오류: {str(e)}'})

@app.route('/download/<path:filename>')
def download_file(filename):
    """결과 파일 다운로드"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            # 파일명에 따라 다운로드 이름 지정
            if filename.startswith('step1'):
                download_name = '생육지표_예측결과.xlsx'
            elif filename.startswith('step2'):
                download_name = '주당누적수확수_예측결과.xlsx'
            elif filename.startswith('harvest_direct'):
                download_name = '주별_수확수_예측결과(최대최소).xlsx'
            else:
                download_name = filename
            
            return send_file(filepath, as_attachment=True, download_name=download_name)
        else:
            return jsonify({'status': 'error', 'message': '파일을 찾을 수 없습니다.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'다운로드 오류: {str(e)}'})

if __name__ == '__main__':
    print("=" * 60)
    print("생육지표 + 수확수 예측 웹 애플리케이션")
    print("=" * 60)
    print(f"\n원클릭: http://localhost:5000 (환경 → 수확수 직접 예측)")
    print(f"1단계: http://localhost:5000/step1 (생육지표만 예측)")
    print(f"2단계: http://localhost:5000/step2 (수확수만 예측)")
    print(f"\n모델 폴더: {os.path.abspath(app.config['MODEL_FOLDER'])}")
    print(f"\n종료하려면 Ctrl+C를 누르세요.\n")
    app.run(debug=True, host='0.0.0.0', port=5000)