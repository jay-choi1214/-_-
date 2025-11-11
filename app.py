"""
온실 생육지표 및 수확량 예측 Streamlit 애플리케이션
- 원클릭: 환경 데이터 → 생육지표 → 수확량 예측
- 1단계: 환경 데이터 → 생육지표 예측
- 2단계: 환경+생육 데이터 → 수확량 예측
"""

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from datetime import datetime
import io

# 페이지 설정
st.set_page_config(
    page_title="온실 생육·수확 예측 시스템",
    page_icon="🌱",
    layout="wide"
)

# 상수 정의
MODEL_FOLDER = "models"
KEY_FARM = "농가명"
KEY_DATE = "일시"

TRAIN_TARGETS = [
    "개화마디", "착과마디", "수확마디", "착과수", "열매수", "초장",
    "생장길이", "엽수", "엽장", "엽폭", "줄기굵기", "화방높이"
]
HARVEST_TARGET = "주당누적수확수"

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

# ==================== 모델 클래스 ====================
class GRURegVar(nn.Module):
    """가변 길이 시퀀스 GRU 모델"""
    def __init__(self, in_dim, hidden=96, layers=2, out_dim=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x, lens):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        
        packed = pack_padded_sequence(x, lengths=lens.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        
        idx = (lens - 1).view(-1, 1, 1).expand(out.size(0), 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)
        
        return self.fc(last)

# ==================== 캐싱된 함수들 ====================
@st.cache_data
def sanitize_datetime_col(df, date_col):
    """날짜 컬럼 정제"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    return df

@st.cache_resource
def load_model_and_scalers(target, model_root):
    """모델 및 스케일러 로드 (캐싱)"""
    target_dir = os.path.join(model_root, target)
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {target_dir}")

    # 메타데이터
    with open(os.path.join(target_dir, "model_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 스케일러
    scaler_x = joblib.load(os.path.join(target_dir, "scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(target_dir, "scaler_y.pkl"))

    hp = meta["hparams_best"]
    feature_cols = meta["features"]

    # 모델
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = os.path.join(target_dir, "best_model.pt")
    checkpoint = torch.load(weight_path, map_location=device)

    try:
        hidden_size = checkpoint['gru.weight_hh_l0'].shape[1]
        num_layers = max([int(k.split('_l')[1].split('.')[0]) for k in checkpoint.keys()
                         if 'gru.weight_ih_l' in k]) + 1
    except:
        hidden_size = hp.get("HIDDEN", 96)
        num_layers = hp.get("LAYERS", 2)

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
    """일별 예측"""
    df_env_scaled = df_env.copy()
    df_env_scaled[feature_cols] = scaler_x.transform(df_env[feature_cols], feature_cols)

    min_days = hp["MIN_DAYS"]
    max_days = hp["MAX_DAYS"]
    predictions = []

    for farm, group in df_env_scaled.groupby(KEY_FARM):
        group = group.sort_values(KEY_DATE).set_index(KEY_DATE)

        for current_date in group.index:
            start_date = current_date - pd.Timedelta(days=max_days)
            end_date = current_date - pd.Timedelta(days=1)

            if end_date < start_date:
                continue

            date_range = pd.date_range(start_date, end_date, freq="D")
            available_dates = group.index.intersection(date_range)

            if len(available_dates) < min_days:
                continue

            sequence = group.loc[available_dates, feature_cols]

            if sequence.isna().any().any():
                continue

            if len(sequence) > max_days:
                sequence = sequence.iloc[-max_days:]

            x = torch.tensor(sequence.values, dtype=torch.float32).unsqueeze(0).to(device)
            lens = torch.tensor([len(sequence)], dtype=torch.long).to(device)

            with torch.no_grad():
                y_hat = model(x, lens).cpu().numpy().reshape(-1)

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

# ==================== 예측 함수들 ====================
def process_step1(df_env, df_measured=None):
    """1단계: 생육지표 예측"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_predictions = []
    total = len(TRAIN_TARGETS)

    for idx, target in enumerate(TRAIN_TARGETS):
        try:
            status_text.text(f"예측 중: {target} ({idx+1}/{total})")
            
            model, scaler_x, scaler_y, hp, feature_cols, device = load_model_and_scalers(target, MODEL_FOLDER)
            df_pred = predict_daily_values(df_env, target, model, scaler_x, scaler_y, hp, feature_cols, device)
            all_predictions.append(df_pred)
            
            progress_bar.progress((idx + 1) / total)
        except Exception as e:
            st.warning(f"{target} 예측 실패: {str(e)}")
            continue

    if len(all_predictions) == 0:
        raise Exception("모든 예측이 실패했습니다.")

    status_text.text("결과 병합 중...")
    
    # 병합
    merged = all_predictions[0]
    for df in all_predictions[1:]:
        merged = merged.merge(df, on=[KEY_FARM, KEY_DATE], how="outer")

    result = df_env.merge(merged, on=[KEY_FARM, KEY_DATE], how="left")

    # 실측값 병합
    if df_measured is not None:
        for target in TRAIN_TARGETS:
            if target in df_measured.columns:
                measured_data = df_measured[[KEY_FARM, KEY_DATE, target]].copy()
                result = result.merge(measured_data, on=[KEY_FARM, KEY_DATE], how="left", suffixes=('_pred', ''))
                
                if f'{target}_pred' in result.columns:
                    result[target] = result[target].fillna(result[f'{target}_pred'])
                    result.drop(columns=[f'{target}_pred'], inplace=True)

    result = result.sort_values([KEY_FARM, KEY_DATE])
    
    progress_bar.progress(1.0)
    status_text.text("완료!")
    
    return result

def process_step2(df_combined):
    """2단계: 수확량 예측"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("결측치 제거 중...")
    original_rows = len(df_combined)
    df_clean = df_combined.dropna()
    removed = original_rows - len(df_clean)
    
    if len(df_clean) == 0:
        raise Exception("결측치 제거 후 데이터가 없습니다.")
    
    st.info(f"결측치 {removed}행 제거됨")
    progress_bar.progress(0.3)
    
    status_text.text("수확량 예측 중...")
    model, scaler_x, scaler_y, hp, feature_cols, device = load_model_and_scalers(HARVEST_TARGET, MODEL_FOLDER)
    
    progress_bar.progress(0.6)
    df_harvest = predict_daily_values(df_clean, HARVEST_TARGET, model, scaler_x, scaler_y, hp, feature_cols, device)
    
    status_text.text("결과 병합 중...")
    result = df_clean.merge(df_harvest, on=[KEY_FARM, KEY_DATE], how="left")
    result = result.sort_values([KEY_FARM, KEY_DATE])
    
    progress_bar.progress(1.0)
    status_text.text("완료!")
    
    return result, removed

def process_direct(df_env, df_measured=None):
    """환경 → 생육 → 수확"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: 생육지표
    status_text.text("생육지표 예측 중...")
    all_predictions = []
    total = len(TRAIN_TARGETS)

    for idx, target in enumerate(TRAIN_TARGETS):
        try:
            model, scaler_x, scaler_y, hp, feature_cols, device = load_model_and_scalers(target, MODEL_FOLDER)
            df_pred = predict_daily_values(df_env, target, model, scaler_x, scaler_y, hp, feature_cols, device)
            all_predictions.append(df_pred)
            progress_bar.progress(0.5 * (idx + 1) / total)
        except:
            continue

    merged = all_predictions[0]
    for df in all_predictions[1:]:
        merged = merged.merge(df, on=[KEY_FARM, KEY_DATE], how="outer")

    df_combined = df_env.merge(merged, on=[KEY_FARM, KEY_DATE], how="left")

    # 실측값 병합
    if df_measured is not None:
        for target in TRAIN_TARGETS:
            if target in df_measured.columns:
                measured_data = df_measured[[KEY_FARM, KEY_DATE, target]].copy()
                df_combined = df_combined.merge(measured_data, on=[KEY_FARM, KEY_DATE], how="left", suffixes=('_pred', ''))
                if f'{target}_pred' in df_combined.columns:
                    df_combined[target] = df_combined[target].fillna(df_combined[f'{target}_pred'])
                    df_combined.drop(columns=[f'{target}_pred'], inplace=True)

    status_text.text("결측치 제거 중...")
    progress_bar.progress(0.6)
    
    original_rows = len(df_combined)
    df_clean = df_combined.dropna()
    removed = original_rows - len(df_clean)
    
    # Step 2: 수확량
    status_text.text("수확량 예측 중...")
    progress_bar.progress(0.75)
    
    model, scaler_x, scaler_y, hp, feature_cols, device = load_model_and_scalers(HARVEST_TARGET, MODEL_FOLDER)
    df_harvest = predict_daily_values(df_clean, HARVEST_TARGET, model, scaler_x, scaler_y, hp, feature_cols, device)
    
    status_text.text("최종 결과 생성 중...")
    result = df_clean.merge(df_harvest, on=[KEY_FARM, KEY_DATE], how="left")
    
    # 주별 집계
    weekly_results = []
    for farm in result[KEY_FARM].unique():
        farm_data = result[result[KEY_FARM] == farm].copy()
        farm_data['년도'] = farm_data[KEY_DATE].dt.isocalendar().year
        farm_data['주차'] = farm_data[KEY_DATE].dt.isocalendar().week
        
        for (year, week), group in farm_data.groupby(['년도', '주차']):
            if HARVEST_TARGET in group.columns:
                mean_harvest = group[HARVEST_TARGET].mean() * 2
                weekly_results.append({
                    KEY_FARM: farm,
                    '년도': int(year),
                    '주차': int(week),
                    HARVEST_TARGET: mean_harvest
                })
    
    df_weekly = pd.DataFrame(weekly_results).sort_values([KEY_FARM, '년도', '주차'])
    
    progress_bar.progress(1.0)
    status_text.text("완료!")
    
    return df_weekly, removed

# ==================== Streamlit UI ====================
st.title("🌱 온실 생육·수확 예측 시스템")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["🚀 예측", "📊 1단계: 생육지표", "🌾 2단계: 수확량"])

# ==================== 원클릭 탭 ====================
with tab1:
    st.header("수확량 예측")
    st.info("환경 데이터만으로 생육지표를 먼저 예측한 후, 수확량까지 한 번에 예측합니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        env_file = st.file_uploader("📁 환경 데이터 (필수)", type=['xlsx', 'xls'], key="direct_env")
    
    with col2:
        growth_file = st.file_uploader("📁 생육지표 실측값 (선택)", type=['xlsx', 'xls'], key="direct_growth")
    
    if st.button("🚀 예측 시작", type="primary", use_container_width=True):
        if env_file is None:
            st.error("환경 데이터 파일을 업로드해주세요.")
        else:
            try:
                with st.spinner("데이터 로드 중..."):
                    df_env = pd.read_excel(env_file)
                    df_env = sanitize_datetime_col(df_env, KEY_DATE)
                    
                    df_measured = None
                    if growth_file:
                        df_measured = pd.read_excel(growth_file)
                        df_measured = sanitize_datetime_col(df_measured, KEY_DATE)
                
                st.success(f"환경 데이터: {len(df_env)}행")
                
                df_result, removed = process_direct(df_env, df_measured)
                
                st.success(f"✅ 예측 완료! (결측치 {removed}행 제거)")
                
                st.subheader("📊 주별 수확량 예측 결과")
                st.dataframe(df_result, use_container_width=True, height=400)
                
                # 통계
                col1, col2, col3 = st.columns(3)
                col1.metric("총 주차 수", len(df_result))
                col2.metric("평균 수확량", f"{df_result[HARVEST_TARGET].mean():.2f}")
                col3.metric("최대 수확량", f"{df_result[HARVEST_TARGET].max():.2f}")
                
                # 다운로드
                output = io.BytesIO()
                df_result.to_excel(output, index=False, engine='openpyxl')
                output.seek(0)
                
                st.download_button(
                    label="📥 결과 다운로드 (Excel)",
                    data=output,
                    file_name=f"주별수확량_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

# ==================== 1단계 탭 ====================
with tab2:
    st.header("1단계: 생육지표 예측")
    st.info("환경 데이터를 입력하여 12가지 생육지표를 예측합니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        env_file_s1 = st.file_uploader("📁 환경 데이터 (필수)", type=['xlsx', 'xls'], key="step1_env")
    
    with col2:
        growth_file_s1 = st.file_uploader("📁 생육지표 실측값 (선택)", type=['xlsx', 'xls'], key="step1_growth")
    
    if st.button("📊 생육지표 예측", type="primary", use_container_width=True):
        if env_file_s1 is None:
            st.error("환경 데이터 파일을 업로드해주세요.")
        else:
            try:
                with st.spinner("데이터 로드 중..."):
                    df_env = pd.read_excel(env_file_s1)
                    df_env = sanitize_datetime_col(df_env, KEY_DATE)
                    
                    df_measured = None
                    if growth_file_s1:
                        df_measured = pd.read_excel(growth_file_s1)
                        df_measured = sanitize_datetime_col(df_measured, KEY_DATE)
                
                df_result = process_step1(df_env, df_measured)
                
                st.success("✅ 예측 완료!")
                
                st.subheader("📊 생육지표 예측 결과")
                st.dataframe(df_result, use_container_width=True, height=400)
                
                # 다운로드
                output = io.BytesIO()
                df_result.to_excel(output, index=False, engine='openpyxl')
                output.seek(0)
                
                st.download_button(
                    label="📥 결과 다운로드 (Excel)",
                    data=output,
                    file_name=f"생육지표_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

# ==================== 2단계 탭 ====================
with tab3:
    st.header("2단계: 수확량 예측")
    st.info("환경 + 생육지표 데이터를 입력하여 수확량을 예측합니다.")
    
    combined_file = st.file_uploader("📁 환경+생육 통합 데이터", type=['xlsx', 'xls'], key="step2_combined")
    
    if st.button("🌾 수확량 예측", type="primary", use_container_width=True):
        if combined_file is None:
            st.error("통합 데이터 파일을 업로드해주세요.")
        else:
            try:
                with st.spinner("데이터 로드 중..."):
                    df_combined = pd.read_excel(combined_file)
                    df_combined = sanitize_datetime_col(df_combined, KEY_DATE)
                
                df_result, removed = process_step2(df_combined)
                
                st.success(f"✅ 예측 완료! (결측치 {removed}행 제거)")
                
                st.subheader("📊 수확량 예측 결과")
                st.dataframe(df_result, use_container_width=True, height=400)
                
                # 통계
                if HARVEST_TARGET in df_result.columns:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("평균 수확량", f"{df_result[HARVEST_TARGET].mean():.2f}")
                    col2.metric("최소 수확량", f"{df_result[HARVEST_TARGET].min():.2f}")
                    col3.metric("최대 수확량", f"{df_result[HARVEST_TARGET].max():.2f}")
                
                # 다운로드
                output = io.BytesIO()
                df_result.to_excel(output, index=False, engine='openpyxl')
                output.seek(0)
                
                st.download_button(
                    label="📥 결과 다운로드 (Excel)",
                    data=output,
                    file_name=f"수확량_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

# 사이드바
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    
    st.markdown("""
    ### 원클릭 예측
    환경 데이터만으로 생육지표와 수확량을 한 번에 예측합니다.
    
    ### 1단계: 생육지표 예측
    환경 데이터로 12가지 생육지표를 예측합니다.
    
    ### 2단계: 수확량 예측
    1단계 결과 또는 실측 생육지표로 수확량을 예측합니다.
    
    ---
    
    ### 📁 필요한 데이터
    - **환경 데이터**: 온실 환경 측정값 (온도, 습도 등)
    - **생육지표**: 실측값 (선택사항, 있으면 정확도 향상)
    
    ### 📊 예측 항목
    - **생육지표**: 개화마디, 착과마디, 수확마디 등 12개
    - **수확량**: 주당누적수확수
    """)
    
    st.markdown("---")
    st.markdown("🌱 **온실 생육·수확 예측 시스템**")
    st.markdown("Powered by GRU Deep Learning")
