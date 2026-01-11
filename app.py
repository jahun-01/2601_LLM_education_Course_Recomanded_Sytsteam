import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import glob
import re
import numpy as np
import os
from pathlib import Path

# ==========================================
# 1. 환경 설정 및 API 키
# ==========================================
st.set_page_config(page_title="K-하이테크 교육 추천(최종)", layout="wide", page_icon="🏭")

# ⚠️ 경로 수정 (사용자 지정 경로)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ⚠️ API 키 입력 필수
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

if not GOOGLE_API_KEY:
    st.error("⚠️ 코드 내 'GOOGLE_API_KEY' 변수에 API 키를 입력해주세요.")
    st.stop()
else:
    genai.configure(api_key=GOOGLE_API_KEY)


# ==========================================
# 2. 데이터 로드 및 전처리 (Curriculum & RAG)
# ==========================================
def normalize_course_name(name):
    if not isinstance(name, str): return ""
    name = re.sub(r'\(.*?\)', '', name)  # 괄호 제거
    name = re.sub(r'\[.*?\]', '', name)  # 대괄호 제거
    for suffix in ["양성과정", "훈련과정", "교육과정", "과정"]:
        name = name.replace(suffix, "")
    name = re.sub(r'\d+(기|차|회|단계|Step)', '', name, flags=re.IGNORECASE)
    return name.strip()


@st.cache_data
def load_all_data(base_path):
    if not os.path.exists(base_path): return [], [], {}, ""

    # ---------------------------------------------------------
    # 1. [기준] 강좌명.csv (Master List)
    # ---------------------------------------------------------
    master_file = os.path.join(DATA_DIR, "02_강의_meta_Data.csv")
    valid_courses = []
    if os.path.exists(master_file):
        try:
        # 인코딩 안전 처리
            try:
                df_master = pd.read_csv(master_file, encoding="cp949")
            except UnicodeDecodeError:
                df_master = pd.read_csv(master_file, encoding="utf-8-sig")

            # 컬럼명 확인
            st.write("df_master columns:", df_master.columns.tolist())

            # 컬럼명 공백 제거(안전장치)
            df_master.columns = df_master.columns.astype(str).str.strip()

            # 필요한 컬럼 존재 검사
            col = "훈련과정명(정답라벨)"
            if col not in df_master.columns:
                st.error(f"⚠️ '{col}' 컬럼이 없습니다. 실제 컬럼: {df_master.columns.tolist()}")
            else:
                valid_courses = df_master[col].dropna().astype(str).unique().tolist()
                st.success(f"✅ valid_courses 로드 성공: {len(valid_courses)}개")

        except Exception as e:
            st.exception(e)
    else:
        st.error("⚠️ master_file 경로에 파일이 없습니다.")

    # ---------------------------------------------------------
    # 2. [구조] 커리큘럼.csv (Curriculum Tracks)
    # ---------------------------------------------------------
    curr_file = os.path.join(DATA_DIR, "03_전체_커리큘럼.csv")
    curriculum_text = ""
    if os.path.exists(curr_file):
        try:
            try:
                df_curr = pd.read_csv(curr_file, encoding='cp949')
            except:
                df_curr = pd.read_csv(curr_file, encoding='utf-8')

            # 프롬프트에 넣기 좋게 텍스트로 변환
            # 형식: [트랙명] 과정1 -> 과정2 -> 과정3 ...
            grouped = df_curr.groupby(['트랙ID', '트랙명', '트랙설명'])
            for (tid, tname, tdesc), group in grouped:
                courses = group.sort_values('과정순서')['과정명'].tolist()
                curriculum_text += f"\n[트랙 {tid}: {tname}]\n- 설명: {tdesc}\n- 연계 과정 순서: {' -> '.join(courses)}\n"
        except Exception as e:
            st.error(f"커리큘럼 로드 실패: {e}")

    # ---------------------------------------------------------
    # 3. [학습] 수정.csv (RAG Knowledge Base)  
    # ---------------------------------------------------------
    # rag_file = os.path.join(DATA_DIR, "수정실험.csv")
    rag_file = os.path.join(DATA_DIR, "01_병합+정규화_Data.csv")
    rag_data = []
    if os.path.exists(rag_file):
        try:
            try:
                df_rag = pd.read_csv(rag_file, encoding='cp949')
            except:
                df_rag = pd.read_csv(rag_file, encoding='utf-8')

            for _, row in df_rag.iterrows():
                rag_data.append({
                    'course': str(row.get('훈련과정명(정답라벨)', '')),
                    'pain': str(row.get('DT 에로사항', '')),
                    'issue': str(row.get('기업현황 및 DT이슈', '')),
                    'as_is': str(row.get('AS-IS', '')),
                    'to_be': str(row.get('To_Be', '')),
                    'goal': str(row.get('목표', '')),
                    'job': str(row.get('훈련 직무', ''))
                })
        except:
            pass

    return valid_courses, rag_data, curriculum_text


# 데이터 로드 실행
valid_courses, rag_data, curriculum_text = load_all_data(BASE_DIR)

# ==========================================
# 3. 메인 UI
# ==========================================
st.title("🏭 K-하이테크 기업 진단 및 강의 추천 (커리큘럼 기반)")

if not valid_courses:
    st.stop()

with st.container():
    st.info("💡 '수정.csv'의 노란색 컬럼에 해당하는 기업 정보를 입력해주세요.")

    with st.form("diagnosis_form"):
        col1, col2 = st.columns(2)
        with col1:
            industry = st.text_input("주업종", placeholder="예: 제조업 (자동차 부품)")
            corp_type = st.selectbox("기업 유형", ["법인", "개인사업자", "기타"])
        with col2:
            comp_name = st.text_input("지원 기업명", placeholder="(주)OOO")
            date = st.date_input("수행 일자")

        st.markdown("---")
        st.subheader("📝 기업 진단 입력")

        st.markdown("**1. 기업현황 및 DT 이슈**")
        current_issue = st.text_area("내용 입력", height=100,
                                     placeholder="예: 주요 생산품은 차체용접설비이며, MES 도입 필요성은 인지하나 이해 부족으로 구축하지 못함.")

        st.markdown("**2. DT 애로사항 (Pain Points)**")
        pain_point = st.text_area("내용 입력", height=100, placeholder="예: 설비 제작 시 물류 관리 전산화가 안 되어 업무 Loss 발생.")

        st.markdown("**3. 기타 의견 (교육 니즈)**")
        etc_opinion = st.text_area("내용 입력", height=80, placeholder="예: 스마트공장 구축 실무 경험 부족으로 기초 교육 희망.")

        submit = st.form_submit_button("🚀 AI 맞춤 강의 추천 시작")

# ==========================================
# 4. AI 분석 (Curriculum & RAG)
# ==========================================
if submit:
    # 1. RAG Context 구성 (유사 사례 검색)
    related_cases = []
    search_keywords = (pain_point + " " + current_issue).split()

    for case in rag_data:
        score = 0
        case_text = str(case['pain']) + str(case['issue'])
        for kw in search_keywords:
            if len(kw) > 1 and kw in case_text:
                score += 1
        if score > 0:
            related_cases.append((score, case))

    related_cases.sort(key=lambda x: x[0], reverse=True)
    top_cases = [c[1] for c in related_cases[:3]]

    rag_context_text = ""
    for i, case in enumerate(top_cases):
        rag_context_text += f"""
        [유사 사례 {i + 1}]
        - 상황: {case['issue'][:50]}...
        - 애로사항: {case['pain'][:50]}...
        - -> 해결 강의: {case['course']}
        - -> 결과(To-Be): {case['to_be']}
        """

    available_courses_str = ", ".join(valid_courses)

    # 2. 프롬프트 작성
    prompt = f"""
    당신은 한국공학대학교 K-하이테크 플랫폼의 교육 컨설턴트입니다.
    기업 정보를 분석하여 가장 적합한 **교육 커리큘럼(3단계 코스)**을 설계하세요.

    [입력된 기업 정보]
    - 주업종: {industry}
    - 기업현황 및 이슈: {current_issue}
    - DT 애로사항: {pain_point}
    - 기타 니즈: {etc_opinion}

    [제공된 커리큘럼 구조 (Curriculum Tracks)]
    **반드시 아래 트랙 중에서 하나를 선택하여, 그 안에 포함된 강의들로 3단계 코스를 구성하세요.**
    {curriculum_text}

    [참고 가능한 과거 사례 (RAG)]
    {rag_context_text}

    [결과물 작성 양식 (Strict Format)]
    제목 크기는 작게(####) 작성하세요.

    #### 1. 기업 진단 내용
    (기업의 현황과 애로사항을 분석하여 교육 필요성을 3~4줄로 요약)

    #### 2. 추천 훈련과정 (3단계 커리큘럼)
    선정된 트랙: [트랙명 작성]

    - **Step 1 (기초/입문):** [강의명]
      - *선정 이유:* (해당 트랙의 기초 과정으로서의 역할 설명)

    - **Step 2 (핵심/해결):** [강의명]
      - *선정 이유:* (기업의 애로사항 "{pain_point}"을 직접 해결하는 핵심 강의임)

    - **Step 3 (심화/확장):** [강의명]
      - *선정 이유:* (심화 학습 또는 연계 역량 강화 목적)

    #### 3. 훈련 목표
    (전체 커리큘럼을 통해 달성하고자 하는 목표)

    #### 4. 훈련 주요 내용
    (Step 2 핵심 강의를 중심으로 주요 내용 3가지 요약)

    #### 5. 기대 효과 (AS-IS -> To-Be)
    - **AS-IS (현재):** {pain_point} 요약
    - **To-Be (변화):** 교육 이수 후 개선될 모습
    """

    with st.spinner("🧠 최적의 커리큘럼 트랙을 선정하고 로드맵을 설계 중입니다..."):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            res = model.generate_content(prompt)

            st.markdown("### 📋 K-하이테크 맞춤형 교육 커리큘럼 제안서")
            st.divider()
            st.markdown(res.text)

            with st.expander("🔍 AI가 참고한 유사 기업 사례 (RAG)"):
                st.write(rag_context_text)

            with st.expander("📚 전체 커리큘럼 목록 보기"):
                st.text(curriculum_text)

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")