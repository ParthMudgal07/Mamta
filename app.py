from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "individual_risk_rf_model.pkl"
DISTRICT_RISK_PATH = APP_DIR / "district_risk_table.csv"
MEDICAL_WEIGHT = 0.7
DISTRICT_WEIGHT = 0.3
RISK_MAP = {
    0: 90,
    1: 25,
    2: 60,
}

load_dotenv(APP_DIR / ".env", override=True)


st.set_page_config(
    page_title="Mamta",
    page_icon="H",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_district_risk_table() -> pd.DataFrame:
    df = pd.read_csv(DISTRICT_RISK_PATH)
    df.columns = df.columns.str.strip()
    df["District Names"] = df["District Names"].astype(str).str.strip()
    df["State/UT"] = df["State/UT"].astype(str).str.strip()
    df["DistrictRisk"] = pd.to_numeric(df["DistrictRisk"], errors="coerce")
    return df.dropna(subset=["DistrictRisk"]).reset_index(drop=True)


def get_class_scores(model) -> np.ndarray:
    missing_labels = [label for label in model.classes_ if label not in RISK_MAP]
    if missing_labels:
        raise ValueError(f"Missing risk mapping for model classes: {missing_labels}")
    return np.array([RISK_MAP[label] for label in model.classes_], dtype=float)


def get_medical_risk_score(model, patient_df: pd.DataFrame) -> pd.DataFrame:
    class_scores = get_class_scores(model)
    probs = model.predict_proba(patient_df)
    medical_scores = probs @ class_scores
    predicted_labels = model.predict(patient_df)

    result = patient_df.copy()
    result["PredictedRiskClass"] = predicted_labels
    result["MedicalRiskScore"] = medical_scores.round(2)
    return result


def get_district_risk(district_name: str, state_ut: str, district_risk_df: pd.DataFrame) -> float:
    match = district_risk_df[
        (district_risk_df["District Names"].str.lower() == district_name.strip().lower())
        & (district_risk_df["State/UT"].str.lower() == state_ut.strip().lower())
    ]

    if match.empty:
        raise ValueError(f"District risk not found for {district_name}, {state_ut}")

    return float(match.iloc[0]["DistrictRisk"])


def get_risk_category(final_risk: float) -> str:
    if final_risk <= 33:
        return "Low Risk"
    if final_risk <= 66:
        return "Moderate Risk"
    return "High Risk"


def compute_final_risk(
    model,
    patient_df: pd.DataFrame,
    district_name: str,
    state_ut: str,
    district_risk_df: pd.DataFrame,
    medical_weight: float = MEDICAL_WEIGHT,
    district_weight: float = DISTRICT_WEIGHT,
) -> pd.DataFrame:
    if medical_weight < 0 or district_weight < 0 or not np.isclose(medical_weight + district_weight, 1.0):
        raise ValueError("Medical and district weights must be non-negative and sum to 1.")

    result = get_medical_risk_score(model, patient_df)
    district_risk = get_district_risk(district_name, state_ut, district_risk_df)

    result["District Names"] = district_name
    result["State/UT"] = state_ut
    result["DistrictRisk"] = round(district_risk, 2)
    result["FinalRiskPercentage"] = (
        medical_weight * result["MedicalRiskScore"] + district_weight * result["DistrictRisk"]
    ).round(2)
    result["FinalRiskPercentage"] = result["FinalRiskPercentage"].clip(0, 100)
    result["FinalRiskCategory"] = result["FinalRiskPercentage"].apply(get_risk_category)
    return result


def build_patient_df(
    age: int,
    systolic_bp: int,
    diastolic_bp: int,
    blood_sugar: float,
    body_temp: float,
    heart_rate: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Age": age,
                "SystolicBP": systolic_bp,
                "DiastolicBP": diastolic_bp,
                "BS": blood_sugar,
                "BodyTemp": body_temp,
                "HeartRate": heart_rate,
            }
        ]
    )


def get_openrouter_client() -> OpenAI | None:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        try:
            if "OPENROUTER_API_KEY" in st.secrets:
                api_key = st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        return None

    api_key = str(api_key).strip().strip('"').strip("'")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def generate_fallback_guidance(final_risk_percentage: float, final_risk_category: str) -> str:
    urgency = {
        "Low Risk": "Continue routine antenatal follow-up and keep monitoring your health.",
        "Moderate Risk": "A doctor review should be prioritized and symptoms should be monitored closely.",
        "High Risk": "Please seek prompt clinical evaluation and do not wait if any warning sign appears.",
    }[final_risk_category]

    return (
        f"Risk summary: Your current maternal risk estimate is {final_risk_percentage:.2f}% "
        f"({final_risk_category}). {urgency}\n\n"
        "Diet: Eat balanced meals with iron-rich foods, protein, fruits, vegetables, and enough fluids.\n\n"
        "Daily precautions: Rest well, take prescribed supplements on time, attend antenatal visits, "
        "and keep track of blood pressure, blood sugar, and unusual symptoms.\n\n"
        "Urgent warning signs: Severe headache, blurred vision, heavy bleeding, severe abdominal pain, "
        "reduced fetal movement, swelling with breathlessness, or convulsions require immediate medical attention.\n\n"
        "Follow-up: Use this result as a support tool only and discuss it with a qualified doctor or nurse."
    )


def generate_medical_guidance(
    final_risk_percentage: float,
    final_risk_category: str,
    patient_context: dict[str, str | float | int],
) -> tuple[str, str]:
    client = get_openrouter_client()
    if client is None:
        return (
            generate_fallback_guidance(final_risk_percentage, final_risk_category),
            "LLM summary is using the built-in fallback because `OPENROUTER_API_KEY` is not set.",
        )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Dr. Elena Reyes, a compassionate obstetrician with 20+ years of experience "
                        "specializing in high-risk pregnancies. Your role is to analyze a pregnant woman's health "
                        "and environmental data, provide a clear risk assessment, and deliver personalized guidance "
                        "in a warm, reassuring tone, like a trusted doctor during a consultation.\n\n"
                        "Base your guidance on standard ACOG and WHO-aligned pregnancy care principles. "
                        "Do not diagnose diseases, prescribe medicines, or claim to replace in-person medical care. "
                        "Use simple, empathetic language, avoid jargon or briefly explain it when needed, and be supportive without sounding alarming. "
                        "Return one single cohesive summary between 400 and 550 words."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Patient Data:\n\n"
                        f"Gestational age: {patient_context.get('gestational_age', 'Not provided')}\n\n"
                        "Medical vitals:\n"
                        f"- Blood sugar level: {patient_context.get('blood_sugar', 'Not provided')}\n"
                        f"- Blood pressure: {patient_context.get('blood_pressure', 'Not provided')}\n"
                        f"- Heart rate: {patient_context.get('heart_rate', 'Not provided')}\n"
                        f"- Other vitals: {patient_context.get('other_vitals', 'Not provided')}\n\n"
                        f"Medical history/conditions: {patient_context.get('medical_history', 'Not provided')}\n\n"
                        f"Current symptoms: {patient_context.get('current_symptoms', 'Not provided')}\n\n"
                        f"Location/Environment: {patient_context.get('location_environment', 'Not provided')}\n\n"
                        f"Environmental suitability analysis: {patient_context.get('environmental_analysis', 'Not provided')}\n\n"
                        "Pre-computed Risk Output from System (use exactly as provided, do not change):\n\n"
                        f"Final Risk Category: {final_risk_category}\n"
                        f"Risk Percentage: {final_risk_percentage:.2f}%\n\n"
                        "Generate a single, cohesive summary strictly between 400 and 550 words, structured as follows:\n\n"
                        "1. Opening Reassurance (1-2 sentences): Greet warmly, acknowledge her stage of pregnancy, "
                        "and state the overall risk category and percentage positively.\n\n"
                        "2. Key Findings (100-150 words): Summarize vitals, conditions, and environmental factors. "
                        "Highlight positives first, then concerns. Explain implications simply.\n\n"
                        "3. Dietary Recommendations (80-100 words): Give personalized, practical advice based on the data. "
                        "Include 4-5 sample daily meals or snacks, approximate portion sizes, and hydration tips.\n\n"
                        "4. Precautions & Lifestyle Guidance (100-150 words): Tailor precautions to the patient data. "
                        "Cover monitoring, activity, sleep, stress management, and environment-specific tips. "
                        "Clearly mention symptoms that need urgent medical attention.\n\n"
                        "5. Closing Motivation (1-2 sentences): End encouragingly with next steps and an offer for follow-up.\n\n"
                        "Important instructions:\n"
                        "- If some patient details are unavailable, use only the provided information and do not invent facts.\n"
                        "- Keep the tone warm, reassuring, and specific.\n"
                        "- Do not use bullet points in the final answer.\n"
                        "- End with the exact text format: [FINAL WORD COUNT: <number>]."
                    ),
                },
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip(), "LLM summary generated using OpenRouter."
    except Exception as exc:
        return (
            generate_fallback_guidance(final_risk_percentage, final_risk_category),
            f"LLM summary fell back to the built-in guidance because the API request failed: {exc}",
        )


def render_sidebar(district_risk_df: pd.DataFrame) -> tuple[str, str]:
    st.sidebar.header("Location")
    states = sorted(district_risk_df["State/UT"].unique().tolist())
    state_ut = st.sidebar.selectbox("State / UT", states)

    districts = (
        district_risk_df.loc[district_risk_df["State/UT"] == state_ut, "District Names"]
        .sort_values()
        .tolist()
    )
    district_name = st.sidebar.selectbox("District", districts)
    return state_ut, district_name


def render_results(result_row: pd.Series, llm_guidance: str, llm_status: str) -> None:
    st.subheader("Assessment Result")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Final Risk Percentage", f"{result_row['FinalRiskPercentage']:.2f}%")
    metric_cols[1].metric("Final Risk Category", str(result_row["FinalRiskCategory"]))
    metric_cols[2].metric("District Risk", f"{result_row['DistrictRisk']:.2f}")

    st.caption(
        f"Predicted medical risk score: {result_row['MedicalRiskScore']:.2f} | "
        f"Predicted model class: {result_row['PredictedRiskClass']}"
    )

    st.subheader("Guidance Summary")
    st.info(llm_status)
    st.markdown(llm_guidance)

    with st.expander("See submitted values"):
        st.dataframe(result_row.to_frame().T, use_container_width=True)


def main() -> None:
    st.title("Mamta")
    st.write(
        "Enter the user details and location"
    )

    try:
        model = load_model()
        district_risk_df = load_district_risk_table()
    except Exception as exc:
        st.error(f"Failed to load application files: {exc}")
        st.stop()

    state_ut, district_name = render_sidebar(district_risk_df)

    st.subheader("Patient Inputs")
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Age", min_value=10, max_value=60, value=28, step=1)
    systolic_bp = col2.number_input("Systolic BP", min_value=70, max_value=220, value=130, step=1)
    diastolic_bp = col3.number_input("Diastolic BP", min_value=40, max_value=140, value=85, step=1)

    col4, col5, col6 = st.columns(3)
    blood_sugar = col4.number_input("Blood Sugar (BS)", min_value=1.0, max_value=30.0, value=7.5, step=0.1)
    body_temp = col5.number_input("Body Temperature", min_value=90.0, max_value=110.0, value=98.4, step=0.1)
    heart_rate = col6.number_input("Heart Rate", min_value=30, max_value=220, value=82, step=1)

    if st.button("Calculate Risk", type="primary"):
        try:
            patient_df = build_patient_df(
                age=age,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                blood_sugar=blood_sugar,
                body_temp=body_temp,
                heart_rate=heart_rate,
            )
            result = compute_final_risk(
                model=model,
                patient_df=patient_df,
                district_name=district_name,
                state_ut=state_ut,
                district_risk_df=district_risk_df,
            )
            result_row = result.iloc[0]
            llm_guidance, llm_status = generate_medical_guidance(
                final_risk_percentage=float(result_row["FinalRiskPercentage"]),
                final_risk_category=str(result_row["FinalRiskCategory"]),
                patient_context={
                    "gestational_age": "Not provided in this demo",
                    "blood_sugar": f"{blood_sugar}",
                    "blood_pressure": f"{systolic_bp}/{diastolic_bp} mmHg",
                    "heart_rate": f"{heart_rate} bpm",
                    "other_vitals": f"Age: {age}, Body temperature: {body_temp}",
                    "medical_history": "Not provided in this demo",
                    "current_symptoms": "Not provided in this demo",
                    "location_environment": f"{district_name}, {state_ut}",
                    "environmental_analysis": f"District risk score: {float(result_row['DistrictRisk']):.2f}",
                },
            )
            render_results(result_row, llm_guidance, llm_status)
        except Exception as exc:
            st.error(f"Could not calculate the assessment: {exc}")

    with st.expander("Important note"):
        st.write(
            "This app is a decision-support demo and not a substitute for diagnosis, emergency care, "
            "or clinical judgment from a qualified healthcare professional."
        )


if __name__ == "__main__":
    main()
