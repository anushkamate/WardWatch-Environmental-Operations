import io
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["NH3", "NO2", "CO", "PM2.5", "Temp", "Pressure", "Humidity", "O3"]
POLLUTANTS = ["NH3", "NO2", "CO", "PM2.5", "O3"]
CLASSES = ["Normal", "Review", "Priority"]
ROOMS = ["All rooms", "ICU", "Operating Theatre", "General Ward", "Patient Room", "Laboratory"]

st.set_page_config(
    page_title="WardWatch | Environmental Operations",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#182b3a; --muted:#697b89; --line:#e1e8ed; --paper:#f5f8fa; --teal:#0c766e; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: #f5f8fa; color:var(--ink); }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-0.035em; color:#182b3a; }
h1 { font-size:2.05rem !important; font-weight:800 !important; }
h2 { font-size:1.35rem !important; font-weight:750 !important; }
h3 { font-size:1.05rem !important; font-weight:750 !important; }
[data-testid="stSidebar"] { background:#142b3a; }
[data-testid="stSidebar"] * { color:#eef5f7 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:#203d4d; border-color:#426170; }
[data-testid="stSidebar"] .stButton button { background:#b7eee3; color:#153b3b !important; border:0; font-weight:700; }
[data-testid="stSidebar"] hr { border-color:#385261; }
div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:15px; padding:17px 18px; box-shadow:0 2px 8px rgba(25,50,68,.025); }
div[data-testid="stMetricLabel"] { color:#6a7d89; font-size:.82rem; font-weight:600; }
div[data-testid="stMetricValue"] { color:#183647; font-family:'Manrope',sans-serif; font-size:1.85rem; font-weight:800; }
div[data-testid="stMetricDelta"] { font-size:.75rem; }
.panel { background:white; border:1px solid #e1e8ed; border-radius:16px; padding:20px 22px; box-shadow:0 3px 12px rgba(20,45,60,.025); }
.eyebrow { color:#6c7f8d; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:800; margin-bottom:7px; }
.hero { background:linear-gradient(115deg,#173545 0%,#1d5960 100%); border-radius:20px; padding:27px 30px; color:#f5fbfb; margin-bottom:20px; position:relative; overflow:hidden; }
.hero h1 { color:#fff !important; margin:0 0 8px 0; font-size:2rem !important; }
.hero p { color:#d0e5e7; margin:0; max-width:720px; line-height:1.6; }
.hero .hero-tag { display:inline-block; border:1px solid #65969a; background:#2d676c; color:#e7fbf7; border-radius:999px; padding:5px 10px; font-size:.72rem; margin-bottom:13px; }
.section-note { color:#738492; font-size:.84rem; margin-top:-8px; margin-bottom:14px; }
.status-pill { display:inline-block; padding:5px 10px; border-radius:999px; font-size:.75rem; font-weight:700; }
.pill-normal { background:#e2f5ed; color:#146c50; }
.pill-review { background:#fff2d9; color:#8b5b00; }
.pill-priority { background:#fde7e5; color:#a6312c; }
.small-muted { color:#738492; font-size:.78rem; }
hr { border-color:#e1e8ed; }
.stButton button, .stDownloadButton button { border-radius:10px; min-height:2.7rem; font-weight:700; }
[data-testid="stDataFrame"] { border:1px solid #e1e8ed; border-radius:12px; overflow:hidden; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

def normalize_columns(frame):
    df = frame.copy()
    normalized = {
        str(c).strip().lower().replace(" ", "").replace("_", ""): c
        for c in df.columns
    }
    aliases = {
        "NH3": ["nh3", "ammonia"],
        "NO2": ["no2", "nitrogendioxide"],
        "CO": ["co", "carbonmonoxide"],
        "PM2.5": ["pm2.5", "pm25", "pm2_5", "pm2,5"],
        "Temp": ["temp", "temperature", "temperaturec"],
        "Pressure": ["pressure", "atmosphericpressure"],
        "Humidity": ["humidity", "relativehumidity"],
        "O3": ["o3", "ozone"],
    }
    rename = {}
    for target, names in aliases.items():
        for name in names:
            key = name.lower().replace(" ", "").replace("_", "")
            if key in normalized:
                rename[normalized[key]] = target
                break
    return df.rename(columns=rename)

def prepare_data(raw):
    df = normalize_columns(raw)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            ". Please check that your CSV contains the eight required measurements."
        )
    df = df[FEATURES].copy()
    for c in FEATURES:
        # Handle decimal commas in imported CSVs.
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} valid rows found. At least 30 are recommended.")
    return df

def label_data(df):
    data = df.copy()
    limits_warning = data[POLLUTANTS].quantile(.75)
    limits_priority = data[POLLUTANTS].quantile(.90)
    def label(row):
        warning_count = sum(row[c] > limits_warning[c] for c in POLLUTANTS)
        priority_count = sum(row[c] > limits_priority[c] for c in POLLUTANTS)
        temp, humidity = row["Temp"], row["Humidity"]
        if priority_count >= 2 or temp < 18 or temp > 30 or humidity < 30 or humidity > 70:
            return "Priority"
        if warning_count >= 2 or temp < 20 or temp > 27 or humidity < 35 or humidity > 65:
            return "Review"
        return "Normal"
    data["Condition"] = data.apply(label, axis=1)
    return data, limits_warning, limits_priority

@st.cache_data(show_spinner=False, max_entries=3)
def train_model(csv_bytes):
    raw = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    data = prepare_data(raw)
    labeled, warn_limits, priority_limits = label_data(data)
    counts = labeled["Condition"].value_counts()
    present = [c for c in CLASSES if counts.get(c, 0) > 0]
    if len(present) < 2:
        raise ValueError(
            "The rule-based labels produced fewer than two classes. "
            "Check dataset units or revise the demonstration thresholds."
        )
    if min(counts[c] for c in present) < 2:
        raise ValueError("A generated class has fewer than two examples. Dataset may not support stratified evaluation.")
    X_train, X_test, y_train, y_test = train_test_split(
        labeled[FEATURES], labeled["Condition"], test_size=.2,
        random_state=42, stratify=labeled["Condition"]
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=140, max_depth=16, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, labels=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, predictions, labels=CLASSES)
    return labeled, model, accuracy, report, cm, warn_limits, priority_limits, present

def card_label(status):
    cls = {"Normal": "pill-normal", "Review": "pill-review", "Priority": "pill-priority"}.get(status, "pill-review")
    return f'<span class="status-pill {cls}">{status}</span>'

def fmt_num(value):
    return f"{value:,.2f}"

with st.sidebar:
    st.markdown("## ◉ WardWatch")
    st.markdown("<div style='color:#aac0ca;font-size:.78rem;line-height:1.5;margin-top:-8px'>ENVIRONMENTAL OPERATIONS</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Data source")
    uploaded = st.file_uploader("Upload air-quality CSV", type=["csv"], help="CSV needs NH3, NO2, CO, PM2.5, Temp, Pressure, Humidity and O3.")
    st.markdown("<div class='small-muted'>Data stays in this app session. Generated labels are for demonstration only.</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Workspace")
    st.markdown("<div style='font-size:.82rem;line-height:1.8;color:#d2e0e5'>Model: Random Forest<br>Inference: local session<br>Deployment: Streamlit</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='color:#a9c1ca;font-size:.72rem'>WARDWATCH · DEMO BUILD 1.0</div>", unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">FACILITIES INTELLIGENCE · DEMO</div>
      <h1>A clearer view of indoor conditions.</h1>
      <p>Review air-quality measurements, explore room-level readings and flag unusual conditions for facilities follow-up — in one calm, practical workspace.</p>
    </div>
    """, unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="panel"><div class="eyebrow">01 / Import</div><h3>Bring your dataset</h3><p class="small-muted">Upload the indoor air-pollution CSV to validate and prepare the measurements.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="eyebrow">02 / Review</div><h3>Explore patterns</h3><p class="small-muted">Review model performance, measurement distributions and generated condition labels.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel"><div class="eyebrow">03 / Simulate</div><h3>Test a room</h3><p class="small-muted">Enter a set of readings and run local inference for a simulated room.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.info("To begin, upload your CSV using **Data source** in the left sidebar.")
    st.markdown("#### Expected measurements")
    st.code("NH3 · NO2 · CO · PM2.5 · Temp · Pressure · Humidity · O3")
    st.caption("This is an educational prototype. It does not receive live sensor data or establish that any hospital room is safe.")
    st.stop()

try:
    with st.spinner("Preparing measurements and fitting the demonstration model…"):
        data, model, accuracy, report, cm, warning_limits, priority_limits, present_classes = train_model(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not prepare this dataset: {exc}")
    st.markdown("Check the column names and numeric units, then upload a corrected CSV.")
    st.stop()

st.markdown(f"""
<div class="hero">
  <div class="hero-tag">FACILITIES INTELLIGENCE · DATA LOADED</div>
  <h1>Environmental overview</h1>
  <p>Review the current dataset, identify unusual patterns and inspect simulated room readings. Updated {datetime.now().strftime("%d %b %Y, %H:%M")}.</p>
</div>
""", unsafe_allow_html=True)

total = len(data)
condition_counts = data["Condition"].value_counts()
normal_n = int(condition_counts.get("Normal", 0))
review_n = int(condition_counts.get("Review", 0))
priority_n = int(condition_counts.get("Priority", 0))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Valid measurements", f"{total:,}", help="Rows remaining after numeric validation.")
m2.metric("Normal · rule label", f"{normal_n:,}")
m3.metric("Review · rule label", f"{review_n:,}")
m4.metric("Priority · rule label", f"{priority_n:,}")

tab_overview, tab_room, tab_data, tab_model = st.tabs([
    "Overview", "Room assessment", "Dataset explorer", "Model evaluation"
])

with tab_overview:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Condition distribution")
        st.markdown('<div class="section-note">Rule-generated categories across the uploaded dataset</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        values = [condition_counts.get(c, 0) for c in CLASSES]
        colors = ["#2b8a72", "#d59b37", "#c95b54"]
        ax.bar(CLASSES, values, color=colors, width=.56)
        ax.set_ylabel("Measurements")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("### Measurement snapshot")
        st.markdown('<div class="section-note">Median values from the uploaded dataset</div>', unsafe_allow_html=True)
        summary = data[FEATURES].median().rename("Median").reset_index()
        summary.columns = ["Measurement", "Median"]
        summary["Median"] = summary["Median"].map(lambda x: f"{x:,.3f}")
        st.dataframe(summary, hide_index=True, use_container_width=True, height=330)
    st.markdown("### Quick notes")
    st.markdown(
        f"- **{review_n + priority_n:,}** rows received a Review or Priority demonstration label.\n"
        f"- **{len(present_classes)}** condition classes are present in the generated labels.\n"
        "- Use the labels as a workflow demonstration only; they are not validated safety thresholds."
    )
    st.download_button(
        "Download labelled dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="wardwatch_labelled_demo.csv",
        mime="text/csv"
    )

with tab_room:
    st.markdown("### Assess a set of room readings")
    st.markdown('<div class="section-note">Enter measurements using the same units as the uploaded dataset.</div>', unsafe_allow_html=True)
    room = st.selectbox("Room / area", ROOMS[1:], key="room_select")
    medians = data[FEATURES].median()
    with st.form("assessment_form"):
        inputs = {}
        grid = st.columns(4)
        for idx, feature in enumerate(FEATURES):
            with grid[idx % 4]:
                inputs[feature] = st.number_input(
                    feature, value=float(medians[feature]), format="%.4f", key=f"room_{feature}"
                )
        submitted = st.form_submit_button("Assess readings", type="primary", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([inputs], columns=FEATURES)
        pred = str(model.predict(sample)[0])
        probs = model.predict_proba(sample)[0]
        fitted_classes = list(model.named_steps["classifier"].classes_)
        class_prob = dict(zip(fitted_classes, probs))
        confidence = float(class_prob.get(pred, max(probs))) * 100
        st.session_state["assessment"] = {
            "Room": room, **inputs, "Condition": pred,
            "Model class probability (%)": round(confidence, 2),
            "Assessed at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    if "assessment" in st.session_state:
        result = st.session_state["assessment"]
        st.markdown("#### Assessment result")
        if result["Condition"] == "Priority":
            st.error("Priority review recommended. Verify the readings using appropriate instruments and facility procedures.")
        elif result["Condition"] == "Review":
            st.warning("Review recommended. Facilities staff should check the readings and local guidance.")
        else:
            st.success("No review flag produced by this demonstration model.")
        st.metric("Model class probability", f'{result["Model class probability (%)"]:.1f}%')
        st.caption("This is the model's estimated class probability, not a probability that the room is safe.")
        result_frame = pd.DataFrame([result])
        st.dataframe(result_frame, use_container_width=True)
        st.download_button(
            "Download assessment",
            data=result_frame.to_csv(index=False).encode("utf-8"),
            file_name="wardwatch_room_assessment.csv",
            mime="text/csv"
        )

with tab_data:
    st.markdown("### Explore measurements")
    selected = st.multiselect("Measurements to compare", FEATURES, default=["Temp", "Humidity", "PM2.5"])
    if selected:
        st.dataframe(data[selected].describe().T.round(3), use_container_width=True)
        feature_choice = st.selectbox("Distribution", selected)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(data[feature_choice], bins=35, color="#287d78", alpha=.9)
        ax.set_title(f"{feature_choice} distribution")
        ax.set_xlabel(f"{feature_choice} (source dataset units)")
        ax.set_ylabel("Frequency")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.dataframe(data.head(1000), use_container_width=True, height=360)
    st.caption("Showing up to 1,000 rows for browsing.")
    st.download_button(
        "Download cleaned measurements",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="wardwatch_cleaned_measurements.csv",
        mime="text/csv"
    )

with tab_model:
    st.markdown("### Held-out evaluation")
    st.markdown('<div class="section-note">A random 20% test split, evaluated against generated demonstration labels.</div>', unsafe_allow_html=True)
    st.metric("Test accuracy", f"{accuracy * 100:.2f}%")
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(3), use_container_width=True)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    st.markdown("#### Confusion matrix")
    st.dataframe(cm_df, use_container_width=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm, cmap="GnBu")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Rule-generated label")
    ax.set_title("Confusion matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#183647")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.info("Because labels are generated from rules applied to this dataset, this score measures reproduction of those rules—not real-world environmental safety.")

st.divider()
st.markdown(
    "<div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#778995;font-size:.75rem'>"
    "<span>WARDWATCH · ENVIRONMENTAL OPERATIONS</span>"
    "<span>Educational prototype · Not a certified safety system</span></div>",
    unsafe_allow_html=True
)
