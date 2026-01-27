# ASSISTFLOW AI - CLEAN OPERATIONAL VERSION
# Focused on workflow clarity, not experimentation

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(".")

from src.pipeline import AssistFlowPipeline
from src.ingestion import load_and_prepare_data, parse_resolution_time
from src.ticket_state import TicketState, get_initial_state, get_state_icon
from config import DATA_PATH, COL_TICKET_ID, COL_SUBJECT, COL_DESCRIPTION, COL_PRIORITY, COL_TICKET_TYPE

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AssistFlow AI - Support Ops",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STYLES
# =============================================================================
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.priority-critical { color:#ff4b4b; font-weight:700; }
.priority-high { color:#ffa500; font-weight:700; }
.priority-medium { color:#ffcc00; font-weight:700; }
.priority-low { color:#00cc66; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_state():
    for k, v in {
        "tickets_processed": False,
        "tickets_df": None,
        "ticket_states": {},
        "analysis_cache": {}
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =============================================================================
# LOADERS
# =============================================================================
@st.cache_resource
def load_pipeline():
    pipe = AssistFlowPipeline()
    return pipe, pipe.load_models()

@st.cache_data(ttl=300)
def load_data():
    return load_and_prepare_data(DATA_PATH)

# =============================================================================
# PROCESSING
# =============================================================================
def process_tickets(pipe, df, max_rows):
    if st.session_state.tickets_processed:
        return st.session_state.tickets_df

    results = []
    bar = st.progress(0)

    for i, row in df.head(max_rows).iterrows():
        bar.progress((i + 1) / max_rows)

        res = pipe.analyze_ticket(
            full_text=row["full_text"],
            ticket_id=row[COL_TICKET_ID],
            time_to_resolution_hours=parse_resolution_time(row.get("Time to Resolution"))
        )

        state = get_initial_state(res.handler_type)

        d = res.to_dict()
        d.update({
            "subject": row.get(COL_SUBJECT, ""),
            "state": state.value,
            "state_icon": get_state_icon(state)
        })

        st.session_state.analysis_cache[res.ticket_id] = res
        st.session_state.ticket_states[res.ticket_id] = state
        results.append(d)

    bar.empty()
    df_out = pd.DataFrame(results)
    st.session_state.tickets_df = df_out
    st.session_state.tickets_processed = True
    return df_out

# =============================================================================
# UI HELPERS
# =============================================================================
def display_table(df):
    df = df.copy()
    df["Priority"] = df["final_priority"].apply(lambda x: f"🔴 {x}" if x=="Critical" else f"🟡 {x}")
    df["SLA"] = df["sla_status"].str.upper()
    df["Handler"] = df["handler_type"].map({"AI":"🤖 AI","Human":"👤 Human"})
    df["State"] = df["state_icon"] + " " + df["state"]

    st.dataframe(
        df[["ticket_id","subject","Priority","SLA","Handler","State"]],
        use_container_width=True,
        hide_index=True,
        height=420
    )

# =============================================================================
# TICKET DETAIL (CLEANED)
# =============================================================================
def show_ticket_detail(tid, df):
    res = st.session_state.analysis_cache[tid]

    st.markdown(f"## 🎫 Ticket {tid}")

    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Priority", res.final_priority)
    col2.metric("SLA", f"{res.sla_hours}h")
    col3.metric("SLA Status", res.sla_status.upper())
    col4.metric("Handler", res.handler_type)
    col5.metric("State", st.session_state.ticket_states[tid].value)

    if res.was_escalated:
        st.warning(f"⬆️ **ESCALATED** — {res.escalation_reason}")

    # ACTIONS
    a,b = st.columns(2)
    if a.button("✅ Mark Resolved"):
        st.session_state.ticket_states[tid] = TicketState.RESOLVED
        st.rerun()

    if res.handler_type=="AI" and b.button("👤 Escalate to Human"):
        st.session_state.ticket_states[tid] = TicketState.WAITING_FOR_HUMAN
        st.rerun()

    # TABS (ONLY 2)
    tab1, tab2 = st.tabs(["📝 Summary", "🔍 Analysis"])

    with tab1:
        st.info(res.ticket_summary)

    with tab2:
        st.markdown(res.explanation_text)
        st.markdown("**ML Predictions**")
        st.write(f"- Priority: {res.predicted_priority} ({res.priority_confidence:.1%})")
        if res.issue_type:
            st.write(f"- Issue Type: {res.issue_type} ({res.issue_type_confidence:.1%})")

# =============================================================================
# VIEWS
# =============================================================================
def inbox_view(df):
    unresolved = [
        tid for tid,s in st.session_state.ticket_states.items()
        if s != TicketState.RESOLVED
    ]
    view = df[df.ticket_id.isin(unresolved)]

    st.markdown("## 📥 Ticket Inbox")
    st.metric("Unresolved", len(view))

    display_table(view)

    sel = st.selectbox("Open ticket", ["—"] + view.ticket_id.tolist())
    if sel!="—":
        show_ticket_detail(sel, df)

def ai_view(df):
    view = df[df.handler_type=="AI"]
    st.markdown("## 🤖 AI Handling Queue")
    display_table(view)

def human_view(df):
    view = df[df.handler_type=="Human"]
    st.markdown("## 👤 Human Queue")
    display_table(view)

def dashboard_view(df):
    st.markdown("## 📊 Operations Dashboard")
    st.bar_chart(df.final_priority.value_counts())
    st.bar_chart(df.sla_status.value_counts())

# =============================================================================
# MAIN
# =============================================================================
def main():
    init_state()
    pipe, ok = load_pipeline()
    if not ok:
        st.error("Models not loaded")
        return

    st.sidebar.title("🎫 AssistFlow AI")
    nav = st.sidebar.radio(
        "Navigation",
        ["📥 Ticket Inbox","🤖 AI Handling","👤 Human Queue","📊 Dashboard"]
    )

    max_rows = st.sidebar.slider("Max tickets",20,300,100)

    if st.sidebar.button("🔄 Process Tickets"):
        st.session_state.tickets_processed=False
        st.rerun()

    raw = load_data()
    df = process_tickets(pipe, raw, max_rows)

    if nav=="📥 Ticket Inbox":
        inbox_view(df)
    elif nav=="🤖 AI Handling":
        ai_view(df)
    elif nav=="👤 Human Queue":
        human_view(df)
    else:
        dashboard_view(df)

    st.sidebar.caption("AssistFlow AI v1.0")

if __name__ == "__main__":
    main()
