"""AssistFlow AI - Streamlit Dashboard"""

import streamlit as st
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(".")

from src.pipeline import AssistFlowPipeline, TicketAnalysisResult
from src.ingestion import load_and_prepare_data, parse_resolution_time
from src.ticket_state import TicketState, get_initial_state, get_state_icon, get_state_color
from config import DATA_PATH, COL_TICKET_ID, COL_SUBJECT, COL_DESCRIPTION, COL_PRIORITY, COL_TICKET_TYPE

st.set_page_config(
    page_title="AssistFlow AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .priority-critical { background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; }
    .priority-high { background-color: #fd7e14; color: white; padding: 2px 8px; border-radius: 4px; }
    .priority-medium { background-color: #ffc107; color: black; padding: 2px 8px; border-radius: 4px; }
    .priority-low { background-color: #28a745; color: white; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'tickets_processed' not in st.session_state:
        st.session_state.tickets_processed = False
    if 'tickets_df' not in st.session_state:
        st.session_state.tickets_df = None
    if 'ticket_states' not in st.session_state:
        st.session_state.ticket_states = {}
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}


@st.cache_resource
def load_pipeline():
    pipeline = AssistFlowPipeline()
    success = pipeline.load_models()
    return pipeline, success


@st.cache_data(ttl=300)
def load_raw_dataset():
    return load_and_prepare_data(DATA_PATH)


def process_all_tickets(pipeline, df, max_tickets=200):
    if st.session_state.tickets_processed and st.session_state.tickets_df is not None:
        return st.session_state.tickets_df
    
    results = []
    sample_df = df.head(max_tickets)
    progress_bar = st.progress(0)
    status = st.empty()
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        status.text(f"Processing ticket {i+1}/{len(sample_df)}...")
        resolution_time = parse_resolution_time(row.get("Time to Resolution", None))
        
        try:
            result = pipeline.analyze_ticket(
                full_text=row["full_text"],
                ticket_id=row[COL_TICKET_ID],
                time_to_resolution_hours=resolution_time
            )
            result_dict = result.to_dict()
            result_dict['subject'] = row.get(COL_SUBJECT, '')
            result_dict['description'] = row.get(COL_DESCRIPTION, '')[:200]
            result_dict['original_priority'] = row.get(COL_PRIORITY, '')
            result_dict['original_type'] = row.get(COL_TICKET_TYPE, '')
            
            state = get_initial_state(result.handler_type)
            result_dict['state'] = state.value
            result_dict['state_icon'] = get_state_icon(state)
            
            st.session_state.analysis_cache[result.ticket_id] = result
            st.session_state.ticket_states[result.ticket_id] = state
            results.append(result_dict)
        except Exception as e:
            print(f"Error processing ticket {row[COL_TICKET_ID]}: {e}")
        
        progress_bar.progress((i + 1) / len(sample_df))
    
    progress_bar.empty()
    status.empty()
    
    results_df = pd.DataFrame(results)
    st.session_state.tickets_df = results_df
    st.session_state.tickets_processed = True
    return results_df


def get_priority_sort_order(priority):
    return {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}.get(priority, 4)


def get_sla_sort_order(sla_status):
    return {'breached': 0, 'at_risk': 1, 'met': 2}.get(sla_status, 3)


def format_priority(priority):
    colors = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
    return f"{colors.get(priority, '⚪')} {priority}"


def format_sla_status(status):
    icons = {'breached': '🚨 BREACHED', 'at_risk': '⚠️ AT RISK', 'met': '✅ Met'}
    return icons.get(status, status)


def format_handler(handler):
    return '👤 Human' if handler == 'Human' else '🤖 AI'


def display_ticket_table(df, show_columns=None, key_prefix="table"):
    if df.empty:
        st.info("No tickets to display.")
        return None
    
    if show_columns is None:
        show_columns = ['ticket_id', 'subject', 'final_priority', 'sla_status', 'handler_type', 'state']
    
    available_cols = [c for c in show_columns if c in df.columns]
    display_df = df[available_cols].copy()
    
    if 'final_priority' in display_df.columns:
        display_df['final_priority'] = display_df['final_priority'].apply(format_priority)
    if 'sla_status' in display_df.columns:
        display_df['sla_status'] = display_df['sla_status'].apply(format_sla_status)
    if 'handler_type' in display_df.columns:
        display_df['handler_type'] = display_df['handler_type'].apply(format_handler)
    if 'state' in display_df.columns:
        display_df['state'] = df['state_icon'] + ' ' + display_df['state']
    
    column_names = {
        'ticket_id': 'Ticket ID', 'subject': 'Subject', 'final_priority': 'Priority',
        'sla_status': 'SLA Status', 'handler_type': 'Handler', 'state': 'State', 'issue_type': 'Issue Type'
    }
    display_df = display_df.rename(columns=column_names)
    
    if 'Subject' in display_df.columns:
        display_df['Subject'] = display_df['Subject'].str[:50] + '...'
    
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
    return df['ticket_id'].tolist()


def display_ticket_detail(ticket_id, df, analysis_cache):
    if ticket_id not in analysis_cache:
        st.error(f"Ticket {ticket_id} not found.")
        return
    
    result = analysis_cache[ticket_id]
    ticket_row = df[df['ticket_id'] == ticket_id].iloc[0] if not df[df['ticket_id'] == ticket_id].empty else None
    
    st.markdown(f"### 🎫 Ticket: {ticket_id}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✅ Mark Resolved", key=f"resolve_{ticket_id}"):
            st.session_state.ticket_states[ticket_id] = TicketState.RESOLVED
            st.success("Resolved!")
            st.rerun()
    with col2:
        if result.handler_type == "AI":
            if st.button("👤 Escalate", key=f"escalate_{ticket_id}"):
                st.session_state.ticket_states[ticket_id] = TicketState.WAITING_FOR_HUMAN
                st.success("Escalated!")
                st.rerun()
    
    st.divider()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Priority", result.final_priority)
    with col2:
        st.metric("SLA", f"{result.sla_hours}h")
    with col3:
        st.metric("SLA Status", result.sla_status.upper())
    with col4:
        st.metric("Handler", result.handler_type)
    with col5:
        current_state = st.session_state.ticket_states.get(ticket_id, TicketState.NEW)
        st.metric("State", current_state.value)
    
    if result.was_escalated:
        st.warning(f"⬆️ ESCALATED: {result.escalation_reason}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔍 Analysis", "✉️ Response", "📊 Raw"])
    
    with tab1:
        st.markdown("**Subject:**")
        st.write(ticket_row['subject'] if ticket_row is not None else "N/A")
        st.markdown("**Summary:**")
        st.info(result.ticket_summary)
        with st.expander("Full text"):
            st.write(result.full_text)
    
    with tab2:
        st.markdown(result.explanation_text)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Priority: {result.predicted_priority} ({result.priority_confidence:.1%})")
        with col2:
            if result.issue_type:
                st.write(f"Issue: {result.issue_type} ({result.issue_type_confidence:.1%})")
    
    with tab3:
        response = st.text_area("Edit response:", value=result.suggested_response, height=300, key=f"response_{ticket_id}")
        if st.button("📤 Send", type="primary", key=f"send_{ticket_id}"):
            st.success("Sent!")
            st.session_state.ticket_states[ticket_id] = TicketState.RESOLVED
    
    with tab4:
        st.json(result.to_dict())


def view_ticket_inbox(df, analysis_cache):
    st.markdown("## 📥 Ticket Inbox")
    
    resolved_ids = [tid for tid, state in st.session_state.ticket_states.items() if state == TicketState.RESOLVED]
    unresolved_df = df[~df['ticket_id'].isin(resolved_ids)].copy()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 Total", len(unresolved_df))
    with col2:
        st.metric("🔴 Critical", len(unresolved_df[unresolved_df['final_priority'] == 'Critical']))
    with col3:
        st.metric("🚨 Breached", len(unresolved_df[unresolved_df['sla_status'] == 'breached']))
    with col4:
        st.metric("⚠️ At Risk", len(unresolved_df[unresolved_df['sla_status'] == 'at_risk']))
    with col5:
        st.metric("👤 Human", len(unresolved_df[unresolved_df['handler_type'] == 'Human']))
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        priority_filter = st.multiselect("Priority", ['Critical', 'High', 'Medium', 'Low'], default=[])
    with col2:
        sla_filter = st.multiselect("SLA Status", ['breached', 'at_risk', 'met'], default=[])
    with col3:
        sort_by = st.selectbox("Sort", ['Priority', 'SLA Risk', 'Ticket ID'], index=0)
    
    filtered_df = unresolved_df.copy()
    if priority_filter:
        filtered_df = filtered_df[filtered_df['final_priority'].isin(priority_filter)]
    if sla_filter:
        filtered_df = filtered_df[filtered_df['sla_status'].isin(sla_filter)]
    
    if sort_by == 'Priority':
        filtered_df['_sort'] = filtered_df['final_priority'].apply(get_priority_sort_order)
        filtered_df = filtered_df.sort_values('_sort').drop(columns=['_sort'])
    elif sort_by == 'SLA Risk':
        filtered_df['_sort'] = filtered_df['sla_status'].apply(get_sla_sort_order)
        filtered_df = filtered_df.sort_values('_sort').drop(columns=['_sort'])
    
    st.markdown(f"**{len(filtered_df)} tickets**")
    
    urgent_df = filtered_df[(filtered_df['sla_status'].isin(['breached', 'at_risk'])) | (filtered_df['final_priority'] == 'Critical')]
    if not urgent_df.empty:
        st.error(f"🚨 {len(urgent_df)} urgent tickets!")
    
    display_ticket_table(filtered_df, key_prefix="inbox")
    
    st.divider()
    ticket_ids = filtered_df['ticket_id'].tolist()
    if ticket_ids:
        selected = st.selectbox("Select ticket:", ['--'] + ticket_ids, key="inbox_select")
        if selected != '--':
            display_ticket_detail(selected, df, analysis_cache)


def view_ai_handling(df, analysis_cache):
    st.markdown("## 🤖 AI Handling Queue")
    
    resolved_ids = [tid for tid, state in st.session_state.ticket_states.items() if state == TicketState.RESOLVED]
    ai_df = df[(df['handler_type'] == 'AI') & (~df['ticket_id'].isin(resolved_ids))].copy()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 AI Handling", len(ai_df))
    with col2:
        avg_conf = ai_df['priority_confidence'].mean() if not ai_df.empty else 0
        st.metric("📊 Avg Confidence", f"{avg_conf:.1%}")
    with col3:
        st.metric("⚠️ May Escalate", len(ai_df[ai_df['sla_status'] == 'at_risk']))
    
    st.divider()
    
    if ai_df.empty:
        st.info("No tickets in AI queue.")
        return
    
    display_ticket_table(ai_df, show_columns=['ticket_id', 'subject', 'final_priority', 'sla_status', 'issue_type', 'state'], key_prefix="ai")
    
    st.divider()
    selected = st.selectbox("Review:", ['--'] + ai_df['ticket_id'].tolist(), key="ai_select")
    
    if selected != '--':
        result = analysis_cache[selected]
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.text_area("AI Response", value=result.suggested_response, height=250, disabled=True)
        
        with col2:
            st.write(f"Priority: {result.priority_confidence:.1%}")
            if st.button("✅ Approve", type="primary", key=f"approve_{selected}"):
                st.session_state.ticket_states[selected] = TicketState.RESOLVED
                st.success("Approved!")
                st.rerun()
            if st.button("👤 Escalate", key=f"esc_{selected}"):
                st.session_state.ticket_states[selected] = TicketState.WAITING_FOR_HUMAN
                st.warning("Escalated!")
                st.rerun()


def view_human_queue(df, analysis_cache):
    st.markdown("## 👤 Human Queue")
    
    resolved_ids = [tid for tid, state in st.session_state.ticket_states.items() if state == TicketState.RESOLVED]
    human_df = df[(df['handler_type'] == 'Human') & (~df['ticket_id'].isin(resolved_ids))].copy()
    
    escalated_ids = [tid for tid, state in st.session_state.ticket_states.items() if state == TicketState.WAITING_FOR_HUMAN]
    escalated_df = df[df['ticket_id'].isin(escalated_ids)]
    human_df = pd.concat([human_df, escalated_df]).drop_duplicates(subset=['ticket_id'])
    
    human_df['_p'] = human_df['final_priority'].apply(get_priority_sort_order)
    human_df['_s'] = human_df['sla_status'].apply(get_sla_sort_order)
    human_df = human_df.sort_values(['_p', '_s']).drop(columns=['_p', '_s'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Queue", len(human_df))
    with col2:
        st.metric("🔴 Critical", len(human_df[human_df['final_priority'] == 'Critical']))
    with col3:
        st.metric("🚨 Breached", len(human_df[human_df['sla_status'] == 'breached']))
    with col4:
        in_progress = len([tid for tid in human_df['ticket_id'] if st.session_state.ticket_states.get(tid) == TicketState.IN_PROGRESS])
        st.metric("🔄 In Progress", in_progress)
    
    st.divider()
    
    if human_df.empty:
        st.success("🎉 Queue empty!")
        return
    
    urgent = human_df[(human_df['final_priority'] == 'Critical') | (human_df['sla_status'] == 'breached')]
    if not urgent.empty:
        st.error(f"🚨 {len(urgent)} critical/breached tickets!")
    
    display_ticket_table(human_df, show_columns=['ticket_id', 'subject', 'final_priority', 'sla_status', 'issue_type', 'state'], key_prefix="human")
    
    st.divider()
    selected = st.selectbox("Work on:", ['--'] + human_df['ticket_id'].tolist(), key="human_select")
    if selected != '--':
        display_ticket_detail(selected, df, analysis_cache)


def view_analyze_ticket(pipeline, df, analysis_cache):
    st.markdown("## 🔍 Analyze Ticket")
    
    input_method = st.radio("Input", ["Processed tickets", "Manual entry", "Raw dataset"], horizontal=True)
    
    if input_method == "Processed tickets":
        if df is not None and not df.empty:
            ticket_ids = df['ticket_id'].tolist()
            selected = st.selectbox("Select:", ticket_ids)
            if st.button("🔍 View", type="primary"):
                display_ticket_detail(selected, df, analysis_cache)
        else:
            st.warning("Process tickets first.")
    
    elif input_method == "Manual entry":
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Subject", placeholder="e.g., Product issue")
        with col2:
            ticket_id = st.text_input("ID (optional)", placeholder="MANUAL-001")
        
        description = st.text_area("Description", placeholder="Describe the issue...", height=150)
        resolution_time = st.slider("Resolution time (hours)", 0.0, 100.0, 24.0, 1.0)
        
        if st.button("🔍 Analyze", type="primary"):
            if subject or description:
                full_text = f"{subject} | {description}"
                with st.spinner("Analyzing..."):
                    result = pipeline.analyze_ticket(
                        full_text=full_text,
                        ticket_id=ticket_id or f"MANUAL-{datetime.now().strftime('%H%M%S')}",
                        time_to_resolution_hours=resolution_time
                    )
                    st.session_state.analysis_cache[result.ticket_id] = result
                
                st.success("Done!")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Priority", result.final_priority)
                with col2:
                    st.metric("SLA", f"{result.sla_hours}h")
                with col3:
                    st.metric("Status", result.sla_status.upper())
                with col4:
                    st.metric("Handler", result.handler_type)
                
                if result.was_escalated:
                    st.warning(f"⬆️ Escalated: {result.escalation_reason}")
                
                with st.expander("📝 Summary", expanded=True):
                    st.write(result.ticket_summary)
                with st.expander("🔍 Explanation"):
                    st.markdown(result.explanation_text)
                with st.expander("✉️ Response"):
                    st.text_area("Response", value=result.suggested_response, height=250)
            else:
                st.warning("Enter subject or description.")
    
    else:
        raw_df = load_raw_dataset()
        options = raw_df[COL_TICKET_ID].tolist()[:100]
        selected = st.selectbox("Select:", options)
        
        if selected:
            row = raw_df[raw_df[COL_TICKET_ID] == selected].iloc[0]
            st.write(f"**Subject:** {row[COL_SUBJECT]}")
            st.write(f"**Original Priority:** {row[COL_PRIORITY]}")
            
            if st.button("🔍 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    resolution_time = parse_resolution_time(row.get("Time to Resolution", None))
                    result = pipeline.analyze_ticket(
                        full_text=row["full_text"],
                        ticket_id=selected,
                        time_to_resolution_hours=resolution_time
                    )
                    st.session_state.analysis_cache[result.ticket_id] = result
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Priority", result.final_priority)
                with col2:
                    st.metric("SLA", f"{result.sla_hours}h ({result.sla_status})")
                with col3:
                    st.metric("Handler", result.handler_type)
                with col4:
                    st.metric("Escalated", "Yes" if result.was_escalated else "No")


def view_dashboard(df):
    st.markdown("## 📊 Dashboard")
    
    if df is None or df.empty:
        st.warning("No data. Process tickets first.")
        return
    
    total = len(df)
    resolved = len([tid for tid, state in st.session_state.ticket_states.items() if state == TicketState.RESOLVED])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 Total", total)
    with col2:
        st.metric("✅ Resolved", resolved)
    with col3:
        st.metric("📊 Rate", f"{resolved/total*100:.1f}%" if total > 0 else "0%")
    with col4:
        st.metric("🤖 AI", len(df[df['handler_type'] == 'AI']))
    with col5:
        st.metric("👤 Human", len(df[df['handler_type'] == 'Human']))
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Priority Distribution")
        st.bar_chart(df['final_priority'].value_counts())
    with col2:
        st.markdown("### SLA Status")
        st.bar_chart(df['sla_status'].value_counts())
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Handler Distribution")
        st.bar_chart(df['handler_type'].value_counts())
    with col2:
        st.markdown("### Escalation Rate")
        escalated = len(df[df['was_escalated'] == True])
        st.bar_chart(pd.Series({'Escalated': escalated, 'Not Escalated': len(df) - escalated}))
    
    if 'issue_type' in df.columns and df['issue_type'].notna().any():
        st.markdown("### Issue Types")
        st.bar_chart(df['issue_type'].value_counts())


def main():
    init_session_state()
    pipeline, models_loaded = load_pipeline()
    
    if not models_loaded:
        st.error("⚠️ Models not loaded. Run `python train_models.py` first.")
        st.stop()
    
    st.sidebar.title("🎫 AssistFlow AI")
    st.sidebar.divider()
    
    nav = st.sidebar.radio("Navigation", ["📥 Inbox", "🤖 AI Queue", "👤 Human Queue", "🔍 Analyze", "📊 Dashboard"])
    
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Settings")
    
    max_tickets = st.sidebar.slider("Max tickets", 50, 500, 100, 50)
    
    if st.sidebar.button("🔄 Refresh", type="primary"):
        st.session_state.tickets_processed = False
        st.session_state.tickets_df = None
        st.session_state.analysis_cache = {}
        st.session_state.ticket_states = {}
        st.rerun()
    
    raw_df = load_raw_dataset()
    
    if not st.session_state.tickets_processed:
        st.info("🔄 Processing...")
        df = process_all_tickets(pipeline, raw_df, max_tickets)
    else:
        df = st.session_state.tickets_df
    
    if df is not None:
        st.sidebar.markdown("---")
        resolved = len([tid for tid, state in st.session_state.ticket_states.items() if state == TicketState.RESOLVED])
        st.sidebar.metric("Unresolved", len(df) - resolved)
        st.sidebar.metric("Resolved", resolved)
    
    analysis_cache = st.session_state.analysis_cache
    
    if nav == "📥 Inbox":
        view_ticket_inbox(df, analysis_cache)
    elif nav == "🤖 AI Queue":
        view_ai_handling(df, analysis_cache)
    elif nav == "👤 Human Queue":
        view_human_queue(df, analysis_cache)
    elif nav == "🔍 Analyze":
        view_analyze_ticket(pipeline, df, analysis_cache)
    elif nav == "📊 Dashboard":
        view_dashboard(df)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("AssistFlow AI v1.0")


if __name__ == "__main__":
    main()
