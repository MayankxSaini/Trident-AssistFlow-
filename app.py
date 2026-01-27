# ASSISTFLOW AI APPLICATION STRUCTURE
# This app has multiple views:
# 1. Ticket Inbox (default)
# 2. AI Handling Queue
# 3. Human Agent Queue
# 4. Single Ticket Analysis (existing)
"""
AssistFlow AI - Customer Support Operations Dashboard

A real-time ticket management system for support teams.
Run with: streamlit run app.py

VIEWS:
- 📥 Ticket Inbox (Default) - All unresolved tickets
- 🤖 AI Handling - Tickets being handled by AI
- 👤 Human Queue - Tickets requiring human attention
- 🔍 Analyze Ticket - Deep dive into single ticket
- 📊 Dashboard - Summary metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(".")

from src.pipeline import AssistFlowPipeline, TicketAnalysisResult
from src.ingestion import load_and_prepare_data, parse_resolution_time
from src.ticket_state import TicketState, get_initial_state, get_state_icon, get_state_color
from config import DATA_PATH, COL_TICKET_ID, COL_SUBJECT, COL_DESCRIPTION, COL_PRIORITY, COL_TICKET_TYPE

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AssistFlow AI - Support Dashboard",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* Main layout */
    .block-container { padding-top: 1rem; }
    
    /* Header styles */
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    
    /* Priority badges */
    .priority-critical { background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }
    .priority-high { background-color: #fd7e14; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }
    .priority-medium { background-color: #ffc107; color: black; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }
    .priority-low { background-color: #28a745; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }
    
    /* SLA status */
    .sla-breached { background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; }
    .sla-at-risk { background-color: #fd7e14; color: white; padding: 2px 8px; border-radius: 4px; }
    .sla-met { background-color: #28a745; color: white; padding: 2px 8px; border-radius: 4px; }
    
    /* Ticket row highlighting */
    .urgent-ticket { background-color: #fff3cd; border-left: 4px solid #dc3545; padding: 10px; margin: 5px 0; border-radius: 4px; }
    
    /* State badges */
    .state-badge { padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }
    
    /* Metrics */
    .metric-urgent { color: #dc3545; font-weight: bold; }
    
    /* Table styling */
    .dataframe { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    if 'tickets_processed' not in st.session_state:
        st.session_state.tickets_processed = False
    if 'tickets_df' not in st.session_state:
        st.session_state.tickets_df = None
    if 'selected_ticket_id' not in st.session_state:
        st.session_state.selected_ticket_id = None
    if 'ticket_states' not in st.session_state:
        st.session_state.ticket_states = {}
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================
@st.cache_resource
def load_pipeline():
    """Load the ML pipeline (cached)."""
    pipeline = AssistFlowPipeline()
    success = pipeline.load_models()
    return pipeline, success


@st.cache_data(ttl=300)
def load_raw_dataset():
    """Load the raw ticket dataset."""
    return load_and_prepare_data(DATA_PATH)


def process_all_tickets(pipeline, df, max_tickets=200):
    """
    Process all tickets through the pipeline and cache results.
    
    Args:
        pipeline: AssistFlowPipeline instance
        df: Raw DataFrame
        max_tickets: Maximum tickets to process
        
    Returns:
        DataFrame with analysis results
    """
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
            
            # Convert to dict and add extra fields
            result_dict = result.to_dict()
            result_dict['subject'] = row.get(COL_SUBJECT, '')
            result_dict['description'] = row.get(COL_DESCRIPTION, '')[:200]
            result_dict['original_priority'] = row.get(COL_PRIORITY, '')
            result_dict['original_type'] = row.get(COL_TICKET_TYPE, '')
            
            # Determine initial state
            state = get_initial_state(result.handler_type)
            result_dict['state'] = state.value
            result_dict['state_icon'] = get_state_icon(state)
            
            # Store in cache
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
    """Get numeric sort order for priority."""
    order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    return order.get(priority, 4)


def get_sla_sort_order(sla_status):
    """Get numeric sort order for SLA status."""
    order = {'breached': 0, 'at_risk': 1, 'met': 2}
    return order.get(sla_status, 3)


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================
def format_priority(priority):
    """Format priority with color badge."""
    colors = {
        'Critical': '🔴',
        'High': '🟠',
        'Medium': '🟡',
        'Low': '🟢'
    }
    return f"{colors.get(priority, '⚪')} {priority}"


def format_sla_status(status):
    """Format SLA status with icon."""
    icons = {
        'breached': '🚨 BREACHED',
        'at_risk': '⚠️ AT RISK',
        'met': '✅ Met'
    }
    return icons.get(status, status)


def format_handler(handler):
    """Format handler type."""
    if handler == 'Human':
        return '👤 Human'
    return '🤖 AI'


def display_ticket_table(df, show_columns=None, key_prefix="table"):
    """
    Display an interactive ticket table.
    
    Args:
        df: DataFrame with ticket data
        show_columns: List of columns to show
        key_prefix: Unique key prefix for the table
    """
    if df.empty:
        st.info("No tickets to display.")
        return None
    
    # Default columns to display
    if show_columns is None:
        show_columns = [
            'ticket_id', 'subject', 'final_priority', 
            'sla_status', 'handler_type', 'state'
        ]
    
    # Filter to available columns
    available_cols = [c for c in show_columns if c in df.columns]
    display_df = df[available_cols].copy()
    
    # Format columns for display
    if 'final_priority' in display_df.columns:
        display_df['final_priority'] = display_df['final_priority'].apply(format_priority)
    if 'sla_status' in display_df.columns:
        display_df['sla_status'] = display_df['sla_status'].apply(format_sla_status)
    if 'handler_type' in display_df.columns:
        display_df['handler_type'] = display_df['handler_type'].apply(format_handler)
    if 'state' in display_df.columns:
        display_df['state'] = df['state_icon'] + ' ' + display_df['state']
    
    # Rename columns for display
    column_names = {
        'ticket_id': 'Ticket ID',
        'subject': 'Subject',
        'final_priority': 'Priority',
        'sla_status': 'SLA Status',
        'handler_type': 'Handler',
        'state': 'State',
        'issue_type': 'Issue Type'
    }
    display_df = display_df.rename(columns=column_names)
    
    # Truncate subject
    if 'Subject' in display_df.columns:
        display_df['Subject'] = display_df['Subject'].str[:50] + '...'
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    return df['ticket_id'].tolist()


def display_ticket_detail(ticket_id, df, analysis_cache):
    """Display detailed view of a single ticket."""
    
    if ticket_id not in analysis_cache:
        st.error(f"Ticket {ticket_id} not found in cache.")
        return
    
    result = analysis_cache[ticket_id]
    ticket_row = df[df['ticket_id'] == ticket_id].iloc[0] if not df[df['ticket_id'] == ticket_id].empty else None
    
    # Header
    st.markdown(f"### 🎫 Ticket: {ticket_id}")
    
    # Quick Actions
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✅ Mark Resolved", key=f"resolve_{ticket_id}"):
            st.session_state.ticket_states[ticket_id] = TicketState.RESOLVED
            st.success("Ticket marked as resolved!")
            st.rerun()
    with col2:
        if result.handler_type == "AI":
            if st.button("👤 Escalate to Human", key=f"escalate_{ticket_id}"):
                st.session_state.ticket_states[ticket_id] = TicketState.WAITING_FOR_HUMAN
                st.success("Ticket escalated to human!")
                st.rerun()
    with col3:
        current_state = st.session_state.ticket_states.get(ticket_id, TicketState.NEW)
        if current_state == TicketState.WAITING_FOR_HUMAN:
            if st.button("🔄 Start Working", key=f"start_{ticket_id}"):
                st.session_state.ticket_states[ticket_id] = TicketState.IN_PROGRESS
                st.success("Ticket marked as in progress!")
                st.rerun()
    
    st.divider()
    
    # Metrics row
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
    
    # Escalation warning
    if result.was_escalated:
        st.warning(f"⬆️ **ESCALATED**: {result.escalation_reason}")
    
    # Tabs for detail sections
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔍 Analysis", "✉️ Response", "📊 Raw Data"])
    
    with tab1:
        st.markdown("**Ticket Subject:**")
        st.write(ticket_row['subject'] if ticket_row is not None else "N/A")
        
        st.markdown("**Ticket Summary:**")
        st.info(result.ticket_summary)
        
        st.markdown("**Full Description:**")
        with st.expander("Show full text"):
            st.write(result.full_text)
    
    with tab2:
        st.markdown("**Decision Explanation:**")
        st.markdown(result.explanation_text)
        
        st.markdown("**ML Predictions:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- **Initial Priority:** {result.predicted_priority} ({result.priority_confidence:.1%})")
        with col2:
            if result.issue_type:
                st.write(f"- **Issue Type:** {result.issue_type} ({result.issue_type_confidence:.1%})")
    
    with tab3:
        st.markdown("**Suggested Response:**")
        response = st.text_area(
            "Edit before sending:",
            value=result.suggested_response,
            height=300,
            key=f"response_edit_{ticket_id}"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📤 Send Response", type="primary", key=f"send_{ticket_id}"):
                st.success("Response sent! (Simulated)")
                st.session_state.ticket_states[ticket_id] = TicketState.RESOLVED
    
    with tab4:
        st.json(result.to_dict())


# =============================================================================
# VIEW: TICKET INBOX (DEFAULT)
# =============================================================================
def view_ticket_inbox(df, analysis_cache):
    """Display the main ticket inbox - all unresolved tickets."""
    
    st.markdown("## 📥 Ticket Inbox")
    st.markdown("*All unresolved tickets requiring attention*")
    
    # Filter unresolved tickets
    resolved_ids = [tid for tid, state in st.session_state.ticket_states.items() 
                    if state == TicketState.RESOLVED]
    unresolved_df = df[~df['ticket_id'].isin(resolved_ids)].copy()
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📋 Total Unresolved", len(unresolved_df))
    with col2:
        critical_count = len(unresolved_df[unresolved_df['final_priority'] == 'Critical'])
        st.metric("🔴 Critical", critical_count)
    with col3:
        breached_count = len(unresolved_df[unresolved_df['sla_status'] == 'breached'])
        st.metric("🚨 SLA Breached", breached_count)
    with col4:
        at_risk_count = len(unresolved_df[unresolved_df['sla_status'] == 'at_risk'])
        st.metric("⚠️ At Risk", at_risk_count)
    with col5:
        human_count = len(unresolved_df[unresolved_df['handler_type'] == 'Human'])
        st.metric("👤 Need Human", human_count)
    
    st.divider()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=[]
        )
    with col2:
        sla_filter = st.multiselect(
            "Filter by SLA Status",
            options=['breached', 'at_risk', 'met'],
            default=[]
        )
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=['Priority (Critical First)', 'SLA Risk', 'Ticket ID'],
            index=0
        )
    
    # Apply filters
    filtered_df = unresolved_df.copy()
    if priority_filter:
        filtered_df = filtered_df[filtered_df['final_priority'].isin(priority_filter)]
    if sla_filter:
        filtered_df = filtered_df[filtered_df['sla_status'].isin(sla_filter)]
    
    # Apply sorting
    if sort_by == 'Priority (Critical First)':
        filtered_df['_sort'] = filtered_df['final_priority'].apply(get_priority_sort_order)
        filtered_df = filtered_df.sort_values('_sort').drop(columns=['_sort'])
    elif sort_by == 'SLA Risk':
        filtered_df['_sort'] = filtered_df['sla_status'].apply(get_sla_sort_order)
        filtered_df = filtered_df.sort_values('_sort').drop(columns=['_sort'])
    
    # Display ticket table
    st.markdown(f"**Showing {len(filtered_df)} tickets**")
    
    # Highlight urgent tickets
    urgent_df = filtered_df[(filtered_df['sla_status'].isin(['breached', 'at_risk'])) | 
                            (filtered_df['final_priority'] == 'Critical')]
    
    if not urgent_df.empty:
        st.error(f"🚨 **{len(urgent_df)} URGENT TICKETS** require immediate attention!")
    
    # Table
    display_ticket_table(filtered_df, key_prefix="inbox")
    
    # Ticket selection
    st.divider()
    ticket_ids = filtered_df['ticket_id'].tolist()
    if ticket_ids:
        selected = st.selectbox(
            "Select a ticket to view details:",
            options=['-- Select --'] + ticket_ids,
            key="inbox_select"
        )
        
        if selected != '-- Select --':
            display_ticket_detail(selected, df, analysis_cache)


# =============================================================================
# VIEW: AI HANDLING
# =============================================================================
def view_ai_handling(df, analysis_cache):
    """Display tickets being handled by AI."""
    
    st.markdown("## 🤖 AI Handling Queue")
    st.markdown("*Tickets being automatically processed by AI*")
    
    # Filter AI-handled tickets (not resolved)
    resolved_ids = [tid for tid, state in st.session_state.ticket_states.items() 
                    if state == TicketState.RESOLVED]
    ai_df = df[(df['handler_type'] == 'AI') & (~df['ticket_id'].isin(resolved_ids))].copy()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 AI Handling", len(ai_df))
    with col2:
        avg_confidence = ai_df['priority_confidence'].mean() if not ai_df.empty else 0
        st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")
    with col3:
        escalatable = len(ai_df[ai_df['sla_status'] == 'at_risk'])
        st.metric("⚠️ May Need Escalation", escalatable)
    
    st.divider()
    
    if ai_df.empty:
        st.info("No tickets currently in AI handling queue.")
        return
    
    # Display table
    display_ticket_table(
        ai_df,
        show_columns=['ticket_id', 'subject', 'final_priority', 'sla_status', 'issue_type', 'state'],
        key_prefix="ai"
    )
    
    # Ticket details with AI response
    st.divider()
    st.markdown("### 📝 AI Response Preview")
    
    selected_ai = st.selectbox(
        "Select ticket to review AI response:",
        options=['-- Select --'] + ai_df['ticket_id'].tolist(),
        key="ai_select"
    )
    
    if selected_ai != '-- Select --':
        result = analysis_cache[selected_ai]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**AI-Generated Response:**")
            st.text_area(
                "Response",
                value=result.suggested_response,
                height=250,
                disabled=True,
                key=f"ai_response_{selected_ai}"
            )
        
        with col2:
            st.markdown("**Confidence Scores:**")
            st.write(f"Priority: {result.priority_confidence:.1%}")
            if result.issue_type_confidence:
                st.write(f"Issue Type: {result.issue_type_confidence:.1%}")
            
            st.markdown("**Quick Actions:**")
            if st.button("✅ Approve & Send", type="primary", key=f"approve_{selected_ai}"):
                st.session_state.ticket_states[selected_ai] = TicketState.RESOLVED
                st.success("Response approved and sent!")
                st.rerun()
            
            if st.button("👤 Escalate to Human", key=f"esc_{selected_ai}"):
                st.session_state.ticket_states[selected_ai] = TicketState.WAITING_FOR_HUMAN
                st.warning("Escalated to human queue!")
                st.rerun()


# =============================================================================
# VIEW: HUMAN QUEUE
# =============================================================================
def view_human_queue(df, analysis_cache):
    """Display tickets requiring human attention."""
    
    st.markdown("## 👤 Human Queue")
    st.markdown("*Tickets requiring human agent attention - sorted by urgency*")
    
    # Filter human-handled tickets (not resolved)
    resolved_ids = [tid for tid, state in st.session_state.ticket_states.items() 
                    if state == TicketState.RESOLVED]
    human_df = df[(df['handler_type'] == 'Human') & (~df['ticket_id'].isin(resolved_ids))].copy()
    
    # Also include escalated tickets
    escalated_ids = [tid for tid, state in st.session_state.ticket_states.items() 
                     if state == TicketState.WAITING_FOR_HUMAN]
    escalated_df = df[df['ticket_id'].isin(escalated_ids)]
    
    # Combine and deduplicate
    human_df = pd.concat([human_df, escalated_df]).drop_duplicates(subset=['ticket_id'])
    
    # Sort by priority then SLA
    human_df['_priority_sort'] = human_df['final_priority'].apply(get_priority_sort_order)
    human_df['_sla_sort'] = human_df['sla_status'].apply(get_sla_sort_order)
    human_df = human_df.sort_values(['_priority_sort', '_sla_sort']).drop(
        columns=['_priority_sort', '_sla_sort']
    )
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total in Queue", len(human_df))
    with col2:
        critical = len(human_df[human_df['final_priority'] == 'Critical'])
        st.metric("🔴 Critical", critical)
    with col3:
        breached = len(human_df[human_df['sla_status'] == 'breached'])
        st.metric("🚨 SLA Breached", breached)
    with col4:
        in_progress = len([tid for tid in human_df['ticket_id'] 
                          if st.session_state.ticket_states.get(tid) == TicketState.IN_PROGRESS])
        st.metric("🔄 In Progress", in_progress)
    
    st.divider()
    
    if human_df.empty:
        st.success("🎉 No tickets in human queue! All caught up!")
        return
    
    # Urgent alerts
    urgent = human_df[(human_df['final_priority'] == 'Critical') | 
                      (human_df['sla_status'] == 'breached')]
    if not urgent.empty:
        st.error(f"🚨 **{len(urgent)} CRITICAL/BREACHED TICKETS** - Address immediately!")
    
    # Display table
    display_ticket_table(
        human_df,
        show_columns=['ticket_id', 'subject', 'final_priority', 'sla_status', 'issue_type', 'state'],
        key_prefix="human"
    )
    
    # Work on ticket
    st.divider()
    st.markdown("### 🔧 Work on Ticket")
    
    selected_human = st.selectbox(
        "Select ticket to work on:",
        options=['-- Select --'] + human_df['ticket_id'].tolist(),
        key="human_select"
    )
    
    if selected_human != '-- Select --':
        display_ticket_detail(selected_human, df, analysis_cache)


# =============================================================================
# VIEW: ANALYZE SINGLE TICKET (EXISTING FUNCTIONALITY)
# =============================================================================
def view_analyze_ticket(pipeline, df, analysis_cache):
    """Analyze a single ticket - existing functionality preserved."""
    
    st.markdown("## 🔍 Analyze Ticket")
    st.markdown("*Deep dive analysis of a single ticket*")
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Select from processed tickets", "Enter manually", "Select from raw dataset"],
        horizontal=True
    )
    
    if input_method == "Select from processed tickets":
        if df is not None and not df.empty:
            ticket_ids = df['ticket_id'].tolist()
            selected = st.selectbox("Select ticket:", ticket_ids)
            
            if st.button("🔍 View Analysis", type="primary"):
                display_ticket_detail(selected, df, analysis_cache)
        else:
            st.warning("No processed tickets available. Process tickets first from the Inbox.")
    
    elif input_method == "Enter manually":
        col1, col2 = st.columns(2)
        
        with col1:
            ticket_subject = st.text_input("Ticket Subject", placeholder="e.g., Product not working")
        with col2:
            ticket_id = st.text_input("Ticket ID (optional)", placeholder="e.g., MANUAL-001")
        
        ticket_description = st.text_area(
            "Ticket Description",
            placeholder="Describe the customer's issue in detail...",
            height=150
        )
        
        resolution_time = st.slider(
            "Simulated Resolution Time (hours)",
            min_value=0.0, max_value=100.0, value=24.0, step=1.0,
            help="For SLA status simulation"
        )
        
        if st.button("🔍 Analyze Ticket", type="primary"):
            if ticket_subject or ticket_description:
                full_text = f"{ticket_subject} | {ticket_description}"
                
                with st.spinner("Analyzing ticket..."):
                    result = pipeline.analyze_ticket(
                        full_text=full_text,
                        ticket_id=ticket_id or f"MANUAL-{datetime.now().strftime('%H%M%S')}",
                        time_to_resolution_hours=resolution_time
                    )
                    st.session_state.analysis_cache[result.ticket_id] = result
                
                st.success("Analysis complete!")
                
                # Display result
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Priority", result.final_priority)
                with col2:
                    st.metric("SLA Hours", result.sla_hours)
                with col3:
                    st.metric("SLA Status", result.sla_status.upper())
                with col4:
                    st.metric("Handler", result.handler_type)
                
                if result.was_escalated:
                    st.warning(f"⬆️ Escalated: {result.escalation_reason}")
                
                with st.expander("📝 Ticket Summary", expanded=True):
                    st.write(result.ticket_summary)
                
                with st.expander("🔍 Decision Explanation"):
                    st.markdown(result.explanation_text)
                
                with st.expander("✉️ Suggested Response"):
                    st.text_area("Response", value=result.suggested_response, height=250)
            else:
                st.warning("Please enter a subject or description.")
    
    else:  # Select from raw dataset
        raw_df = load_raw_dataset()
        ticket_options = raw_df[COL_TICKET_ID].tolist()[:100]
        selected_raw = st.selectbox("Select ticket:", ticket_options)
        
        if selected_raw:
            ticket_row = raw_df[raw_df[COL_TICKET_ID] == selected_raw].iloc[0]
            
            st.markdown("**Selected Ticket:**")
            st.write(f"- **Subject:** {ticket_row[COL_SUBJECT]}")
            st.write(f"- **Original Priority:** {ticket_row[COL_PRIORITY]}")
            
            if st.button("🔍 Analyze This Ticket", type="primary"):
                with st.spinner("Analyzing..."):
                    resolution_time = parse_resolution_time(ticket_row.get("Time to Resolution", None))
                    result = pipeline.analyze_ticket(
                        full_text=ticket_row["full_text"],
                        ticket_id=selected_raw,
                        time_to_resolution_hours=resolution_time
                    )
                    st.session_state.analysis_cache[result.ticket_id] = result
                
                # Quick display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Priority", result.final_priority)
                with col2:
                    st.metric("SLA", f"{result.sla_hours}h ({result.sla_status})")
                with col3:
                    st.metric("Handler", result.handler_type)
                with col4:
                    st.metric("Escalated", "Yes" if result.was_escalated else "No")


# =============================================================================
# VIEW: DASHBOARD
# =============================================================================
def view_dashboard(df):
    """Display summary dashboard with metrics."""
    
    st.markdown("## 📊 Operations Dashboard")
    st.markdown("*Overview of support operations metrics*")
    
    if df is None or df.empty:
        st.warning("No data available. Process tickets from the Inbox first.")
        return
    
    # Summary metrics row 1
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(df)
    resolved = len([tid for tid, state in st.session_state.ticket_states.items() 
                    if state == TicketState.RESOLVED])
    
    with col1:
        st.metric("📋 Total Processed", total)
    with col2:
        st.metric("✅ Resolved", resolved)
    with col3:
        st.metric("📊 Resolution Rate", f"{resolved/total*100:.1f}%" if total > 0 else "0%")
    with col4:
        ai_handled = len(df[df['handler_type'] == 'AI'])
        st.metric("🤖 AI Handled", ai_handled)
    with col5:
        human_handled = len(df[df['handler_type'] == 'Human'])
        st.metric("👤 Human Required", human_handled)
    
    st.divider()
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Priority Distribution")
        priority_counts = df['final_priority'].value_counts()
        st.bar_chart(priority_counts)
    
    with col2:
        st.markdown("### SLA Status Distribution")
        sla_counts = df['sla_status'].value_counts()
        st.bar_chart(sla_counts)
    
    # More charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Handler Distribution")
        handler_counts = df['handler_type'].value_counts()
        st.bar_chart(handler_counts)
    
    with col2:
        st.markdown("### Escalation Rate")
        escalated = len(df[df['was_escalated'] == True])
        not_escalated = len(df[df['was_escalated'] == False])
        st.bar_chart(pd.Series({'Escalated': escalated, 'Not Escalated': not_escalated}))
    
    # Issue type breakdown
    if 'issue_type' in df.columns and df['issue_type'].notna().any():
        st.markdown("### Issue Type Distribution")
        type_counts = df['issue_type'].value_counts()
        st.bar_chart(type_counts)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Load pipeline
    pipeline, models_loaded = load_pipeline()
    
    if not models_loaded:
        st.error("⚠️ Models not loaded. Please run `python train_models.py` first.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🎫 AssistFlow AI")
    st.sidebar.markdown("*Customer Support Operations*")
    st.sidebar.divider()
    
    # Navigation
    nav_options = {
        "📥 Ticket Inbox": "inbox",
        "🤖 AI Handling": "ai",
        "👤 Human Queue": "human",
        "🔍 Analyze Ticket": "analyze",
        "📊 Dashboard": "dashboard"
    }
    
    selected_nav = st.sidebar.radio(
        "Navigation",
        options=list(nav_options.keys()),
        index=0
    )
    
    current_view = nav_options[selected_nav]
    
    st.sidebar.divider()
    
    # Processing controls
    st.sidebar.markdown("### ⚙️ Data Processing")
    
    max_tickets = st.sidebar.slider(
        "Max tickets to process",
        min_value=50, max_value=500, value=100, step=50
    )
    
    if st.sidebar.button("🔄 Process Tickets", type="primary"):
        st.session_state.tickets_processed = False
        st.session_state.tickets_df = None
        st.session_state.analysis_cache = {}
        st.session_state.ticket_states = {}
        st.rerun()
    
    # Process tickets if needed
    raw_df = load_raw_dataset()
    
    if not st.session_state.tickets_processed:
        st.info("🔄 Processing tickets... This may take a moment.")
        df = process_all_tickets(pipeline, raw_df, max_tickets)
    else:
        df = st.session_state.tickets_df
    
    # Status in sidebar
    if df is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Quick Stats")
        resolved = len([tid for tid, state in st.session_state.ticket_states.items() 
                        if state == TicketState.RESOLVED])
        unresolved = len(df) - resolved
        st.sidebar.metric("Unresolved", unresolved)
        st.sidebar.metric("Resolved", resolved)
    
    # Render current view
    analysis_cache = st.session_state.analysis_cache
    
    if current_view == "inbox":
        view_ticket_inbox(df, analysis_cache)
    elif current_view == "ai":
        view_ai_handling(df, analysis_cache)
    elif current_view == "human":
        view_human_queue(df, analysis_cache)
    elif current_view == "analyze":
        view_analyze_ticket(pipeline, df, analysis_cache)
    elif current_view == "dashboard":
        view_dashboard(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("AssistFlow AI v1.0")


if __name__ == "__main__":
    main()
