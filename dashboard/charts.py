"""
Plotly Chart Generators for CyberWorld-AI Streamlit Dashboard.
Creates dark SOC control-room interactive charts for risk gauges, forecasts, SHAP feature contributions, and temporal attention.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Dark SOC theme palette
BG_COLOR = "#111827"
CARD_BG = "#1F2937"
TEXT_COLOR = "#F9FAFB"
GRID_COLOR = "#374151"

def create_risk_gauge_chart(risk_score: float) -> go.Figure:
    """Generates 0-100 Plotly Gauge chart for current risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Risk Score (0-100)", 'font': {'size': 18, 'color': TEXT_COLOR}},
        number={'suffix': " / 100", 'font': {'size': 24, 'color': TEXT_COLOR}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': TEXT_COLOR},
            'bar': {'color': "#3B82F6" if risk_score < 40 else ("#F59E0B" if risk_score < 70 else "#EF4444")},
            'bgcolor': CARD_BG,
            'borderwidth': 1,
            'bordercolor': GRID_COLOR,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},   # Low (Green)
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},  # Moderate (Yellow)
                {'range': [60, 80], 'color': 'rgba(249, 115, 22, 0.2)'},  # High (Orange)
                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.2)'}   # Critical (Red)
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font={'color': TEXT_COLOR},
        margin=dict(l=20, r=20, t=40, b=20),
        height=250
    )
    return fig

def create_future_risk_chart(future_risks: list) -> go.Figure:
    """Generates line chart for 5-step recursive World Model future risk forecast."""
    steps = ["Current", "+5 sec", "+10 sec", "+15 sec", "+20 sec", "+25 sec"]
    values = [future_risks[0]] + list(future_risks) if len(future_risks) == 5 else list(future_risks)
    if len(values) > 6:
        values = values[:6]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=values,
        mode="lines+markers",
        name="Predicted Risk",
        line=dict(color="#F59E0B", width=3),
        marker=dict(size=8, color="#F59E0B")
    ))
    fig.update_layout(
        title="5-Step World Model Future Risk Trajectory",
        xaxis_title="Simulation Horizon",
        yaxis_title="Predicted Risk Score (0-100)",
        yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=50, b=40),
        height=300
    )
    return fig

def create_attack_prob_chart(current_prob: float, future_probs: list) -> go.Figure:
    """Generates bar/line chart for predicted attack probability progression."""
    steps = ["Current", "+5 sec", "+10 sec", "+15 sec", "+20 sec", "+25 sec"]
    probs = [current_prob * 100.0] + [p * 100.0 for p in future_probs]
    if len(probs) > 6:
        probs = probs[:6]
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=steps,
        y=probs,
        name="Attack Probability (%)",
        marker_color=["#10B981" if p < 40 else ("#F59E0B" if p < 70 else "#EF4444") for p in probs]
    ))
    fig.update_layout(
        title="Predicted Attack Probability Progression (%)",
        xaxis_title="Simulation Horizon",
        yaxis_title="Attack Probability (%)",
        yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=50, b=40),
        height=300
    )
    return fig

def create_shap_bar_chart(top_shap_features: list, top_n: int = 10) -> go.Figure:
    """Generates horizontal bar chart of local SHAP feature contributions."""
    display_feats = top_shap_features[:top_n][::-1]
    
    names = [f["feature"] for f in display_feats]
    s_vals = [f["shap_value"] for f in display_feats]
    colors = ["#EF4444" if v > 0 else "#3B82F6" for v in s_vals]
    
    fig = go.Figure(go.Bar(
        x=s_vals,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in s_vals],
        textposition="auto"
    ))
    fig.update_layout(
        title=f"Top {len(names)} Local Feature Contributions (SHAP Explanation)",
        xaxis_title="SHAP Contribution (+ Increases Risk / - Decreases Risk)",
        yaxis_title="Feature Name",
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=150, r=20, t=50, b=40),
        height=400
    )
    return fig

def create_shap_group_chart(group_importance: dict) -> go.Figure:
    """Generates bar chart of SHAP macro feature group share percentages."""
    groups = list(group_importance.keys())
    shares = [group_importance[g]["percentage_share"] for g in groups]
    
    fig = go.Figure(go.Bar(
        x=groups,
        y=shares,
        marker_color="#8B5CF6",
        text=[f"{s:.1f}%" for s in shares],
        textposition="auto"
    ))
    fig.update_layout(
        title="SHAP Macro Feature Category Importance Share (%)",
        xaxis_title="Feature Category",
        yaxis_title="Importance Share (%)",
        yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    return fig

def create_attention_chart(timesteps: list, weights: list, max_label: str) -> go.Figure:
    """Generates bar chart of PyTorch World Model temporal window attention weights."""
    colors = ["#EF4444" if t == max_label else "#3B82F6" for t in timesteps]
    pct_weights = [w * 100.0 for w in weights]
    
    fig = go.Figure(go.Bar(
        x=timesteps,
        y=pct_weights,
        marker_color=colors,
        text=[f"{pw:.1f}%" for pw in pct_weights],
        textposition="auto"
    ))
    fig.update_layout(
        title="PyTorch LSTM World Model Temporal Window Attention (t-9 ... t)",
        xaxis_title="Historical Network State Windows",
        yaxis_title="Attention Weight (%)",
        yaxis=dict(range=[0, max(pct_weights) * 1.3], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=50, b=40),
        height=300
    )
    return fig

# --- REAL-TIME LIVE SOC CHARTS (5-MINUTE ROLLING HISTORY) ---

def create_pps_chart(df_rolling: pd.DataFrame) -> go.Figure:
    """Generates real-time line chart for Packets Per Second (5-min rolling history)."""
    fig = go.Figure()
    if df_rolling is not None and not df_rolling.empty and "pps" in df_rolling.columns:
        x_vals = df_rolling["timestamp"]
        y_vals = df_rolling["pps"]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            name="Packets/sec",
            line=dict(color="#06B6D4", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.15)"
        ))
    else:
        fig.add_annotation(text="Awaiting live traffic telemetry...", showarrow=False, font=dict(color=TEXT_COLOR, size=14))
        
    fig.update_layout(
        title="1. Packets Per Second (PPS Trend)",
        xaxis_title="Time",
        yaxis_title="Packets / Sec",
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=45, b=35),
        height=260
    )
    return fig

def create_active_connections_chart(df_rolling: pd.DataFrame) -> go.Figure:
    """Generates real-time chart for Concurrent Active Flows / Connections."""
    fig = go.Figure()
    if df_rolling is not None and not df_rolling.empty and "active_connections" in df_rolling.columns:
        x_vals = df_rolling["timestamp"]
        y_vals = df_rolling["active_connections"]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            name="Active Connections",
            line=dict(color="#8B5CF6", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.15)"
        ))
    else:
        fig.add_annotation(text="Awaiting live connection telemetry...", showarrow=False, font=dict(color=TEXT_COLOR, size=14))
        
    fig.update_layout(
        title="2. Concurrent Active Connections",
        xaxis_title="Time",
        yaxis_title="Flow Count",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=45, b=35),
        height=260
    )
    return fig

def create_bandwidth_chart(df_rolling: pd.DataFrame) -> go.Figure:
    """Generates real-time bandwidth usage chart in Mbps."""
    fig = go.Figure()
    if df_rolling is not None and not df_rolling.empty and "mbps" in df_rolling.columns:
        x_vals = df_rolling["timestamp"]
        y_vals = df_rolling["mbps"]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            name="Bandwidth (Mbps)",
            line=dict(color="#10B981", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.15)"
        ))
    else:
        fig.add_annotation(text="Awaiting bandwidth telemetry...", showarrow=False, font=dict(color=TEXT_COLOR, size=14))
        
    fig.update_layout(
        title="3. Network Bandwidth Telemetry (Mbps)",
        xaxis_title="Time",
        yaxis_title="Throughput (Mbps)",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=45, b=35),
        height=260
    )
    return fig

def create_risk_trend_chart(df_rolling: pd.DataFrame) -> go.Figure:
    """Generates real-time risk score trend chart (0-100) with threshold zones."""
    fig = go.Figure()
    if df_rolling is not None and not df_rolling.empty and "risk_score" in df_rolling.columns:
        x_vals = df_rolling["timestamp"]
        y_vals = df_rolling["risk_score"]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            name="Risk Score",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=5, color="#F59E0B")
        ))
        # Danger and High Risk horizontal guide lines
        fig.add_hline(y=75, line_dash="dot", line_color="#EF4444", annotation_text="Danger (75)", annotation_position="top right")
        fig.add_hline(y=40, line_dash="dot", line_color="#F59E0B", annotation_text="Risk (40)", annotation_position="bottom right")
    else:
        fig.add_annotation(text="Awaiting risk analysis...", showarrow=False, font=dict(color=TEXT_COLOR, size=14))
        
    fig.update_layout(
        title="4. Risk Score Temporal Trend (0-100)",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 105], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=45, b=35),
        height=260
    )
    return fig

def create_attack_prob_trend_chart(df_rolling: pd.DataFrame) -> go.Figure:
    """Generates real-time attack probability trend chart (%)."""
    fig = go.Figure()
    if df_rolling is not None and not df_rolling.empty and "attack_probability" in df_rolling.columns:
        x_vals = df_rolling["timestamp"]
        y_vals = [p * 100.0 for p in df_rolling["attack_probability"]]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            name="Attack Prob (%)",
            line=dict(color="#F43F5E", width=2.5),
            marker=dict(size=5, color="#F43F5E"),
            fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.12)"
        ))
    else:
        fig.add_annotation(text="Awaiting threat probability...", showarrow=False, font=dict(color=TEXT_COLOR, size=14))
        
    fig.update_layout(
        title="5. Attack Probability Progression (%)",
        xaxis_title="Time",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 105], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=40, r=20, t=45, b=35),
        height=260
    )
    return fig

def create_protocol_distribution_chart(proto_dist: dict) -> go.Figure:
    """Generates responsive Donut chart of Protocol Distribution (TCP, UDP, ICMP, ARP)."""
    labels = list(proto_dist.keys()) if proto_dist else ["TCP", "UDP", "ICMP", "ARP"]
    values = [proto_dist[k]["count"] if proto_dist else 0 for k in labels]
    
    # Check if all zeros
    if sum(values) == 0:
        values = [75, 20, 3, 2] # baseline proportions
        
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#6B7280"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
        textinfo="label+percent",
        insidetextorientation="radial"
    )])
    
    fig.update_layout(
        title="6. Protocol Distribution",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=20, r=20, t=45, b=20),
        height=260,
        showlegend=False
    )
    return fig

def create_top_dst_ips_chart(top_dst: list) -> go.Figure:
    """Generates horizontal bar chart for Top Destination IPs."""
    fig = go.Figure()
    if top_dst:
        ips = [item[0] for item in top_dst][::-1]
        counts = [item[1] for item in top_dst][::-1]
        fig.add_trace(go.Bar(
            x=counts,
            y=ips,
            orientation="h",
            marker=dict(color="#3B82F6", line=dict(color="#1D4ED8", width=1)),
            text=[str(c) for c in counts],
            textposition="auto"
        ))
    else:
        fig.add_annotation(text="No destination IP data", showarrow=False, font=dict(color=TEXT_COLOR))
        
    fig.update_layout(
        title="7. Top Destination IPs",
        xaxis_title="Packet Count",
        yaxis_title="IP Address",
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=110, r=20, t=45, b=35),
        height=260
    )
    return fig

def create_top_src_ips_chart(top_src: list) -> go.Figure:
    """Generates horizontal bar chart for Top Source IPs."""
    fig = go.Figure()
    if top_src:
        ips = [item[0] for item in top_src][::-1]
        counts = [item[1] for item in top_src][::-1]
        fig.add_trace(go.Bar(
            x=counts,
            y=ips,
            orientation="h",
            marker=dict(color="#10B981", line=dict(color="#047857", width=1)),
            text=[str(c) for c in counts],
            textposition="auto"
        ))
    else:
        fig.add_annotation(text="No source IP data", showarrow=False, font=dict(color=TEXT_COLOR))
        
    fig.update_layout(
        title="8. Top Source IPs",
        xaxis_title="Packet Count",
        yaxis_title="IP Address",
        xaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=110, r=20, t=45, b=35),
        height=260
    )
    return fig

def create_topology_chart(topology_data: dict) -> go.Figure:
    """
    Generates interactive network topology graph showing Local Machine,
    Connected Gateway, and Internet Nodes with active flow links.
    """
    nodes = topology_data.get("nodes", [])
    links = topology_data.get("links", [])
    
    fig = go.Figure()
    
    # Calculate positions for nodes in a radial / tiered network layout
    pos = {}
    pos["local"] = (0.0, 0.0)
    pos["gateway"] = (0.0, 0.45)
    
    ext_idx = 0
    num_ext = max(len([n for n in nodes if n.get("type") == "external"]), 1)
    
    node_x, node_y, node_colors, node_sizes, node_labels, node_hover = [], [], [], [], [], []
    
    for n in nodes:
        ntype = n.get("type", "external")
        nid = n.get("id")
        if ntype == "local":
            x, y = 0.0, -0.4
            color = "#10B981" # Green
            size = 28
        elif ntype == "gateway":
            x, y = 0.0, 0.1
            color = "#3B82F6" # Blue
            size = 24
        else:
            # Spread external nodes in upper arc
            angle = np.pi * 0.15 + (np.pi * 0.7) * (ext_idx / max(num_ext - 1, 1))
            radius = 0.75
            x = radius * np.cos(angle)
            y = 0.1 + radius * np.sin(angle)
            ext_idx += 1
            color = "#F59E0B" if not nid.startswith("104.") else "#EF4444"
            size = 18
            
        pos[nid] = (x, y)
        node_x.append(x)
        node_y.append(y)
        node_colors.append(color)
        node_sizes.append(size)
        node_labels.append(n.get("label", nid))
        node_hover.append(f"<b>{n.get('label', nid)}</b><br>Type: {ntype.upper()}<br>Status: Active")

    # Draw edge lines
    for l in links:
        s_pos = pos.get(l["source"], (0, 0))
        t_pos = pos.get(l["target"], (0, 0))
        
        status_color = "#10B981" if l.get("status") == "Normal" else "#EF4444"
        line_width = max(1.5, min(5.0, l.get("packets", 10) / 20.0))
        
        fig.add_trace(go.Scatter(
            x=[s_pos[0], t_pos[0]],
            y=[s_pos[1], t_pos[1]],
            mode="lines",
            line=dict(color=status_color, width=line_width),
            hoverinfo="text",
            text=f"Flow: {l['source']} ➔ {l['target']}<br>Packets: {l.get('packets', 0)} | Bytes: {l.get('bytes', 0)}<br>Status: {l.get('status', 'Normal')}",
            showlegend=False
        ))
        
    # Draw nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors, line=dict(color=TEXT_COLOR, width=1.5)),
        text=[l.split("(")[0].strip() for l in node_labels],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_hover,
        showlegend=False
    ))
    
    fig.update_layout(
        title="Live Network Node Topology & Telemetry Flow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_BG,
        font={'color': TEXT_COLOR},
        margin=dict(l=20, r=20, t=50, b=20),
        height=380
    )
    return fig

