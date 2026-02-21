"""
HR Analytics Dashboard
Built with Plotly Dash for HR leadership insights.
Shows query trends, FAQ patterns, escalation rates, department distribution.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, callback
import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE = f"http://localhost:{os.getenv('API_PORT', 8000)}"
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8050))

# ─── Mock Data (used when API not available) ──────────────────────────────────

MOCK_DATA = {
    "total_queries": 1247,
    "queries_today": 43,
    "avg_confidence": 0.78,
    "escalation_rate": 0.12,
    "resolution_rate": 0.88,
    "avg_response_time_ms": 2340,
    "pending_escalations": 14,
    "category_distribution": {
        "Leave Policy": 387, "Reimbursement": 218, "Insurance": 156,
        "Payroll": 134, "Onboarding": 98, "Benefits": 87,
        "Remote Work": 76, "Performance": 54,
    },
    "department_distribution": {
        "Engineering": 312, "Sales": 198, "Marketing": 176, "HR": 145,
        "Finance": 134, "Operations": 112, "Product": 89, "Legal": 81,
    },
    "daily_trends": [
        {"date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
         "queries": 35 + (i * 3 % 15), "escalations": 4 + (i % 5)}
        for i in range(14, -1, -1)
    ],
    "top_faq": [
        {"question": "How do I apply for casual leave?", "count": 89},
        {"question": "What is the medical reimbursement limit?", "count": 76},
        {"question": "How many sick leaves per year?", "count": 68},
        {"question": "How to claim travel expenses?", "count": 61},
        {"question": "What are WFH policy guidelines?", "count": 54},
        {"question": "When is the performance review cycle?", "count": 47},
        {"question": "How to add dependents to insurance?", "count": 43},
        {"question": "What is the probation period?", "count": 38},
    ],
}

# ─── Color Palette ────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#4F46E5",    # Indigo
    "success": "#10B981",    # Emerald
    "warning": "#F59E0B",    # Amber
    "danger": "#EF4444",     # Red
    "info": "#3B82F6",       # Blue
    "bg": "#F8FAFC",         # Light gray
    "card": "#FFFFFF",
    "text": "#1E293B",
    "muted": "#64748B",
}

CHART_COLORS = [
    "#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#3B82F6",
    "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#6366F1"
]

# ─── App Init ─────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="HR Analytics Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)


# ─── Layout Helpers ───────────────────────────────────────────────────────────

def metric_card(title, value, subtitle="", color=COLORS["primary"], icon="📊"):
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "24px"}),
            html.H3(str(value), style={
                "margin": "8px 0 4px",
                "fontSize": "2rem",
                "fontWeight": "700",
                "color": color,
            }),
            html.P(title, style={"margin": "0", "fontWeight": "600", "color": COLORS["text"]}),
            html.P(subtitle, style={"margin": "4px 0 0", "fontSize": "0.8rem", "color": COLORS["muted"]}),
        ])
    ], style={
        "background": COLORS["card"],
        "borderRadius": "12px",
        "padding": "20px 24px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
        "borderLeft": f"4px solid {color}",
        "flex": "1",
        "minWidth": "200px",
    })


# ─── Charts ───────────────────────────────────────────────────────────────────

def build_query_trend_chart(daily_trends):
    df = pd.DataFrame(daily_trends)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["queries"],
        mode="lines+markers",
        name="Total Queries",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(79,70,229,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["escalations"],
        mode="lines+markers",
        name="Escalations",
        line=dict(color=COLORS["danger"], width=2, dash="dash"),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title="Query Volume & Escalations (Last 15 Days)",
        xaxis_title="Date",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    )
    return fig


def build_category_donut(category_dist):
    labels = list(category_dist.keys())
    values = list(category_dist.values())
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=CHART_COLORS),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        title="Queries by HR Category",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        annotations=[dict(text=f"{sum(values)}<br>Total", x=0.5, y=0.5,
                          font_size=16, showarrow=False, font_color=COLORS["primary"])]
    )
    return fig


def build_department_bar(dept_dist):
    departments = list(dept_dist.keys())
    values = list(dept_dist.values())
    sorted_pairs = sorted(zip(values, departments), reverse=True)
    values, departments = zip(*sorted_pairs)

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(departments),
        orientation="h",
        marker=dict(
            color=list(values),
            colorscale=[[0, "#E0E7FF"], [1, COLORS["primary"]]],
            showscale=False,
        ),
        text=list(values),
        textposition="outside",
    ))
    fig.update_layout(
        title="Queries by Department",
        xaxis_title="Query Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=60, t=50, b=20),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    )
    return fig


def build_faq_table(top_faq):
    rows = []
    for i, item in enumerate(top_faq, 1):
        max_count = top_faq[0]["count"]
        bar_width = (item["count"] / max_count) * 100
        rows.append(html.Tr([
            html.Td(f"#{i}", style={"color": COLORS["muted"], "fontSize": "0.85rem", "paddingRight": "12px"}),
            html.Td(item["question"], style={"flex": "1"}),
            html.Td([
                html.Div([
                    html.Div(style={
                        "width": f"{bar_width}%", "height": "6px",
                        "background": COLORS["primary"], "borderRadius": "3px",
                        "minWidth": "4px",
                    }),
                    html.Span(f" {item['count']}", style={"fontSize": "0.85rem", "color": COLORS["muted"], "marginLeft": "8px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"})
            ], style={"minWidth": "150px"}),
        ], style={"borderBottom": "1px solid #F1F5F9"}))
    return html.Table([
        html.Thead(html.Tr([
            html.Th("#", style={"width": "30px"}),
            html.Th("Frequently Asked Question"),
            html.Th("Frequency"),
        ], style={"fontSize": "0.8rem", "color": COLORS["muted"], "textTransform": "uppercase",
                  "letterSpacing": "0.05em", "paddingBottom": "8px"})),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.9rem"})


# ─── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div([

    # Header
    html.Div([
        html.Div([
            html.H1("🏢 HR Intelligence Dashboard", style={
                "margin": "0", "fontSize": "1.5rem", "fontWeight": "700", "color": "white"
            }),
            html.P("Enterprise HR Query Analytics & Insights", style={
                "margin": "2px 0 0", "color": "rgba(255,255,255,0.75)", "fontSize": "0.85rem"
            }),
        ]),
        html.Div([
            html.Span(id="last-updated", style={"color": "rgba(255,255,255,0.7)", "fontSize": "0.8rem"}),
            html.Button("🔄 Refresh", id="refresh-btn", n_clicks=0, style={
                "marginLeft": "12px", "background": "rgba(255,255,255,0.15)",
                "border": "1px solid rgba(255,255,255,0.3)", "color": "white",
                "borderRadius": "6px", "padding": "6px 14px", "cursor": "pointer",
                "fontSize": "0.85rem",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "background": f"linear-gradient(135deg, {COLORS['primary']}, #7C3AED)",
        "padding": "20px 30px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
    }),

    # Main Content
    html.Div([

        # Metric Cards Row
        html.Div(id="metric-cards", style={
            "display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"
        }),

        # Charts Row 1
        html.Div([
            html.Div([
                dcc.Graph(id="trend-chart", config={"displayModeBar": False}),
            ], style={
                "flex": "2", "background": COLORS["card"], "borderRadius": "12px",
                "padding": "16px", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"
            }),
            html.Div([
                dcc.Graph(id="category-chart", config={"displayModeBar": False}),
            ], style={
                "flex": "1", "background": COLORS["card"], "borderRadius": "12px",
                "padding": "16px", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"
            }),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # Charts Row 2
        html.Div([
            html.Div([
                dcc.Graph(id="dept-chart", config={"displayModeBar": False}),
            ], style={
                "flex": "1", "background": COLORS["card"], "borderRadius": "12px",
                "padding": "16px", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"
            }),
            html.Div([
                html.H3("🔥 Top Frequently Asked Questions", style={
                    "margin": "0 0 16px", "fontSize": "1rem", "fontWeight": "600",
                    "color": COLORS["text"]
                }),
                html.Div(id="faq-table"),
            ], style={
                "flex": "1.2", "background": COLORS["card"], "borderRadius": "12px",
                "padding": "20px 24px", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"
            }),
        ], style={"display": "flex", "gap": "16px"}),

    ], style={"padding": "24px 30px", "background": COLORS["bg"], "minHeight": "calc(100vh - 80px)"}),

    # Interval for auto-refresh
    dcc.Interval(id="interval", interval=60_000, n_intervals=0),
    dcc.Store(id="analytics-store"),

], style={"fontFamily": "Inter, -apple-system, BlinkMacSystemFont, sans-serif"})


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("analytics-store", "data"),
    Input("interval", "n_intervals"),
    Input("refresh-btn", "n_clicks"),
)
def fetch_data(n_intervals, n_clicks):
    """Fetch analytics data from API or use mock."""
    try:
        resp = httpx.get(
            f"{API_BASE}/analytics/overview",
            headers={"X-Employee-Role": "hr_admin"},
            timeout=5.0
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return MOCK_DATA


@app.callback(
    Output("metric-cards", "children"),
    Output("trend-chart", "figure"),
    Output("category-chart", "figure"),
    Output("dept-chart", "figure"),
    Output("faq-table", "children"),
    Output("last-updated", "children"),
    Input("analytics-store", "data"),
)
def update_dashboard(data):
    if not data:
        data = MOCK_DATA

    # Metric cards
    cards = [
        metric_card("Total Queries", f"{data['total_queries']:,}", "All time", COLORS["primary"], "💬"),
        metric_card("Queries Today", data["queries_today"], "vs 38 yesterday ↑", COLORS["info"], "📈"),
        metric_card("Avg Confidence", f"{data['avg_confidence']*100:.0f}%", "Response quality", COLORS["success"], "✅"),
        metric_card("Escalation Rate", f"{data['escalation_rate']*100:.0f}%", f"{data['pending_escalations']} pending", COLORS["warning"], "⚠️"),
        metric_card("Resolution Rate", f"{data['resolution_rate']*100:.0f}%", "Successfully resolved", COLORS["success"], "🎯"),
        metric_card("Avg Response", f"{data['avg_response_time_ms']/1000:.1f}s", "Processing time", COLORS["info"], "⚡"),
    ]

    trend_fig = build_query_trend_chart(data["daily_trends"])
    category_fig = build_category_donut(data["category_distribution"])
    dept_fig = build_department_bar(data["department_distribution"])
    faq_table = build_faq_table(data["top_faq"])
    last_updated = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

    return cards, trend_fig, category_fig, dept_fig, faq_table, last_updated


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🚀 HR Analytics Dashboard running at http://localhost:{DASHBOARD_PORT}")
    app.run(debug=False, port=DASHBOARD_PORT, host="0.0.0.0")
