# =========================================
# NovaRetail Dashboard — app.py (LOCAL)
# =========================================
# Requirements:
#   pip install dash dash-bootstrap-components plotly pandas
#
# Files required in same folder:
#   campaign_oct2025.csv
#   leads_crm_oct2025.csv
#
# Folder structure:
#   dashboard/
#     app.py
#     campaign_oct2025.csv
#     leads_crm_oct2025.csv
#     assets/
#       styles.css
# =========================================

import os
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


# ============================================================
# 1) DATA LOADING (LOCAL CSV)
# ============================================================
REQUIRED_FILES = ["campaign_oct2025.csv", "leads_crm_oct2025.csv"]
missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        f"Fichiers manquants dans le dossier courant: {missing}\n"
        "Assure-toi que campaign_oct2025.csv et leads_crm_oct2025.csv sont dans le meme dossier que app.py."
    )

df_campaign = pd.read_csv("campaign_oct2025.csv", parse_dates=["date"])
df_leads = pd.read_csv("leads_crm_oct2025.csv", parse_dates=["date"])

# Perimetre (octobre 2025)
START = pd.Timestamp("2025-10-01")
END = pd.Timestamp("2025-10-31")

df_campaign = df_campaign[(df_campaign["date"] >= START) & (df_campaign["date"] <= END)].copy()
df_leads = df_leads[(df_leads["date"] >= START) & (df_leads["date"] <= END)].copy()

# Normaliser labels (robuste)
if "channel" in df_campaign.columns:
    df_campaign["channel"] = df_campaign["channel"].astype(str)

for col in ["channel", "device", "status", "company_size", "sector", "region"]:
    if col in df_leads.columns:
        df_leads[col] = df_leads[col].astype("object")

CHANNELS = sorted(df_leads["channel"].dropna().unique().tolist())
STATUSES = sorted(df_leads["status"].dropna().unique().tolist())
DEFAULT_DAY = pd.Timestamp("2025-10-15")


# ============================================================
# 2) SAFE METRICS & HELPERS
# ============================================================
def safe_div(n, d):
    try:
        if d is None or pd.isna(d) or d == 0:
            return np.nan
        return n / d
    except Exception:
        return np.nan

def fmt_int(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{int(round(x)):,}".replace(",", " ")

def fmt_money(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.0f} €".replace(",", " ")

def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.2f}%"

def fmt_float(x, nd=3):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{nd}f}"

def clamp_date(d):
    if d < START:
        return START
    if d > END:
        return END
    return d


# ============================================================
# 3) FILTERING LAYER (MONTH / DAY)
# ============================================================
def filter_leads(base_df, channels=None, statuses=None, day=None):
    df = base_df
    if channels:
        df = df[df["channel"].isin(channels)]
    if statuses:
        df = df[df["status"].isin(statuses)]
    if day is not None:
        df = df[df["date"] == day]
    return df

def filter_campaign(base_df, channels=None, day=None):
    df = base_df
    if channels:
        df = df[df["channel"].isin(channels)]
    if day is not None:
        df = df[df["date"] == day]
    return df


# ============================================================
# 4) KPI COMPUTATION (MONTH / DAY)
# ============================================================
def compute_campaign_totals(df):
    cost = df["cost"].sum() if len(df) else 0.0
    impr = df["impressions"].sum() if len(df) else 0
    clicks = df["clicks"].sum() if len(df) else 0
    conv = df["conversions"].sum() if len(df) else 0

    ctr = safe_div(clicks, impr)
    cvr = safe_div(conv, clicks)
    cpl = safe_div(cost, conv)

    return {
        "cost": cost,
        "impressions": impr,
        "clicks": clicks,
        "conversions": conv,
        "CTR": ctr,
        "CVR": cvr,
        "CPL": cpl,
    }

def compute_leads_totals(df):
    total_leads = len(df)
    status_counts = df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {}
    mql = status_counts.get("MQL", 0)
    sql = status_counts.get("SQL", 0)
    client = status_counts.get("Client", 0)
    lost = status_counts.get("Lost", 0)

    return {
        "leads": total_leads,
        "MQL": mql,
        "SQL": sql,
        "Client": client,
        "Lost": lost,
        "sql_rate": safe_div(sql, total_leads),
        "client_rate": safe_div(client, total_leads),
    }


# ============================================================
# 5) PLOTLY THEME (Corporate)
# ============================================================
PLOTLY_THEME = {
    "template": "plotly_white",
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font": {"color": "#0B1F3A"},
    "title": {"font": {"color": "#0B1F3A"}},
    "colorway": ["#2F6FED", "#0B1F3A", "#7FB3FF", "#94A3B8", "#1D4ED8"],
    "margin": dict(l=20, r=20, t=60, b=20),
}

def apply_plotly_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(11,31,58,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(11,31,58,0.08)")
    return fig

def empty_fig(title):
    fig = go.Figure()
    fig.update_layout(title=title, height=320)
    apply_plotly_theme(fig)
    return fig


# ============================================================
# 6) FIGURES (6 total): 3 MONTH + 3 DAY
# ============================================================
def fig_month_ctr_by_channel(df_campaign_f):
    agg = df_campaign_f.groupby("channel", as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
    )
    agg["CTR"] = agg.apply(lambda r: safe_div(r["clicks"], r["impressions"]), axis=1)
    agg = agg.sort_values("CTR", ascending=False)

    fig = px.bar(agg, x="channel", y="CTR", title="CTR par canal (Mois)")
    fig.update_yaxes(tickformat=".2%")
    fig.update_layout(height=320)
    apply_plotly_theme(fig)
    return fig

def fig_month_cpl_by_channel(df_campaign_f):
    agg = df_campaign_f.groupby("channel", as_index=False).agg(
        cost=("cost", "sum"),
        conversions=("conversions", "sum"),
    )
    agg["CPL"] = agg.apply(lambda r: safe_div(r["cost"], r["conversions"]), axis=1)
    agg = agg.sort_values("CPL", ascending=True)

    fig = px.bar(agg, x="channel", y="CPL", title="Cout par lead (CPL) par canal (Mois)")
    fig.update_layout(height=320)
    apply_plotly_theme(fig)
    return fig

def fig_month_funnel_by_channel(df_leads_f):
    tab = pd.crosstab(df_leads_f["channel"], df_leads_f["status"]).reset_index()
    long = tab.melt(id_vars=["channel"], var_name="status", value_name="count")

    fig = px.bar(
        long,
        x="channel",
        y="count",
        color="status",
        barmode="stack",
        title="Repartition des statuts par canal (Mois)"
    )
    fig.update_layout(height=340, legend_title_text="")
    apply_plotly_theme(fig)
    return fig

def fig_day_timeseries_focus(df_campaign_f, day):
    daily = df_campaign_f.groupby("date", as_index=False).agg(
        cost=("cost", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["cost"], mode="lines", name="Cost"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["clicks"], mode="lines", name="Clicks"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["conversions"], mode="lines", name="Conversions"))
    fig.add_vline(x=day, line_width=2, line_dash="dash", line_color="#2F6FED")

    fig.update_layout(
        title="Evolution journaliere (Mois) + focus Jour",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_plotly_theme(fig)
    return fig

def fig_day_spend_by_channel(df_campaign_day):
    agg = df_campaign_day.groupby("channel", as_index=False)["cost"].sum()
    agg = agg.sort_values("cost", ascending=False)

    fig = px.bar(agg, x="channel", y="cost", title="Depenses par canal (Jour)")
    fig.update_layout(height=320)
    apply_plotly_theme(fig)
    return fig

def fig_day_funnel(df_leads_day, day):
    vc = df_leads_day["status"].value_counts(dropna=False).reset_index()
    vc.columns = ["status", "count"]
    order = ["MQL", "SQL", "Client", "Lost"]
    vc["status"] = pd.Categorical(vc["status"], categories=order, ordered=True)
    vc = vc.sort_values("status")

    fig = px.bar(vc, x="status", y="count", title=f"Funnel (Jour: {day.date()})")
    fig.update_layout(height=320)
    apply_plotly_theme(fig)
    return fig


# ============================================================
# 7) UI COMPONENTS
# ============================================================
def kpi_card(title, value, subtitle="", icon="bi-speedometer2"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.I(className=f"bi {icon} me-2", style={"fontSize": "1.1rem"}),
                        html.Span(title, className="text-uppercase fw-semibold"),
                    ],
                    className="kpi-title-row",
                ),
                html.Div(value, className="kpi-value"),
                html.Div(subtitle, className="kpi-subtitle"),
            ]
        ),
        className="kpi-card h-100",
    )

def section_card(title, children):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="fw-bold section-title mb-2"),
                children,
            ]
        ),
        className="section-card h-100",
    )


# ============================================================
# 8) DASH APP (LAYOUT)
# ============================================================
external_stylesheets = [
    dbc.themes.LUX,          # conserver le theme Bootstrap
    dbc.icons.BOOTSTRAP,
]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "NovaRetail — Dashboard"

app.layout = dbc.Container(
    fluid=True,
    children=[
        # ================= HEADER / FILTER BAR =================
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    className="g-3 align-items-center",
                    children=[
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("NOVARETAIL — DASHBOARD DECISIONNEL", className="mb-1 fw-bold header-title"),
                                    html.Div("Octobre 2025", className="header-subtitle"),
                                ]
                            ),
                            md=5,
                        ),

                        dbc.Col(
                            html.Div(
                                [
                                    html.Div("Jour (octobre)", className="small header-label"),
                                    dcc.DatePickerSingle(
                                        id="day-picker",
                                        date=DEFAULT_DAY,
                                        min_date_allowed=START,
                                        max_date_allowed=END,
                                        display_format="YYYY-MM-DD",
                                    ),
                                ]
                            ),
                            md=2,
                        ),

                        dbc.Col(
                            html.Div(
                                [
                                    html.Div("Canaux", className="small header-label"),
                                    dcc.Dropdown(
                                        id="channel-filter",
                                        options=[{"label": c, "value": c} for c in CHANNELS],
                                        value=CHANNELS,
                                        multi=True,
                                        placeholder="Tous les canaux",
                                        clearable=False,
                                    ),
                                ]
                            ),
                            md=3,
                        ),

                        dbc.Col(
                            html.Div(
                                [
                                    html.Div("Statuts", className="small header-label"),
                                    dcc.Dropdown(
                                        id="status-filter",
                                        options=[{"label": s, "value": s} for s in STATUSES],
                                        value=STATUSES,
                                        multi=True,
                                        placeholder="Tous les statuts",
                                        clearable=False,
                                    ),
                                ]
                            ),
                            md=2,
                        ),
                    ],
                )
            ),
            className="sticky-topbar border-0 shadow-sm mt-3 mx-2",
        ),

        # ================= KPI ROW: MONTH (6) =================
        dbc.Row(
            className="mt-3 g-3 mx-2",
            children=[
                dbc.Col(kpi_card("Depenses (Mois)", "—", "Somme couts", "bi-currency-euro"), md=2, id="kpi-m-cost"),
                dbc.Col(kpi_card("Impressions (Mois)", "—", "Portee totale", "bi-broadcast"), md=2, id="kpi-m-impr"),
                dbc.Col(kpi_card("Clicks (Mois)", "—", "Engagement", "bi-mouse"), md=2, id="kpi-m-clicks"),
                dbc.Col(kpi_card("Conversions (Mois)", "—", "Actions cibles", "bi-check2-circle"), md=2, id="kpi-m-conv"),
                dbc.Col(kpi_card("CTR (Mois)", "—", "clicks / impressions", "bi-bar-chart-line"), md=2, id="kpi-m-ctr"),
                dbc.Col(kpi_card("CPL (Mois)", "—", "cost / conversions", "bi-calculator"), md=2, id="kpi-m-cpl"),
            ],
        ),

        # ================= KPI ROW: DAY (6) =================
        dbc.Row(
            className="mt-2 g-3 mx-2",
            children=[
                dbc.Col(kpi_card("Leads (Jour)", "—", "Nombre de leads", "bi-people"), md=2, id="kpi-d-leads"),
                dbc.Col(kpi_card("MQL (Jour)", "—", "Leads MQL", "bi-envelope"), md=2, id="kpi-d-mql"),
                dbc.Col(kpi_card("SQL (Jour)", "—", "Leads SQL", "bi-briefcase"), md=2, id="kpi-d-sql"),
                dbc.Col(kpi_card("Clients (Jour)", "—", "Clients du jour", "bi-person-check"), md=2, id="kpi-d-client"),
                dbc.Col(kpi_card("Taux SQL (Jour)", "—", "SQL / leads", "bi-graph-up"), md=2, id="kpi-d-sqlrate"),
                dbc.Col(kpi_card("Taux Client (Jour)", "—", "Client / leads", "bi-star"), md=2, id="kpi-d-clientrate"),
            ],
        ),

        # ================= MAIN SPLIT (MONTH / DAY) =================
        dbc.Row(
            className="mt-3 g-3 mx-2",
            children=[
                dbc.Col(
                    section_card(
                        "Vue MOIS (filtres appliques)",
                        dbc.Row(
                            className="g-3",
                            children=[
                                dbc.Col(dcc.Graph(id="g-month-ctr", config={"displayModeBar": False}), md=6),
                                dbc.Col(dcc.Graph(id="g-month-cpl", config={"displayModeBar": False}), md=6),
                                dbc.Col(dcc.Graph(id="g-month-funnel", config={"displayModeBar": False}), md=12),
                            ],
                        ),
                    ),
                    md=7,
                ),

                dbc.Col(
                    section_card(
                        "Vue JOUR (selon la date choisie)",
                        dbc.Row(
                            className="g-3",
                            children=[
                                dbc.Col(dcc.Graph(id="g-day-timeseries", config={"displayModeBar": False}), md=12),
                                dbc.Col(dcc.Graph(id="g-day-spend", config={"displayModeBar": False}), md=6),
                                dbc.Col(dcc.Graph(id="g-day-funnel", config={"displayModeBar": False}), md=6),
                            ],
                        ),
                    ),
                    md=5,
                ),
            ],
        ),

        # ================= FOOTER =================
        html.Div(
            "© 2026 Yasser Derdoum — Tous droits réservés",
            className="footer-bar text-center small",
        ),
    ],
)


# ============================================================
# 9) CALLBACKS
# ============================================================
@app.callback(
    # KPI Month (6)
    Output("kpi-m-cost", "children"),
    Output("kpi-m-impr", "children"),
    Output("kpi-m-clicks", "children"),
    Output("kpi-m-conv", "children"),
    Output("kpi-m-ctr", "children"),
    Output("kpi-m-cpl", "children"),
    # KPI Day (6)
    Output("kpi-d-leads", "children"),
    Output("kpi-d-mql", "children"),
    Output("kpi-d-sql", "children"),
    Output("kpi-d-client", "children"),
    Output("kpi-d-sqlrate", "children"),
    Output("kpi-d-clientrate", "children"),
    # Graphs (6)
    Output("g-month-ctr", "figure"),
    Output("g-month-cpl", "figure"),
    Output("g-month-funnel", "figure"),
    Output("g-day-timeseries", "figure"),
    Output("g-day-spend", "figure"),
    Output("g-day-funnel", "figure"),
    Input("day-picker", "date"),
    Input("channel-filter", "value"),
    Input("status-filter", "value"),
)
def update_dashboard(day_str, channels, statuses):
    day = pd.to_datetime(day_str) if day_str else DEFAULT_DAY
    day = clamp_date(day)

    channels = channels or CHANNELS
    statuses = statuses or STATUSES

    camp_m = filter_campaign(df_campaign, channels=channels, day=None)
    leads_m = filter_leads(df_leads, channels=channels, statuses=statuses, day=None)

    camp_d = filter_campaign(df_campaign, channels=channels, day=day)
    leads_d = filter_leads(df_leads, channels=channels, statuses=statuses, day=day)

    km = compute_campaign_totals(camp_m)
    kd = compute_leads_totals(leads_d)

    # KPI Month
    kpi_m_cost = kpi_card("Depenses (Mois)", fmt_money(km["cost"]), "Somme couts", "bi-currency-euro")
    kpi_m_impr = kpi_card("Impressions (Mois)", fmt_int(km["impressions"]), "Portee totale", "bi-broadcast")
    kpi_m_clicks = kpi_card("Clicks (Mois)", fmt_int(km["clicks"]), "Engagement", "bi-mouse")
    kpi_m_conv = kpi_card("Conversions (Mois)", fmt_int(km["conversions"]), "Actions cibles", "bi-check2-circle")
    kpi_m_ctr = kpi_card("CTR (Mois)", fmt_pct(km["CTR"]), "clicks / impressions", "bi-bar-chart-line")
    kpi_m_cpl = kpi_card("CPL (Mois)", fmt_float(km["CPL"], 3), "cost / conversions", "bi-calculator")

    # KPI Day
    kpi_d_leads = kpi_card("Leads (Jour)", fmt_int(kd["leads"]), "Nombre de leads", "bi-people")
    kpi_d_mql = kpi_card("MQL (Jour)", fmt_int(kd["MQL"]), "Leads MQL", "bi-envelope")
    kpi_d_sql = kpi_card("SQL (Jour)", fmt_int(kd["SQL"]), "Leads SQL", "bi-briefcase")
    kpi_d_client = kpi_card("Clients (Jour)", fmt_int(kd["Client"]), "Clients du jour", "bi-person-check")
    kpi_d_sqlrate = kpi_card("Taux SQL (Jour)", fmt_pct(kd["sql_rate"]), "SQL / leads", "bi-graph-up")
    kpi_d_clientrate = kpi_card("Taux Client (Jour)", fmt_pct(kd["client_rate"]), "Client / leads", "bi-star")

    # Figures
    fig_m_ctr = fig_month_ctr_by_channel(camp_m) if len(camp_m) else empty_fig("CTR par canal (Mois) — aucune donnee")
    fig_m_cpl = fig_month_cpl_by_channel(camp_m) if len(camp_m) else empty_fig("CPL par canal (Mois) — aucune donnee")
    fig_m_funnel = fig_month_funnel_by_channel(leads_m) if len(leads_m) else empty_fig("Funnel par canal (Mois) — aucune donnee")

    fig_d_ts = fig_day_timeseries_focus(camp_m, day) if len(camp_m) else empty_fig("Evolution journaliere — aucune donnee")
    fig_d_spend = fig_day_spend_by_channel(camp_d) if len(camp_d) else empty_fig("Depenses par canal (Jour) — aucune donnee")
    fig_d_funnel = fig_day_funnel(leads_d, day) if len(leads_d) else empty_fig(f"Funnel (Jour: {day.date()}) — aucune donnee")

    return (
        kpi_m_cost, kpi_m_impr, kpi_m_clicks, kpi_m_conv, kpi_m_ctr, kpi_m_cpl,
        kpi_d_leads, kpi_d_mql, kpi_d_sql, kpi_d_client, kpi_d_sqlrate, kpi_d_clientrate,
        fig_m_ctr, fig_m_cpl, fig_m_funnel, fig_d_ts, fig_d_spend, fig_d_funnel
    )


# ============================================================
# 10) RUN
# ============================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)


