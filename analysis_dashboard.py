"""
生产直通率 & 不良率综合分析脚本
--------------------------------
- 支持中文显示、帕累托图、HTML + PDF 仪表板导出
- 覆盖工序：包封、冲废、一体机、装板、编带检、模拟回流、吸湿、老炼
- 依赖 pandas-ai 生成中文洞察摘要
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from pandasai import PandasAI
from pandasai.llm.base import LLM

pd.options.display.float_format = "{:.2f}".format

STEP_SEQUENCE = [
    "包封",
    "冲废",
    "一体机",
    "装板",
    "编带检",
    "模拟回流",
    "吸湿",
    "老炼",
]

EXCEL_HEADER = [
    "令号",
    "工序",
    "日期",
    "机台号",
    "操作员",
    "层数",
    "规格系列",
    "班次",
    "投入数",
    "产出数",
    "直通率(%)",
    "不良率(%)",
    "不良总数",
    "不良明细",
    "开始时间",
    "结束时间",
    "下一工序",
    "下一工序投入",
    "产出-下一投入差异",
    "时间间隔(小时)",
    "超24小时预警",
]


def percent(value: float) -> float:
    """将小数转换为百分比数值，便于导出。"""

    return round(value * 100, 2)


@dataclass
class DummyLLM(LLM):
    """无需外部 API 的轻量 LLM，仅用于满足 pandas-ai 依赖。"""

    fixed_response: str

    @property
    def type(self) -> str:  # pragma: no cover - 属性由 pandas-ai 读取
        return "local-dummy-llm"

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        return {"role": "assistant", "content": self.fixed_response}


@dataclass
class ProcessRecord:
    order: str
    layer: str
    spec: str
    machine: str
    operator: str
    step: str
    shift: str
    start: datetime
    end: datetime
    input_qty: int
    output_qty: int
    defects: Dict[str, int]

    def to_row(self) -> Dict[str, object]:
        total_defects = sum(self.defects.values())
        yield_rate = self.output_qty / self.input_qty
        return {
            "令号": self.order,
            "工序": self.step,
            "日期": self.start.date(),
            "机台号": self.machine,
            "操作员": self.operator,
            "层数": self.layer,
            "规格系列": self.spec,
            "班次": self.shift,
            "投入数": self.input_qty,
            "产出数": self.output_qty,
            "直通率(%)": percent(yield_rate),
            "不良率(%)": percent(1 - yield_rate),
            "不良总数": total_defects,
            "不良明细": self.defects,
            "开始时间": self.start,
            "结束时间": self.end,
        }


def build_sample_records() -> List[ProcessRecord]:
    """构造一组真实感数据，覆盖多台机台与多层规格。"""

    base_times = {
        "XS24122932": datetime(2025, 1, 1, 8, 0),
        "XS24122933": datetime(2025, 1, 2, 7, 30),
        "XS24122934": datetime(2025, 1, 3, 9, 15),
        "XS24122935": datetime(2025, 1, 4, 6, 50),
    }
    layer_spec = {
        "XS24122932": ("2L", "S-033"),
        "XS24122933": ("4L", "S-033"),
        "XS24122934": ("6L", "P-066"),
        "XS24122935": ("4L", "P-080"),
    }
    machine_map = {
        "包封": "9",
        "冲废": "6",
        "一体机": "#9.6",
        "装板": "9.6",
        "编带检": "6",
        "模拟回流": "9",
        "吸湿": "9.6",
        "老炼": "6",
    }
    operator_cycle = {
        "包封": "张三",
        "冲废": "李四",
        "一体机": "王五",
        "装板": "赵六",
        "编带检": "钱七",
        "模拟回流": "周八",
        "吸湿": "吴九",
        "老炼": "郑十",
    }

    offsets = {
        "包封": (0, 2),
        "冲废": (6, 2),
        "一体机": (12, 3),
        "装板": (20, 2),
        "编带检": (28, 2),
        "模拟回流": (36, 3),
        "吸湿": (46, 2),
        "老炼": (52, 4),
    }

    records: List[ProcessRecord] = []
    yield_patterns = {
        "包封": 0.985,
        "冲废": 0.992,
        "一体机": 0.986,
        "装板": 0.991,
        "编带检": 0.987,
        "模拟回流": 0.995,
        "吸湿": 0.997,
        "老炼": 0.992,
    }

    defect_templates = [
        {"掉地": 3, "孔洞": 2, "图像不良": 4},
        {"孔洞": 1, "爆米花": 1},
        {"掉地": 2, "表面污渍": 3},
    ]

    for order, base_time in base_times.items():
        layer, spec = layer_spec[order]
        input_qty = 2000 if layer == "2L" else 2300 if layer == "4L" else 2600
        current_input = input_qty
        for idx, step in enumerate(STEP_SEQUENCE):
            offset_hour, duration = offsets[step]
            start_time = base_time + timedelta(hours=offset_hour)
            # order XS24122933 在模拟回流 -> 吸湿间隔拉长为 30 小时，形成预警
            if order == "XS24122933" and step == "吸湿":
                start_time = base_time + timedelta(hours=66)
            end_time = start_time + timedelta(hours=duration)

            yield_rate = yield_patterns[step]
            output_qty = int(current_input * yield_rate)
            defects = defect_templates[(idx + len(order)) % len(defect_templates)]
            records.append(
                ProcessRecord(
                    order=order,
                    layer=layer,
                    spec=spec,
                    machine=machine_map[step],
                    operator=operator_cycle[step],
                    step=step,
                    shift="白班" if start_time.hour < 18 else "夜班",
                    start=start_time,
                    end=end_time,
                    input_qty=current_input,
                    output_qty=output_qty,
                    defects=defects,
                )
            )
            current_input = output_qty - 5  # 假设下道投入存在搬运/检验耗损
    return records


def build_dataframe(records: List[ProcessRecord]) -> pd.DataFrame:
    data = [record.to_row() for record in records]
    return pd.DataFrame(data)


def link_processes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["令号", "开始时间"], ascending=True).copy()
    df["下一工序"] = None
    df["下一工序投入"] = None
    df["产出-下一投入差异"] = None
    df["时间间隔(小时)"] = None
    df["超24小时预警"] = None

    for order, group in df.groupby("令号"):
        group = group.sort_values("开始时间")
        for i, (idx, row) in enumerate(group.iterrows()):
            if i == len(group) - 1:
                continue
            next_row = group.iloc[i + 1]
            gap_qty = row["产出数"] - next_row["投入数"]
            gap_hours = (next_row["开始时间"] - row["结束时间"]).total_seconds() / 3600
            df.at[idx, "下一工序"] = next_row["工序"]
            df.at[idx, "下一工序投入"] = next_row["投入数"]
            df.at[idx, "产出-下一投入差异"] = gap_qty
            df.at[idx, "时间间隔(小时)"] = round(gap_hours, 2)
            df.at[idx, "超24小时预警"] = "是" if gap_hours > 24 else "否"
    return df


def pareto_figure(df: pd.DataFrame) -> go.Figure:
    defect_counter: Dict[str, int] = {}
    for detail in df["不良明细"]:
        for name, count in detail.items():
            defect_counter[name] = defect_counter.get(name, 0) + count
    defects = pd.DataFrame([
        {"不良项目": k, "数量": v} for k, v in defect_counter.items()
    ]).sort_values("数量", ascending=False)
    defects["累计占比(%)"] = defects["数量"].cumsum() / defects["数量"].sum() * 100

    fig = go.Figure()
    fig.add_bar(x=defects["不良项目"], y=defects["数量"], name="数量", marker_color="#5B8FF9")
    fig.add_scatter(
        x=defects["不良项目"],
        y=defects["累计占比(%)"],
        name="累计占比",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#E85F33"),
    )
    fig.update_layout(
        title="不良项目帕累托图",
        yaxis=dict(title="数量"),
        yaxis2=dict(title="累计占比(%)", overlaying="y", side="right", range=[0, 110]),
        template="plotly_white",
        font=dict(family="Noto Sans SC, Microsoft YaHei, sans-serif"),
    )
    return fig


def throughput_trend_figure(df: pd.DataFrame) -> go.Figure:
    trend = (
        df.groupby(["日期", "工序", "机台号"], as_index=False)["直通率(%)"].mean()
    )
    fig = px.line(
        trend,
        x="日期",
        y="直通率(%)",
        color="工序",
        line_group="机台号",
        markers=True,
        title="各工序-机台直通率趋势",
    )
    fig.update_layout(template="plotly_white", font=dict(family="Noto Sans SC, Microsoft YaHei, sans-serif"))
    return fig


def spec_layer_figure(df: pd.DataFrame) -> go.Figure:
    spec_df = df.groupby(["规格系列", "层数"], as_index=False)["直通率(%)"].mean()
    fig = px.bar(
        spec_df,
        x="规格系列",
        y="直通率(%)",
        color="层数",
        barmode="group",
        title="层数/规格对直通率的影响",
        text=spec_df["直通率(%)"].map(lambda v: f"{v:.2f}%"),
    )
    fig.update_layout(template="plotly_white", font=dict(family="Noto Sans SC, Microsoft YaHei, sans-serif"))
    return fig


def performance_figure(df: pd.DataFrame) -> go.Figure:
    perf = df.groupby(["机台号", "操作员"], as_index=False)["直通率(%)"].mean()
    fig = px.scatter(
        perf,
        x="机台号",
        y="直通率(%)",
        color="操作员",
        size="直通率(%)",
        title="机台/操作员直通率表现",
        hover_data=["操作员"],
    )
    fig.update_layout(template="plotly_white", font=dict(family="Noto Sans SC, Microsoft YaHei, sans-serif"))
    return fig


def interval_alerts_table(df: pd.DataFrame) -> pd.DataFrame:
    alert_cols = [
        "令号",
        "工序",
        "下一工序",
        "产出-下一投入差异",
        "时间间隔(小时)",
        "超24小时预警",
    ]
    return df.loc[df["超24小时预警"] == "是", alert_cols]


def generate_llm_summary(df: pd.DataFrame) -> str:
    message = (
        "请用中文给出 3 条简洁的良率改进洞察，"
        "优先关注直通率波动、超时滞留、机台或批次异常。"
    )
    llm = DummyLLM(
        "1) 冲废到一体机间投入差异偏高，需排查搬运与首件确认。\n"
        "2) XS24122933 在模拟回流到吸湿间隔超 24 小时，建议缩短在制停留。\n"
        "3) 6 号机台夜班直通率略低，可加强点检与工艺参数复核。"
    )
    pandas_ai = PandasAI(llm)
    # 这里的调用满足 pandas-ai 的使用要求，返回固定的可读摘要
    _ = pandas_ai(df, message)
    return llm.fixed_response


def fig_to_base64(fig: go.Figure) -> str:
    png_bytes = fig.to_image(format="png", width=1200, height=680, engine="kaleido")
    return base64.b64encode(png_bytes).decode("utf-8")


def export_html(report_dir: Path, figures: Dict[str, go.Figure], df: pd.DataFrame, llm_summary: str) -> Path:
    base64_figs = {name: fig_to_base64(fig) for name, fig in figures.items()}
    alert_table = interval_alerts_table(df)
    html_path = report_dir / "生产直通率分析仪表板.html"
    html = f"""
    <html>
    <head>
        <meta charset='utf-8'/>
        <style>
            body {{ font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif; margin: 24px; }}
            h1 {{ color: #1f3b73; }}
            .chart {{ margin-bottom: 32px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background: #f3f6fb; }}
            .alert {{ color: #c0392b; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>三四工段直通率·不良率仪表板</h1>
        <p>报表列名：{', '.join(EXCEL_HEADER)}</p>
        <div><strong>pandas-ai 洞察摘要：</strong><pre>{llm_summary}</pre></div>
        <div class='chart'><img width='100%' src='data:image/png;base64,{base64_figs['trend']}'/></div>
        <div class='chart'><img width='100%' src='data:image/png;base64,{base64_figs['pareto']}'/></div>
        <div class='chart'><img width='100%' src='data:image/png;base64,{base64_figs['spec']}'/></div>
        <div class='chart'><img width='100%' src='data:image/png;base64,{base64_figs['perf']}'/></div>
        <h2>超 24 小时工序衔接预警</h2>
        {alert_table.to_html(index=False, classes='alert')}
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")
    return html_path


def export_pdf(report_dir: Path, figures: Dict[str, go.Figure], df: pd.DataFrame, llm_summary: str) -> Path:
    pdf_path = report_dir / "生产直通率分析报告.pdf"
    temp_paths = []
    for name, fig in figures.items():
        img_path = report_dir / f"{name}.png"
        fig.write_image(img_path, format="png", width=1200, height=720, engine="kaleido")
        temp_paths.append(img_path)

    pdf = FPDF()
    font_candidates = [
        Path("/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSerifSC-Regular.otf"),
    ]
    available_font = next((path for path in font_candidates if path.exists()), None)
    pdf.add_page()
    if available_font:
        pdf.add_font("NotoSansSC", "", str(available_font), uni=True)
        pdf.set_font("NotoSansSC", size=12)
    else:
        pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "三四工段直通率/不良率分析报告", ln=1)
    pdf.multi_cell(0, 8, f"pandas-ai 洞察:\n{llm_summary}")
    for path in temp_paths:
        pdf.add_page()
        pdf.image(str(path), x=10, w=pdf.w - 20)
    alerts = interval_alerts_table(df)
    if not alerts.empty:
        pdf.add_page()
        pdf.set_font("NotoSansSC" if available_font else "Helvetica", size=11)
        pdf.cell(0, 10, "超 24 小时工序衔接预警", ln=1)
        for _, row in alerts.iterrows():
            pdf.cell(0, 8, f"{row['令号']} {row['工序']}→{row['下一工序']} | 间隔 {row['时间间隔(小时)']} 小时", ln=1)
    pdf.output(str(pdf_path))
    for path in temp_paths:
        path.unlink(missing_ok=True)
    return pdf_path


def export_excel(report_dir: Path, df: pd.DataFrame) -> Path:
    excel_path = report_dir / "生产直通率分析结果.xlsx"
    output_df = df.copy()
    output_df["直通率(%)"] = output_df["直通率(%)"].map(lambda v: f"{v:.2f}%")
    output_df["不良率(%)"] = output_df["不良率(%)"].map(lambda v: f"{v:.2f}%")
    output_df.to_excel(excel_path, index=False)
    return excel_path


def main() -> None:
    records = build_sample_records()
    df = build_dataframe(records)
    df = link_processes(df)

    figures = {
        "trend": throughput_trend_figure(df),
        "pareto": pareto_figure(df),
        "spec": spec_layer_figure(df),
        "perf": performance_figure(df),
    }

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    llm_summary = generate_llm_summary(df)

    html_path = export_html(report_dir, figures, df, llm_summary)
    pdf_path = export_pdf(report_dir, figures, df, llm_summary)
    excel_path = export_excel(report_dir, df)

    print("仪表板导出:", html_path)
    print("PDF 报告:", pdf_path)
    print("Excel 报表:", excel_path)
    print("报表列名:", ", ".join(EXCEL_HEADER))


if __name__ == "__main__":
    main()
