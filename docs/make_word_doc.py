"""生成《项目全流程记录》Word 文档 (老师同学一眼看明白版)
输出: D:\\projects\\event-driven-physics-warp\\项目全流程记录.docx
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(r"D:\projects\event-driven-physics-warp")
OUT = ROOT / "项目全流程记录.docx"

# ---------------- 步骤内容 (大白话) ----------------
STEPS = [
    dict(
        name="第1步 搬家与修环境（7个缺陷）",
        method="把项目从中文路径搬到英文路径，逐个修掉7个物理/代码缺陷：力传感器被弹簧力污染、"
               "一个任务环境打不开、推块不产生粘滑、验证时\"自己给自己当裁判\"（循环论证）、"
               "标签和输入几乎一模一样、中文路径报错、坐标错位。",
        code="env_stick_slip.py / env_peg_in_hole.py / env_stack_block.py / smoke_env.py",
        example="python smoke_env.py",
        data="无新数据",
        time="约1天",
        result="三个仿真环境全部跑通，物理现象正确",
        plain="仿真里的物理先得是真的，后面所有结论才站得住。",
    ),
    dict(
        name="第2步 设计\"标准答案\"（GT运动学化）",
        method="事件的\"标准答案\"不再用传感器信号判断，改用几何和运动学：靠没靠上=距离小于2毫米；"
               "滑没滑=物体速度过阈值；抓没抓住=物体和夹爪的相对状态。传感器信号只用来做\"自动标注\"，"
               "和标准答案分开比。",
        code="env_*.py（每个环境的 contact_state 输出）",
        example="python check_env_v2.py",
        data="无新数据",
        time="约1天",
        result="标准答案与传感器解耦",
        plain="判卷人和答题人不能是同一个，否则分数没有意义。",
    ),
    dict(
        name="第3步 数据采集",
        method="三个任务各采200条轨迹：推块（滑移最难）、插孔（接触容易）、堆块（抓/放中等）。"
               "力信号每秒500次、画面每秒100次（模拟真实机器人），摩擦系数0.7-1.3随机，"
               "每条轨迹6秒。",
        code="collect_data_step2.py",
        example="python collect_data_step2.py --task stick_slip --num-episodes 200 --output-dir data/stick_slip",
        data="600条轨迹（每条约3.5MB）",
        time="数小时",
        result="数据集 v2，完整性100%",
        plain="先有饭才能做饭：后面的训练全靠这批数据。",
    ),
    dict(
        name="第4步 自动标注 + 质量报告",
        method="用修复后的\"力+速度阈值+迟滞带\"规则自动标事件，再和标准答案比对，"
               "报出事件级F1和时间误差。",
        code="event_annotation_v2.py / annotation_report.py",
        example="python annotation_report.py --task all --data-dir data",
        data="读600条",
        time="约1小时",
        result="标注F1：插孔1.0、推块0.819（时间误差中位数1帧）、堆块0.516",
        plain="自动标注不是完美的，尤其滑移；这条差距本身就是论文要讨论的事。",
    ),
    dict(
        name="第5步 判别式模型训练（6个模型）",
        method="训练6个模型做\"看数据猜当前是哪个事件\"：主模型Event-TCN（画面+力+位置）、"
               "纯力TCN、连续力回归+阈值、纯视觉、LSTM变体、以及零训练的手调阈值基线。"
               "训练/验证/测试按轨迹分70/15/15，用验证集挑模型。",
        code="train_step2.py / dataset.py / models/",
        example="python train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip --epochs 30",
        data="每任务200条",
        time="每个模型几分钟",
        result="帧级F1：纯力TCN 0.963（推块）/0.990（堆块）",
        plain="先回答\"能不能认出来\"，这是后面一切的基础。",
    ),
    dict(
        name="第6步 实验矩阵（47组）",
        method="系统做对照：数据只用10%/50%/100%、力信号加噪声σ=0.1/0.3/0.5牛、"
               "画面与力错位±50/100/200毫秒、主模型3个随机种子复现。",
        code="run_experiments_step3.py",
        example="python run_experiments_step3.py",
        data="同上600条",
        time="约2天",
        result="47组结果：噪声下主模型只退化1.6%，手调阈值崩39%；50%数据超过100%力回归基线",
        plain="学出来的事件判断抗噪、省数据；手调规则一加噪声就垮。",
    ),
    dict(
        name="第7步 诚实复核：换\"事件级\"口径",
        method="发现帧级F1被\"长时间什么都没发生\"的帧抬高了，重新按事件时刻（±5帧容差）评："
               "只认\"滑移开始\"这类关键时刻。同时修掉两个任务的标签泄漏。",
        code="eval_events.py / event_metrics.json",
        example="python eval_events.py --task stick_slip --model force_tcn",
        data="同上",
        time="约半天",
        result="真实滑移检测F1只有0.543（帧级0.96是虚高）；连续力回归根本检不出滑移（0.0）",
        plain="换个尺子量，成绩缩水很多——但这个缩水后的数才是真的。",
    ),
]

STEPS += [
    dict(
        name="第8步 世界模型三路对照（核心）",
        method="让三个结构完全同档的模型\"看历史猜未来128帧\"：①显式事件模型（把未来猜成事件序列）、"
               "②连续力模型（把未来猜成力的数值）、③隐式潜变量模型（不告诉它事件概念，自己学）。",
        code="event_wm.py / anchored_eval.py",
        example="python event_wm.py --model event_wm --epochs 40 --anchored-train",
        data="推块200条（66训练/17测试/117跨摩擦）",
        time="每个模型约10分钟",
        result="干预实验：摩擦×1.5后，只有①预测滑移变晚，方向正确率1.00（3种子，0次反向）；②③根本产不出滑移事件",
        plain="把世界建模成\"什么时候发生什么事\"，模型才学得到因果方向。",
    ),
    dict(
        name="第9步 严谨性自查",
        method="主动找自己的毛病：训练/测试轨迹有没有重叠、模型是不是真用了摩擦这个输入、"
               "指标定义有没有漏洞、结果稳不稳定（3个种子）。发现问题就修。",
        code="scratch/audit_rigor.py（H100版在 audit/）",
        example="python audit/run_all.py --data-dir data/stick_slip",
        data="同上",
        time="约半天",
        result="查出并修正4处：干预计数把\"没变化\"也算对、一个退化视界、缺种子复现、摩擦通道敏感性未验证",
        plain="先自己拆自己的台，省得被审稿人拆。",
    ),
    dict(
        name="第10步 训练中因果正则（负结果）",
        method="尝试在训练时就把\"摩擦越大滑移越晚\"写进损失函数（两种写法），看能不能省掉事后验证。",
        code="event_wm.py --inv-reg",
        example="python event_wm.py --model event_wm --anchored-train --inv-reg",
        data="同上",
        time="每个版本约10分钟",
        result="两种写法都让模型彻底不敢预测滑移（事件F1掉到0）",
        plain="往损失里硬塞物理规律，模型学会偷懒：干脆什么都不预测。",
    ),
    dict(
        name="第11步 换任务验证（插孔/堆块）",
        method="把同一套三路对照搬到另外两个任务。插孔的\"接触\"能靠力阈值看出来，堆块补上物块高度信息。",
        code="event_wm_tasks.py",
        example="python event_wm_tasks.py --task peg_in_hole --model event_wm --epochs 40 --anchored-train",
        data="插孔/堆块各200条",
        time="每任务约1小时",
        result="插孔：事件模型干预方向率3种子全1.00；隐式模型预测最准（0.41）但因果方向只有0.43-0.60；堆块：释放事件谁都预测不了",
        plain="预测得最准的未必最懂因果；有些事件（释放）光靠现有传感器根本学不会。",
    ),
    dict(
        name="第12步 信息探针",
        method="用一个极简单的线性模型做\"事前体检\"：从输入里能猜出事件时刻多少？"
               "猜得出来=信息在输入里，模型不行是模型的问题；猜不出=输入里根本没这信息。",
        code="scratch/probe_info.py（H100版在 audit/）",
        example="python probe_info.py",
        data="同上",
        time="几分钟",
        result="可猜度：推块68%、插孔85%、堆块41%",
        plain="动手训练前先查输入里有没有货，能少走很多弯路。",
    ),
    dict(
        name="第13步 检测器+世界模型联合优化",
        method="把第5步训练好的\"事件检测器\"接到世界模型前面，把它的判断一起喂给世界模型；"
               "后来又发现检测器训练时见过测试数据（泄漏），推倒重来：只用训练集重训检测器，"
               "再让检测器和世界模型一起训练。",
        code="event_wm_det.py",
        example="python event_wm_det.py --epochs 60 --joint",
        data="同上",
        time="每模型约15分钟",
        result="无泄漏联合版：滑移事件F1 0.573（32帧提前量），超过此前所有版本",
        plain="先检测、再预测，两步互相帮忙；但训练不稳定，3个种子只有1个好。",
    ),
    dict(
        name="第14步 知识库提取与\"不忘初心\"核验",
        method="把两份文献知识库PDF转成文本，对照原始设想检查：现在做的实验是不是还是当初想验证的问题。",
        code="scratch/extract_kb.py",
        example="python scratch/extract_kb.py",
        data="2份PDF（14页+26页）",
        time="约10分钟",
        result="设计依据锁定在知识库F076（连续/事件/双路消融）和预注册问题Q6（只有连续力能否自动学出事件）",
        plain="跑得再远也要回头看看出发时的地图。",
    ),
]

DEVICES = (
    "笔记本电脑：RTX 4060（8GB显存），已跑通本项目的全部现有实验。\n"
    "H100服务器：2×80GB显存（共160GB），用于下一步的数据量扩展实验"
    "（6400条以上轨迹、大模型容量扫描、GPU并行仿真采集）。"
)

ONELINER = ("在仿真里验证一个想法：把接触操作中的物理状态表达成离散事件"
            "（没接触/粘住/滑了/放了）来训练世界模型，是否比传统的连续力数值"
            "（以及不说破事件概念的隐式方法）更省数据、更抗噪声、更能学到因果方向。")

TREE_SKIP = {".git", ".venv", "__pycache__", "archive", "scratch", "checkpoints",
             "checkpoints_wm", "checkpoints_wm_det", "checkpoints_wm_tasks",
             "data", "figures", "scripts"}

def build_tree(root: Path, prefix="", skip=TREE_SKIP, depth=0):
    lines = []
    entries = sorted([p for p in root.iterdir()
                      if p.name not in skip and not p.name.startswith(".")],
                     key=lambda p: (p.is_file(), p.name))
    for i, p in enumerate(entries):
        last = i == len(entries) - 1
        if p.is_dir():
            lines.append(f"{prefix}{'└── ' if last else '├── '}{p.name}/")
            lines += build_tree(p, prefix + ("    " if last else "│   "),
                                skip, depth + 1)
        else:
            lines.append(f"{prefix}{'└── ' if last else '├── '}{p.name}")
    return lines


def main():
    doc = Document()
    doc.add_heading("事件驱动物理世界模型 · 项目全流程记录", 0)
    doc.add_paragraph("2026-11-11 · 给老师同学的一眼看懂版")

    doc.add_heading("〇、设备信息", level=1)
    doc.add_paragraph(DEVICES)

    doc.add_heading("一、一句话概括我们在做什么", level=1)
    doc.add_paragraph(ONELINER)

    doc.add_heading("二、每一步做了什么（大白话）", level=1)
    for s in STEPS:
        doc.add_heading(s["name"], level=2)
        rows = [
            ("方法", s["method"]),
            ("代码（相对路径）", s["code"]),
            ("最小运行实例", s["example"]),
            ("处理/创造的数据量", s["data"]),
            ("处理时长", s["time"]),
            ("处理结果", s["result"]),
            ("一句话大白话：这步说明了什么", s["plain"]),
        ]
        for label, text in rows:
            p = doc.add_paragraph()
            run = p.add_run(f"{label}：")
            run.bold = True
            p.add_run(text)

    doc.add_heading("三、项目文件树", level=1)
    tree_lines = build_tree(ROOT)
    p = doc.add_paragraph()
    run = p.add_run("\n".join(tree_lines))
    run.font.name = "Consolas"
    run.font.size = Pt(8)

    doc.add_paragraph(
        "\n注：data/ 含600条轨迹(3任务×200)、checkpoints*/ 含全部训练权重与结果JSON、"
        "scratch/ 含调试脚本与日志、archive/ 为归档的旧产物。")

    OUT.parent.mkdir(exist_ok=True)
    doc.save(OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
