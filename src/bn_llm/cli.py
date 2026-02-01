"""BN-LLM 命令行接口

使用 Typer 构建，提供数据清洗、贝叶斯网络构建、LLM 推理等功能。
"""

import warnings
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

warnings.filterwarnings("ignore", category=FutureWarning)

app = typer.Typer(
    name="bn-llm",
    help="BN-LLM: 基于贝叶斯网络和大语言模型的数据清洗工具",
    no_args_is_help=True,
)

# 子命令组
fd_app = typer.Typer(help="函数依赖相关命令")
bn_app = typer.Typer(help="贝叶斯网络相关命令")
experiment_app = typer.Typer(help="实验相关命令")
data_app = typer.Typer(help="数据管理命令")
results_app = typer.Typer(help="结果管理命令")
cache_app = typer.Typer(help="缓存管理命令")

app.add_typer(fd_app, name="fd")
app.add_typer(bn_app, name="bn")
app.add_typer(experiment_app, name="experiment")
app.add_typer(data_app, name="data")
app.add_typer(results_app, name="results")
app.add_typer(cache_app, name="cache")

console = Console()


def _setup_logging(
    log_level: str = "INFO", dataset_name: Optional[str] = None, run_id: Optional[str] = None
) -> str:
    """设置日志

    Args:
        log_level: 日志级别
        dataset_name: 数据集名称
        run_id: 运行 ID，如果未提供则自动生成

    Returns:
        运行 ID（时间戳）
    """
    from .utils.logger import setup_logger

    _, run_id = setup_logger(level=log_level, dataset_name=dataset_name, run_id=run_id)
    return run_id


def _print_metrics_table(metrics: dict, title: str = "评估指标"):
    """打印指标表格"""
    table = Table(title=title)
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    metric_names = {
        "precision": "精确率 (Precision)",
        "recall": "召回率 (Recall)",
        "f1": "F1 分数",
        "detection_rate": "检测率",
        "repair_accuracy": "修复准确率",
    }

    for key, value in metrics.items():
        name = metric_names.get(key, key)
        if isinstance(value, float):
            table.add_row(name, f"{value:.4f}")
        else:
            table.add_row(name, str(value))

    console.print(table)


# ==================== 主命令 ====================


@app.command()
def clean(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    variant: Optional[str] = typer.Option(None, "--variant", help="数据集变体"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM 模型"),
    method: str = typer.Option("HYBRID", "--method", help="BN 构建方法"),
    limit: int = typer.Option(-1, "--limit", "-l", help="推理数量限制，-1 表示无限制"),
    max_budget: int = typer.Option(
        -1, "--max-budget", "-b", help="LLM 修复最大预算数量，-1 表示使用比例计算"
    ),
    budget_ratio: float = typer.Option(0.01, "--budget-ratio", help="LLM 修复预算比例（0-1）"),
    no_cache: bool = typer.Option(False, "--no-cache", help="不使用缓存"),
    log_level: str = typer.Option("INFO", "--log-level", help="日志级别"),
):
    """执行完整数据清洗流程"""
    # 设置日志并获取运行 ID
    run_id = _setup_logging(log_level, dataset)

    from .config import create_config
    from .pipeline import CleaningPipeline

    console.print("[bold green]开始数据清洗流程[/bold green]")
    console.print(f"  数据集: {dataset}" + (f" ({variant})" if variant else ""))
    console.print(f"  模型: {model}")
    console.print(f"  方法: {method}")
    console.print(f"  运行 ID: {run_id}")
    if max_budget > 0:
        console.print(f"  LLM 修复预算: {max_budget}")
    else:
        console.print(f"  LLM 修复预算比例: {budget_ratio:.1%}")

    try:
        # 创建配置
        cfg = create_config(
            dataset_name=dataset,
            variant=variant,
            model_name=model,
            build_method=method,
            infer_limit=limit,
            llm_repair_max_budget=max_budget,
            llm_repair_budget_ratio=budget_ratio,
        )
        # 设置运行 ID
        cfg.output.run_id = run_id

        # 运行清洗流程
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("清洗中...", total=None)

            pipeline = CleaningPipeline(cfg)
            result = pipeline.run(use_cache=not no_cache)

            progress.update(task, description="完成")

        # 显示结果
        console.print(f"\n[bold]清洗完成[/bold]，耗时: {result.duration_seconds:.2f}s")

        if result.evaluation:
            _print_metrics_table(
                {
                    "precision": result.evaluation.metrics.precision,
                    "recall": result.evaluation.metrics.recall,
                    "f1": result.evaluation.metrics.f1,
                }
            )

            # 显示统计
            stats = result.evaluation.statistics
            console.print(
                f"\n[dim]统计: 总错误={stats.get('total_errors', 0)}, "
                f"正确修复={stats.get('correct_repairs', 0)}, "
                f"错误修复={stats.get('wrong_repairs', 0)}, "
                f"漏检={stats.get('missed_errors', 0)}[/dim]"
            )

    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def infer(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    variant: Optional[str] = typer.Option(None, "--variant", help="数据集变体"),
    limit: int = typer.Option(-1, "--limit", "-l", help="推理数量限制"),
    no_cache: bool = typer.Option(False, "--no-cache", help="不使用缓存"),
):
    """执行 BN 推理（使用已缓存的 BN 结构）"""
    run_id = _setup_logging(dataset_name=dataset)

    from .config import create_config
    from .pipeline import CleaningPipeline

    console.print("[bold]执行 BN 推理[/bold]")
    console.print(f"  数据集: {dataset}")
    console.print(f"  运行 ID: {run_id}")

    try:
        cfg = create_config(
            dataset_name=dataset,
            variant=variant,
            infer_limit=limit,
        )
        cfg.output.run_id = run_id

        pipeline = CleaningPipeline(cfg)
        result = pipeline.run_step_infer()

        console.print("[green]推理完成[/green]")
        console.print(f"  修复单元格数: {len(result.repairs)}")

    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def repair(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    variant: Optional[str] = typer.Option(None, "--variant", help="数据集变体"),
):
    """执行 LLM 修复"""
    _setup_logging(dataset_name=dataset)

    console.print("[bold]执行 LLM 修复[/bold]")
    console.print(f"  数据集: {dataset}")

    # TODO: 实现 LLM 修复独立命令
    console.print("[yellow]LLM 修复作为清洗流程的一部分，请使用 clean 命令[/yellow]")


@app.command()
def evaluate(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    variant: Optional[str] = typer.Option(None, "--variant", help="数据集变体"),
    repaired: Optional[Path] = typer.Option(None, "--repaired", help="修复后数据文件路径"),
):
    """评估清洗结果"""
    _setup_logging(dataset_name=dataset)

    import pandas as pd

    from .config import create_config
    from .core.evaluator import DataCleaningEvaluator
    from .utils.data_loader import DataLoader

    console.print("[bold]评估清洗结果[/bold]")
    console.print(f"  数据集: {dataset}")

    try:
        cfg = create_config(dataset_name=dataset, variant=variant)
        loader = DataLoader(
            data_root=cfg.data_root,
            null_marker=cfg.experiment.null_marker,
        )

        # 加载数据
        discrete_cols = loader.load_discrete_cols(cfg.dataset)
        dirty_data, clean_data = loader.load_both(cfg.dataset, discrete_cols)

        # 加载修复后数据
        if repaired:
            repaired_data = pd.read_csv(repaired)
        else:
            # 尝试从结果目录加载（查找最近的运行）
            results_base = cfg.output.results_dir / cfg.dataset.full_name
            repaired_path = None
            if results_base.exists():
                # 查找最近的运行目录中的 repaired.csv
                run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()], reverse=True)
                for run_dir in run_dirs:
                    candidate = run_dir / "repaired.csv"
                    if candidate.exists():
                        repaired_path = candidate
                        console.print(f"[dim]使用结果文件: {repaired_path}[/dim]")
                        break
            if repaired_path and repaired_path.exists():
                repaired_data = pd.read_csv(repaired_path)
            else:
                console.print(
                    "[red]未找到修复后数据，请指定 --repaired 参数，"
                    "如: outputs/results/hospital/20260118_182308/repaired.csv[/red]"
                )
                raise typer.Exit(1)

        # 评估
        evaluator = DataCleaningEvaluator()
        result = evaluator.evaluate(dirty_data, repaired_data, clean_data)

        _print_metrics_table(
            {
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1": result.metrics.f1,
                "detection_rate": result.metrics.detection_rate,
                "repair_accuracy": result.metrics.repair_accuracy,
            }
        )

    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        raise typer.Exit(1)


# ==================== 函数依赖命令 ====================


@fd_app.command("generate")
def fd_generate(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    method: str = typer.Option("DFD", "--method", help="FD 检测方法 (FD/DFD/FDS)"),
    model: str = typer.Option("gpt-4o-mini-2024-07-18", "--model", "-m", help="LLM 模型"),
    no_cache: bool = typer.Option(False, "--no-cache", help="不使用缓存"),
):
    """生成函数依赖"""
    run_id = _setup_logging(dataset_name=dataset)

    from .config import create_config
    from .pipeline import CleaningPipeline

    console.print("[bold]生成函数依赖[/bold]")
    console.print(f"  数据集: {dataset}")
    console.print(f"  方法: {method}")

    try:
        cfg = create_config(dataset_name=dataset, model_name=model)
        cfg.output.run_id = run_id
        pipeline = CleaningPipeline(cfg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("生成 FD 矩阵...", total=None)
            fd_matrix = pipeline.run_step_fd(use_cache=not no_cache)
            progress.update(task, description="完成")

        console.print("[green]FD 矩阵生成完成[/green]")
        console.print(f"  维度: {fd_matrix.shape[0]} x {fd_matrix.shape[1]}")

    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        raise typer.Exit(1)


# ==================== 贝叶斯网络命令 ====================


@bn_app.command("build")
def bn_build(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    variant: Optional[str] = typer.Option(None, "--variant", help="数据集变体"),
    method: str = typer.Option("HYBRID", "--method", help="构建方法"),
    no_cache: bool = typer.Option(False, "--no-cache", help="不使用缓存"),
):
    """构建贝叶斯网络"""
    run_id = _setup_logging(dataset_name=dataset)

    from .config import create_config
    from .pipeline import CleaningPipeline

    console.print("[bold]构建贝叶斯网络[/bold]")
    console.print(f"  数据集: {dataset}")
    console.print(f"  方法: {method}")
    console.print(f"  运行 ID: {run_id}")

    try:
        cfg = create_config(
            dataset_name=dataset,
            variant=variant,
            build_method=method,
        )
        cfg.output.run_id = run_id
        pipeline = CleaningPipeline(cfg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("构建 BN 结构...", total=None)
            bn_structure = pipeline.run_step_bn(use_cache=not no_cache)
            progress.update(task, description="完成")

        console.print("[green]BN 结构构建完成[/green]")
        console.print(f"  节点数: {len(bn_structure.get('nodes', []))}")
        console.print(f"  一致性组数: {len(bn_structure.get('consistencies', []))}")
        console.print(f"  初始边数: {len(bn_structure.get('initial_edges', []))}")

    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        raise typer.Exit(1)


# ==================== 实验命令 ====================


@experiment_app.command("ablation")
def experiment_ablation(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    variant: Optional[str] = typer.Option(None, "--variant", help="数据集变体"),
    params: Optional[List[str]] = typer.Option(None, "--params", help="消融参数列表"),
    no_cache: bool = typer.Option(False, "--no-cache", help="不使用缓存"),
):
    """运行消融实验"""
    run_id = _setup_logging(dataset_name=dataset)

    from .config import create_config
    from .pipeline import AblationExperiment

    console.print("[bold]运行消融实验[/bold]")
    console.print(f"  数据集: {dataset}")
    if params:
        console.print(f"  参数: {params}")

    try:
        cfg = create_config(dataset_name=dataset, variant=variant)
        cfg.output.run_id = run_id
        experiment = AblationExperiment(cfg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("运行消融实验...", total=None)
            result = experiment.run(params=params, use_cache=not no_cache)
            progress.update(task, description="完成")

        # 显示结果
        console.print("\n[bold]消融实验结果[/bold]")
        console.print(f"基线 F1: {result.baseline_metrics.get('f1', 0):.4f}")

        table = Table(title="消融影响")
        table.add_column("参数", style="cyan")
        table.add_column("禁用后 F1", style="yellow")
        table.add_column("影响", style="red")

        for r in result.results:
            impact = r.impact.get("f1", 0)
            impact_str = f"{impact:+.4f}"
            table.add_row(
                r.param_name,
                f"{r.disabled_metrics.get('f1', 0):.4f}",
                impact_str,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        raise typer.Exit(1)


@experiment_app.command("param-search")
def experiment_param_search(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    variant: Optional[str] = typer.Option(None, "--variant", help="数据集变体"),
    alpha: Optional[str] = typer.Option(None, "--alpha", help="Alpha 参数值（逗号分隔）"),
    beta: Optional[str] = typer.Option(None, "--beta", help="Beta 参数值（逗号分隔）"),
    no_cache: bool = typer.Option(False, "--no-cache", help="不使用缓存"),
):
    """运行参数搜索"""
    run_id = _setup_logging(dataset_name=dataset)

    from .config import create_config
    from .pipeline import ParamSearchExperiment

    console.print("[bold]运行参数搜索[/bold]")
    console.print(f"  数据集: {dataset}")

    # 解析参数值
    alpha_values = [float(x) for x in alpha.split(",")] if alpha else None
    beta_values = [float(x) for x in beta.split(",")] if beta else None

    try:
        cfg = create_config(dataset_name=dataset, variant=variant)
        cfg.output.run_id = run_id
        experiment = ParamSearchExperiment(cfg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("参数搜索中...", total=None)
            result = experiment.grid_search(
                alpha_values=alpha_values,
                beta_values=beta_values,
                use_cache=not no_cache,
            )
            progress.update(task, description="完成")

        # 显示结果
        console.print("\n[bold green]最优参数[/bold green]")
        for k, v in result.best_params.items():
            console.print(f"  {k}: {v}")
        console.print(f"  F1: {result.best_f1:.4f}")

        # 显示所有结果
        if len(result.all_results) > 1:
            table = Table(title="所有结果")
            table.add_column("Alpha")
            table.add_column("Beta")
            table.add_column("F1", style="green")

            for r in sorted(result.all_results, key=lambda x: x.get("f1", 0), reverse=True):
                table.add_row(
                    str(r.get("alpha", "-")),
                    str(r.get("beta", "-")),
                    f"{r.get('f1', 0):.4f}",
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        raise typer.Exit(1)


@experiment_app.command("error-ratio")
def experiment_error_ratio(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    ratios: str = typer.Option("10,20,30,40,50", "--ratios", help="错误率列表（逗号分隔）"),
    no_cache: bool = typer.Option(False, "--no-cache", help="不使用缓存"),
):
    """运行错误率实验"""
    run_id = _setup_logging(dataset_name=dataset)

    from .config import create_config
    from .pipeline import CleaningPipeline

    console.print("[bold]运行错误率实验[/bold]")
    console.print(f"  数据集: {dataset}")
    console.print(f"  错误率: {ratios}")

    ratio_list = [int(x) for x in ratios.split(",")]
    results = []

    try:
        for ratio in ratio_list:
            variant = f"error_{ratio}"
            console.print(f"\n[cyan]处理变体: {variant}[/cyan]")

            cfg = create_config(dataset_name=dataset, variant=variant)
            cfg.output.run_id = run_id
            pipeline = CleaningPipeline(cfg)
            result = pipeline.run(use_cache=not no_cache)

            if result.evaluation:
                results.append(
                    {
                        "ratio": ratio,
                        "precision": result.evaluation.metrics.precision,
                        "recall": result.evaluation.metrics.recall,
                        "f1": result.evaluation.metrics.f1,
                    }
                )

        # 显示结果
        table = Table(title="错误率实验结果")
        table.add_column("错误率", style="cyan")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1", style="green")

        for r in results:
            table.add_row(
                f"{r['ratio']}%",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]错误: {e}[/bold red]")
        raise typer.Exit(1)


# ==================== 数据管理命令 ====================


@data_app.command("verify")
def data_verify(
    dataset: str = typer.Option(..., "-d", "--dataset", help="数据集名称"),
    variant: Optional[str] = typer.Option(None, "--variant", help="数据集变体"),
):
    """验证数据集完整性"""
    from .config import DatasetConfig
    from .utils.data_loader import DataLoader

    config = DatasetConfig(name=dataset, variant=variant)
    loader = DataLoader()

    result = loader.verify_dataset(config)

    if result["valid"]:
        console.print(f"[bold green]✓ 数据集 {config.full_name} 验证通过[/bold green]")
        if "row_count_dirty" in result:
            console.print(f"  行数（脏数据）: {result['row_count_dirty']}")
            console.print(f"  行数（干净数据）: {result['row_count_clean']}")
            console.print(f"  列数: {result['column_count']}")
    else:
        console.print(f"[bold red]✗ 数据集 {config.full_name} 验证失败[/bold red]")
        for error in result["errors"]:
            console.print(f"  - {error}")


@data_app.command("list")
def data_list():
    """列出所有数据集"""
    from .utils.data_loader import DataLoader

    loader = DataLoader()
    datasets = loader.list_datasets()

    if not datasets:
        console.print("[yellow]未找到数据集[/yellow]")
        return

    table = Table(title="可用数据集")
    table.add_column("数据集", style="cyan")
    table.add_column("变体", style="green")

    for ds in datasets:
        variants = loader.list_variants(ds)
        variants_str = ", ".join(variants) if variants else "-"
        table.add_row(ds, variants_str)

    console.print(table)


@data_app.command("migrate")
def data_migrate(
    source: Path = typer.Option(..., "--source", help="原数据目录"),
    target: Path = typer.Option(..., "--target", help="目标数据目录"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只显示计划，不实际执行"),
    datasets: Optional[List[str]] = typer.Option(None, "--datasets", "-d", help="指定数据集"),
):
    """迁移旧数据集格式到新格式"""
    import subprocess
    import sys

    console.print("[bold]迁移数据集[/bold]")
    console.print(f"  源目录: {source}")
    console.print(f"  目标目录: {target}")

    if dry_run:
        console.print("[yellow]（干运行模式，不实际执行）[/yellow]")

    # 构建命令
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "migrate_datasets.py"

    if not script_path.exists():
        console.print(f"[red]错误: 迁移脚本不存在: {script_path}[/red]")
        raise typer.Exit(1)

    cmd = [sys.executable, str(script_path), "--source", str(source), "--target", str(target)]

    if dry_run:
        cmd.append("--dry-run")

    if datasets:
        cmd.extend(["--datasets"] + list(datasets))

    try:
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise typer.Exit(result.returncode)
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(1)


# ==================== 结果管理命令 ====================


@results_app.command("show")
def results_show(
    experiment_id: Optional[str] = typer.Option(None, "--id", help="实验 ID"),
):
    """显示实验结果"""
    from .utils.results import ResultsManager

    manager = ResultsManager(Path("outputs/results"))

    if experiment_id:
        exp_dir = manager.results_dir / experiment_id
    else:
        exp_dir = manager.get_latest_experiment()

    if not exp_dir or not exp_dir.exists():
        console.print("[yellow]未找到实验结果[/yellow]")
        return

    summary = manager.summarize(exp_dir)

    console.print(f"[bold]实验: {summary['experiment_id']}[/bold]")
    console.print(f"数据集: {', '.join(summary['datasets'])}")

    if summary["aggregate"]:
        console.print("\n[bold]聚合指标:[/bold]")
        console.print(f"  平均 F1: {summary['aggregate'].get('mean_f1', 0):.4f}")
        console.print(f"  标准差: {summary['aggregate'].get('std_f1', 0):.4f}")

    if summary["results"]:
        table = Table(title="各数据集结果")
        table.add_column("数据集", style="cyan")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1", style="green")

        for r in summary["results"]:
            table.add_row(
                r.get("dataset", "-"),
                f"{r.get('precision', 0):.4f}",
                f"{r.get('recall', 0):.4f}",
                f"{r.get('f1', 0):.4f}",
            )

        console.print(table)


@results_app.command("list")
def results_list(
    limit: int = typer.Option(10, "--limit", "-n", help="显示数量"),
):
    """列出实验"""
    from .utils.results import ResultsManager

    manager = ResultsManager(Path("outputs/results"))
    experiments = manager.list_experiments(limit=limit)

    if not experiments:
        console.print("[yellow]未找到实验[/yellow]")
        return

    table = Table(title="实验列表")
    table.add_column("ID", style="cyan")
    table.add_column("数据集")
    table.add_column("平均 F1", style="green")

    for exp in experiments:
        f1 = exp.get("mean_f1")
        f1_str = f"{f1:.4f}" if f1 else "-"
        datasets_str = ", ".join(exp["datasets"][:3])
        if len(exp["datasets"]) > 3:
            datasets_str += "..."
        table.add_row(exp["id"], datasets_str, f1_str)

    console.print(table)


@results_app.command("export")
def results_export(
    format: str = typer.Option("csv", "--format", "-f", help="导出格式 (csv/json)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="输出文件路径"),
    experiment_id: Optional[str] = typer.Option(None, "--id", help="实验 ID"),
):
    """导出实验结果"""
    from .utils.results import ResultsManager

    manager = ResultsManager(Path("outputs/results"))

    if experiment_id:
        exp_dir = manager.results_dir / experiment_id
    else:
        exp_dir = manager.get_latest_experiment()

    if not exp_dir or not exp_dir.exists():
        console.print("[yellow]未找到实验结果[/yellow]")
        return

    try:
        output_path = manager.export(exp_dir, format=format, output_path=output)
        console.print(f"[green]已导出到: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]导出失败: {e}[/red]")


# ==================== 缓存管理命令 ====================


@cache_app.command("list")
def cache_list(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="缓存类别"),
):
    """列出缓存"""
    from .utils.cache import CacheManager, format_size

    cache_manager = CacheManager(Path("outputs/cache"))
    cache_info = cache_manager.list_cache(category)

    if not cache_info:
        console.print("[yellow]缓存为空[/yellow]")
        return

    table = Table(title="缓存列表")
    table.add_column("类别", style="cyan")
    table.add_column("文件数", style="green")
    table.add_column("大小", style="yellow")

    for cat, files in cache_info.items():
        size = cache_manager.get_cache_size(cat)
        table.add_row(cat, str(len(files)), format_size(size))

    console.print(table)

    total_size = cache_manager.get_cache_size()
    console.print(f"\n总大小: {format_size(total_size)}")


@cache_app.command("clear")
def cache_clear(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="要清除的类别"),
    force: bool = typer.Option(False, "--force", "-f", help="强制清除，不询问确认"),
):
    """清除缓存"""
    from .utils.cache import CacheManager

    if not force:
        if category:
            confirm = typer.confirm(f"确定要清除 {category} 类别的缓存吗？")
        else:
            confirm = typer.confirm("确定要清除所有缓存吗？")

        if not confirm:
            console.print("[yellow]操作已取消[/yellow]")
            return

    cache_manager = CacheManager(Path("outputs/cache"))
    count = cache_manager.clear(category)

    console.print(f"[green]已清除 {count} 个缓存文件[/green]")


@app.command()
def version():
    """显示版本信息"""
    console.print("[bold]BN-LLM[/bold] v0.2.0")
    console.print("基于贝叶斯网络和大语言模型的数据清洗工具")


def main():
    """CLI 入口点"""
    app()


if __name__ == "__main__":
    main()
