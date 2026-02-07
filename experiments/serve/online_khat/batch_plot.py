#!/usr/bin/env python3
"""
批量运行 plot_kstar_sweep.py 并导出结果文档

用法:
  python batch_plot.py                    # 运行所有目录
  python batch_plot.py --export-only      # 仅导出文档，不重新生成图片
  python batch_plot.py --dir bsAuto*      # 只处理匹配的目录
"""

import argparse
import subprocess
import sys
from pathlib import Path
from fnmatch import fnmatch


def find_result_dirs(base_dir: Path, pattern: str = None) -> list[Path]:
    """查找所有 bs* 和 bsAuto* 结果目录"""
    result_dirs = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and (d.name.startswith('bs') or d.name.startswith('bsAuto')):
            if not d.name.startswith('__'):
                if pattern is None or fnmatch(d.name, pattern):
                    result_dirs.append(d)
    return result_dirs


def find_scenario_dirs(result_dir: Path) -> list[Path]:
    """查找结果目录下的所有 scenario 子目录"""
    scenario_dirs = []
    for d in sorted(result_dir.iterdir()):
        if d.is_dir() and not d.name.startswith('__') and d.name != 'logs':
            scenario_dirs.append(d)
    return scenario_dirs


def run_plot(results_dir: Path, extra_args: list[str] = None, dry_run: bool = False):
    """运行 plot_kstar_sweep.py"""
    script_dir = Path(__file__).parent
    plot_script = script_dir / "plot_kstar_sweep.py"

    cmd = [
        sys.executable, str(plot_script),
        "--results-dir", str(results_dir),
        "--no-show",
        "--mode", "all"
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd[-4:])}")

    if not dry_run:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    Warning: plot failed for {results_dir}")
            if result.stderr:
                # 只打印最后几行错误
                errors = result.stderr.strip().split('\n')[-3:]
                for line in errors:
                    print(f"      {line}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='批量运行绘图脚本并导出结果')
    parser.add_argument('--export-only', action='store_true',
                        help='仅导出文档，不重新生成图片')
    parser.add_argument('--dir', type=str, default=None,
                        help='只处理匹配的目录 (支持通配符，如 bsAuto*)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅显示将要执行的命令，不实际运行')
    parser.add_argument('--n-trajectory-dirs', type=str, nargs='*',
                        default=['bsAuto_c1792_n5000/in128_out1024'],
                        help='需要额外运行 --plot-n-trajectory 的目录列表')
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if not args.export_only:
        print("=" * 60)
        print("批量生成图表")
        print("=" * 60)

        result_dirs = find_result_dirs(base_dir, args.dir)
        print(f"找到 {len(result_dirs)} 个结果目录\n")

        success_count = 0
        fail_count = 0

        for result_dir in result_dirs:
            print(f"\n[{result_dir.name}]")
            scenario_dirs = find_scenario_dirs(result_dir)

            if not scenario_dirs:
                print("  (无 scenario 目录)")
                continue

            for scenario_dir in scenario_dirs:
                rel_path = f"{result_dir.name}/{scenario_dir.name}"
                print(f"  {scenario_dir.name}:")

                # 运行基础绘图
                if run_plot(scenario_dir, dry_run=args.dry_run):
                    success_count += 1
                else:
                    fail_count += 1

                # 检查是否需要额外运行 --plot-n-trajectory
                for n_traj_dir in args.n_trajectory_dirs:
                    if rel_path == n_traj_dir or fnmatch(rel_path, n_traj_dir):
                        print(f"    + Running with --plot-n-trajectory")
                        run_plot(scenario_dir, extra_args=["--plot-n-trajectory"],
                                dry_run=args.dry_run)
                        break

        print("\n" + "=" * 60)
        print(f"绘图完成: {success_count} 成功, {fail_count} 失败")
        print("=" * 60)

    # 导出文档
    print("\n" + "=" * 60)
    print("导出 HTML 文档 (仅 kstar_sweep_results)")
    print("=" * 60)

    export_script = base_dir / "export_results_md.py"
    cmd = [sys.executable, str(export_script), "--html", "--filter", "kstar_sweep_results"]
    print(f"Running: {' '.join(cmd)}")

    if not args.dry_run:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("\n导出完成!")
        else:
            print("\n导出失败!")
            sys.exit(1)


if __name__ == "__main__":
    main()
