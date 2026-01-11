#!/usr/bin/env python3
"""
导出实验结果图片为 Markdown 或 HTML 文档
用法:
  python export_results_md.py [output_file]           # 导出 Markdown (相对路径)
  python export_results_md.py --html [output_file]    # 导出 HTML (嵌入 Base64 图片)
"""

import argparse
import base64
import mimetypes
import re
import sys
from pathlib import Path
from datetime import datetime


def find_images_in_dir(directory: Path, filter_pattern: str = None) -> list[Path]:
    """查找目录下的所有图片文件

    Args:
        directory: 要搜索的目录
        filter_pattern: 可选的过滤模式，只包含文件名包含此模式的图片
    """
    images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        images.extend(directory.glob(ext))
    # 过滤掉空文件名或隐藏文件
    images = [img for img in images if img.stem and not img.name.startswith('.')]
    # 应用过滤模式
    if filter_pattern:
        images = [img for img in images if filter_pattern in img.stem]
    return sorted(images)


def parse_scenario_name(name: str) -> tuple[int, int]:
    """解析 scenario 名称，返回 (input, output) 用于排序"""
    match = re.match(r'in(\d+)_out(\d+)', name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def sort_scenarios(dirs: list[Path]) -> list[Path]:
    """按 input/output 大小排序 scenario 目录"""
    return sorted(dirs, key=lambda d: parse_scenario_name(d.name))


def image_to_base64(img_path: Path) -> str:
    """将图片转换为 base64 data URI"""
    mime_type, _ = mimetypes.guess_type(str(img_path))
    if mime_type is None:
        mime_type = 'image/png'
    with open(img_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime_type};base64,{data}"


def parse_result_dir_name(name: str) -> tuple[int, int, int, int]:
    """
    解析结果目录名，返回排序键 (is_auto, bs_value, c_value, n_value)
    - bs128_c256_n4000 -> (0, 128, 256, 4000)
    - bsAuto_c1792_n5000 -> (1, 0, 1792, 5000)  # bsAuto 排在后面
    """
    is_auto = 1 if name.startswith('bsAuto') else 0

    # 提取 bs 值
    bs_match = re.match(r'bs(\d+)', name)
    bs_value = int(bs_match.group(1)) if bs_match else 0

    # 提取 c 值
    c_match = re.search(r'_c(\d+)', name)
    c_value = int(c_match.group(1)) if c_match else 0

    # 提取 n 值
    n_match = re.search(r'_n(\d+)', name)
    n_value = int(n_match.group(1)) if n_match else 0

    return (is_auto, bs_value, c_value, n_value)


def get_result_dirs(base_dir: Path) -> list[Path]:
    """查找所有 bs* 和 bsAuto* 目录，按 bs 值从小到大排序"""
    result_dirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and (d.name.startswith('bs') or d.name.startswith('bsAuto')):
            if not d.name.startswith('__'):
                result_dirs.append(d)
    # 按 (is_auto, bs_value, c_value, n_value) 排序
    # bs 数值目录在前，bsAuto 目录在后；同类型内按数值排序
    return sorted(result_dirs, key=lambda d: parse_result_dir_name(d.name))


def get_scenario_dirs(result_dir: Path) -> list[Path]:
    """获取排序后的 scenario 子目录"""
    scenario_dirs = [
        d for d in result_dir.iterdir()
        if d.is_dir() and not d.name.startswith('__') and not d.name == 'logs'
    ]
    in_out_dirs = [d for d in scenario_dirs if d.name.startswith('in')]
    other_dirs = [d for d in scenario_dirs if not d.name.startswith('in')]
    return sort_scenarios(in_out_dirs) + sorted(other_dirs)


def export_to_markdown(base_dir: Path, output_file: Path):
    """导出结果为 Markdown 文档 (相对路径)"""
    result_dirs = get_result_dirs(base_dir)

    lines = []
    lines.append("# 实验结果汇总")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 目录
    lines.append("## 目录")
    lines.append("")
    for result_dir in result_dirs:
        anchor = result_dir.name.replace('_', '-').lower()
        lines.append(f"- [{result_dir.name}](#{anchor})")
    lines.append("")
    lines.append("---")
    lines.append("")

    for result_dir in result_dirs:
        lines.append(f"## {result_dir.name}")
        lines.append("")

        # 顶层图片
        for img in find_images_in_dir(result_dir):
            rel_path = img.relative_to(base_dir)
            lines.append(f"![{img.name}]({rel_path})")
            lines.append("")

        # scenario 子目录
        for scenario_dir in get_scenario_dirs(result_dir):
            images = find_images_in_dir(scenario_dir)
            if images:
                lines.append(f"### {scenario_dir.name}")
                lines.append("")
                for img in images:
                    rel_path = img.relative_to(base_dir)
                    lines.append(f"**{img.stem}**")
                    lines.append("")
                    lines.append(f"![{img.name}]({rel_path})")
                    lines.append("")

        lines.append("---")
        lines.append("")

    output_file.write_text('\n'.join(lines), encoding='utf-8')
    print(f"已导出 Markdown 到: {output_file}")
    print(f"包含 {len(result_dirs)} 个实验目录")


def export_to_html(base_dir: Path, output_file: Path, filter_pattern: str = None):
    """导出结果为 HTML 文档 (嵌入 Base64 图片，完全便携)

    Args:
        base_dir: 基础目录
        output_file: 输出文件路径
        filter_pattern: 可选的图片过滤模式
    """
    result_dirs = get_result_dirs(base_dir)

    # Debug: 打印找到的目录
    print(f"找到 {len(result_dirs)} 个结果目录:")
    for d in result_dirs:
        print(f"  - {d.name}")

    html_parts = []

    # HTML 头部 (带侧边栏布局)
    html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实验结果汇总</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        /* 侧边栏 */
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 260px;
            height: 100vh;
            background: #fff;
            border-right: 1px solid #ddd;
            overflow-y: auto;
            padding: 15px;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        .sidebar h2 {
            margin: 0 0 10px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #007bff;
            color: #333;
            font-size: 1.1em;
        }
        .sidebar .timestamp {
            color: #888;
            font-size: 0.75em;
            margin-bottom: 15px;
            display: block;
        }
        .sidebar ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .sidebar li {
            margin: 2px 0;
        }
        .sidebar a {
            display: block;
            padding: 6px 10px;
            color: #333;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.85em;
            transition: background 0.2s, color 0.2s;
        }
        .sidebar a:hover {
            background: #e9ecef;
            color: #007bff;
        }
        .sidebar a.active {
            background: #007bff;
            color: #fff;
        }
        /* 主内容区 - 居中布局 */
        .main-content {
            margin-left: 260px;
            padding: 20px 30px;
            max-width: 1100px;
            margin-right: auto;
        }
        /* 让侧边栏 + 主内容整体居中 */
        @media (min-width: 1500px) {
            .sidebar {
                left: calc((100vw - 1360px) / 2);
            }
            .main-content {
                margin-left: calc((100vw - 1360px) / 2 + 260px);
            }
        }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-top: 0; }
        h2 { color: #007bff; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
        h3 { color: #555; margin-top: 30px; }
        .section {
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            scroll-margin-top: 20px;
        }
        .section h2 { margin-top: 0; }
        .image-container { margin: 15px 0; }
        .image-label { font-weight: bold; color: #333; margin-bottom: 8px; }
        .no-images { color: #999; font-style: italic; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        /* 响应式: 小屏幕隐藏侧边栏 */
        @media (max-width: 900px) {
            .sidebar { display: none; }
            .main-content { margin-left: 0; }
        }
    </style>
</head>
<body>
""")

    # 侧边栏目录
    html_parts.append(f"""
    <nav class="sidebar">
        <h2>目录</h2>
        <span class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        <ul>
""")
    for result_dir in result_dirs:
        anchor = result_dir.name
        html_parts.append(f'            <li><a href="#{anchor}">{result_dir.name}</a></li>\n')
    html_parts.append("""        </ul>
    </nav>

    <div class="main-content">
        <h1>实验结果汇总</h1>
""")

    # 图片计数
    total_images = 0

    # 每个实验目录
    for result_dir in result_dirs:
        anchor = result_dir.name  # 使用原始名称作为 anchor，不做转换

        # Debug: 打印处理的目录
        top_images = find_images_in_dir(result_dir, filter_pattern)
        scenario_dirs = get_scenario_dirs(result_dir)
        print(f"\n  [{result_dir.name}]")
        print(f"    顶层图片: {len(top_images)}")
        print(f"    子目录: {[d.name for d in scenario_dirs]}")

        html_parts.append(f"""
    <div class="section" id="{anchor}">
        <h2>{result_dir.name}</h2>
""")

        section_has_images = False

        # 顶层图片
        for img in top_images:
            print(f"    + 顶层: {img.name}")
            data_uri = image_to_base64(img)
            html_parts.append(f"""
        <div class="image-container">
            <div class="image-label">{img.stem}</div>
            <img src="{data_uri}" alt="{img.name}">
        </div>
""")
            total_images += 1
            section_has_images = True

        # scenario 子目录
        for scenario_dir in scenario_dirs:
            images = find_images_in_dir(scenario_dir, filter_pattern)
            if images:
                print(f"    + {scenario_dir.name}: {len(images)} 张图片")
                html_parts.append(f"""
        <h3>{scenario_dir.name}</h3>
""")
                for img in images:
                    data_uri = image_to_base64(img)
                    html_parts.append(f"""
        <div class="image-container">
            <div class="image-label">{img.stem}</div>
            <img src="{data_uri}" alt="{img.name}">
        </div>
""")
                    total_images += 1
                    section_has_images = True
            else:
                print(f"    - {scenario_dir.name}: 无图片")

        # 如果没有图片，显示提示
        if not section_has_images:
            html_parts.append("""
        <p class="no-images">(暂无图片)</p>
""")

        html_parts.append("""    </div>
""")

    # HTML 尾部 (关闭 main-content div，添加滚动高亮 JS)
    html_parts.append("""
    </div><!-- end main-content -->

    <script>
    // 滚动时高亮当前 section 对应的侧边栏链接
    document.addEventListener('DOMContentLoaded', function() {
        const sections = document.querySelectorAll('.section');
        const navLinks = document.querySelectorAll('.sidebar a');

        function updateActiveLink() {
            let current = '';
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                if (window.scrollY >= sectionTop - 100) {
                    current = section.getAttribute('id');
                }
            });

            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) {
                    link.classList.add('active');
                }
            });
        }

        window.addEventListener('scroll', updateActiveLink);
        updateActiveLink(); // 初始化
    });
    </script>
</body>
</html>
""")

    output_file.write_text(''.join(html_parts), encoding='utf-8')
    print(f"已导出 HTML 到: {output_file}")
    print(f"包含 {len(result_dirs)} 个实验目录, {total_images} 张图片")
    print(f"文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def export_to_html_by_scenario(base_dir: Path, output_file: Path, filter_pattern: str = None):
    """按 Scenario 分组导出 HTML，方便对比同一 scenario 在不同配置下的结果

    Args:
        base_dir: 基础目录
        output_file: 输出文件路径
        filter_pattern: 可选的图片过滤模式
    """
    result_dirs = get_result_dirs(base_dir)

    # 收集所有 scenario 及其对应的 (config_name, scenario_dir) 列表
    scenario_map: dict[str, list[tuple[str, Path]]] = {}
    for result_dir in result_dirs:
        for scenario_dir in get_scenario_dirs(result_dir):
            scenario_name = scenario_dir.name
            if scenario_name not in scenario_map:
                scenario_map[scenario_name] = []
            scenario_map[scenario_name].append((result_dir.name, scenario_dir))

    # 按 scenario 名称排序
    sorted_scenarios = sorted(scenario_map.keys(), key=lambda x: parse_scenario_name(x))

    print(f"找到 {len(sorted_scenarios)} 个 Scenario:")
    for s in sorted_scenarios:
        print(f"  - {s} ({len(scenario_map[s])} 个配置)")

    html_parts = []

    # HTML 头部
    html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实验结果对比 (按 Scenario)</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        /* 侧边栏 */
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 220px;
            height: 100vh;
            background: #fff;
            border-right: 1px solid #ddd;
            overflow-y: auto;
            padding: 15px;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        .sidebar h2 {
            margin: 0 0 10px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #28a745;
            color: #333;
            font-size: 1.1em;
        }
        .sidebar .timestamp {
            color: #888;
            font-size: 0.75em;
            margin-bottom: 15px;
            display: block;
        }
        .sidebar ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .sidebar li {
            margin: 2px 0;
        }
        .sidebar a {
            display: block;
            padding: 8px 12px;
            color: #333;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9em;
            transition: background 0.2s, color 0.2s;
        }
        .sidebar a:hover {
            background: #e9ecef;
            color: #28a745;
        }
        .sidebar a.active {
            background: #28a745;
            color: #fff;
        }
        /* 主内容区 - 居中布局 */
        .main-content {
            margin-left: 220px;
            padding: 20px 30px;
            max-width: 1300px;
            margin-right: auto;
        }
        @media (min-width: 1600px) {
            .sidebar {
                left: calc((100vw - 1520px) / 2);
            }
            .main-content {
                margin-left: calc((100vw - 1520px) / 2 + 220px);
            }
        }
        h1 { color: #333; border-bottom: 2px solid #28a745; padding-bottom: 10px; margin-top: 0; }
        h2 { color: #28a745; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
        h3 { color: #555; margin-top: 20px; margin-bottom: 10px; }
        .section {
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            scroll-margin-top: 20px;
        }
        .section h2 { margin-top: 0; }
        .config-group {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
            background: #fafafa;
        }
        .config-group h3 {
            margin-top: 0;
            color: #007bff;
            font-size: 1em;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        .image-container { margin: 10px 0; }
        .image-label { font-weight: bold; color: #333; margin-bottom: 8px; font-size: 0.9em; }
        .no-images { color: #999; font-style: italic; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        /* 响应式 */
        @media (max-width: 900px) {
            .sidebar { display: none; }
            .main-content { margin-left: 0; }
        }
    </style>
</head>
<body>
""")

    # 侧边栏目录
    html_parts.append(f"""
    <nav class="sidebar">
        <h2>Scenarios</h2>
        <span class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        <ul>
""")
    for scenario_name in sorted_scenarios:
        count = len(scenario_map[scenario_name])
        html_parts.append(f'            <li><a href="#{scenario_name}">{scenario_name} ({count})</a></li>\n')
    html_parts.append("""        </ul>
    </nav>

    <div class="main-content">
        <h1>实验结果对比 (按 Scenario 分组)</h1>
""")

    total_images = 0

    # 每个 scenario
    for scenario_name in sorted_scenarios:
        configs = scenario_map[scenario_name]
        print(f"\n  [{scenario_name}]")

        html_parts.append(f"""
    <div class="section" id="{scenario_name}">
        <h2>{scenario_name}</h2>
""")

        # 每个配置
        for config_name, scenario_dir in configs:
            images = find_images_in_dir(scenario_dir, filter_pattern)
            if images:
                print(f"    + {config_name}: {len(images)} 张图片")
                html_parts.append(f"""
        <div class="config-group">
            <h3>{config_name}</h3>
""")
                for img in images:
                    data_uri = image_to_base64(img)
                    html_parts.append(f"""
            <div class="image-container">
                <div class="image-label">{img.stem}</div>
                <img src="{data_uri}" alt="{img.name}">
            </div>
""")
                    total_images += 1
                html_parts.append("""        </div>
""")
            else:
                print(f"    - {config_name}: 无图片")

        html_parts.append("""    </div>
""")

    # HTML 尾部
    html_parts.append("""
    </div><!-- end main-content -->

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const sections = document.querySelectorAll('.section');
        const navLinks = document.querySelectorAll('.sidebar a');

        function updateActiveLink() {
            let current = '';
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                if (window.scrollY >= sectionTop - 100) {
                    current = section.getAttribute('id');
                }
            });

            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) {
                    link.classList.add('active');
                }
            });
        }

        window.addEventListener('scroll', updateActiveLink);
        updateActiveLink();
    });
    </script>
</body>
</html>
""")

    output_file.write_text(''.join(html_parts), encoding='utf-8')
    print(f"\n已导出 HTML 到: {output_file}")
    print(f"包含 {len(sorted_scenarios)} 个 Scenario, {total_images} 张图片")
    print(f"文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='导出实验结果图片为文档')
    parser.add_argument('output', nargs='?', help='输出文件路径')
    parser.add_argument('--html', action='store_true', help='导出为 HTML (嵌入 Base64 图片)')
    parser.add_argument('--by-scenario', action='store_true', help='按 Scenario 分组导出 (方便对比)')
    parser.add_argument('--md', action='store_true', help='导出为 Markdown (相对路径)')
    parser.add_argument('--filter', type=str, default=None,
                        help='只包含文件名包含此模式的图片 (例如: kstar_sweep_results)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if args.by_scenario:
        # 按 Scenario 分组导出
        output_file = Path(args.output) if args.output else base_dir / "results_by_scenario.html"
        export_to_html_by_scenario(base_dir, output_file, args.filter)
    elif args.html:
        output_file = Path(args.output) if args.output else base_dir / "experiment_results.html"
        export_to_html(base_dir, output_file, args.filter)
    else:
        output_file = Path(args.output) if args.output else base_dir / "experiment_results.md"
        export_to_markdown(base_dir, output_file)


if __name__ == "__main__":
    main()
