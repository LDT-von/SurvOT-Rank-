#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V45 实验结果可视化分析工具

功能：
1. 自动收集V45相关实验结果
2. 生成对比表格和图表
3. 可视化改进趋势
4. 导出分析报告
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 配置matplotlib中文字体（如果可用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class V45Analysis:
    """V45实验结果分析器"""
    
    def __init__(self, result_dir: str = "/data1/sweep_results_30ep"):
        """
        初始化分析器
        
        Args:
            result_dir: 结果目录路径
        """
        self.result_dir = Path(result_dir)
        self.v45_results = {}
        self.comparison_data = []
        
    def scan_v45_experiments(self) -> Dict[str, Dict]:
        """
        扫描所有V45相关实验结果
        
        Returns:
            字典格式的实验结果
        """
        experiments = {}
        
        # 定义V45相关实验模式
        v45_patterns = [
            "otehv2_rankevent",  # V45
            "v45_",              # V45变体
            "otehv2_eventloss",  # V46
            "otehv2_rankcox",    # V47
            "otehv2_globalres",  # V48
            "otehv2_epsanneal",  # V49
            "v50_v45ipcw",       # V50
        ]
        
        # 扫描结果目录
        for method_dir in self.result_dir.glob("*"):
            if not method_dir.is_dir():
                continue
                
            method_name = method_dir.name
            if any(pattern in method_name for pattern in v45_patterns):
                # 查找summary.csv
                summary_file = method_dir / "summary.csv"
                if summary_file.exists():
                    try:
                        df = pd.read_csv(summary_file)
                        if not df.empty and 'val_cindex' in df.columns:
                            # 计算平均指标
                            metrics = {
                                'val_cindex': float(df['val_cindex'].mean()),
                                'std': float(df['val_cindex'].std()),
                                'fold_count': len(df),
                                'method': method_name,
                                'path': str(method_dir)
                            }
                            
                            # 尝试获取其他指标
                            for col in ['ibs', 'iauc', 'loss']:
                                if col in df.columns:
                                    metrics[col] = float(df[col].mean())
                                    
                            experiments[method_name] = metrics
                            print(f"找到实验: {method_name}, C-index: {metrics['val_cindex']:.4f}")
                    except Exception as e:
                        print(f"读取 {summary_file} 失败: {e}")
                        
        return experiments
    
    def load_baseline_results(self) -> Dict[str, float]:
        """
        加载基线结果用于对比
        
        Returns:
            基线方法的结果字典
        """
        baselines = {
            'baseline': 0.7014,
            'v9_otehv2_strongot': 0.7078,
            'ot_v2': 0.7187,
            'ot_v3': 0.7282,
            'surgfix': 0.7094,
        }
        return baselines
    
    def create_comparison_table(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        创建对比表格
        
        Args:
            save_path: 保存路径（可选）
            
        Returns:
            对比表格DataFrame
        """
        # 扫描V45实验
        self.v45_results = self.scan_v45_experiments()
        
        # 加载基线
        baselines = self.load_baseline_results()
        
        # 创建对比数据
        comparison_rows = []
        
        # 添加基线
        for method, cindex in baselines.items():
            comparison_rows.append({
                'Method': method,
                'Type': 'Baseline',
                'C-index': cindex,
                'Delta_vs_V45': cindex - 0.7105 if method != 'v45' else 0.0,
                'Delta_vs_Baseline': cindex - 0.7014,
                'Note': '参考基准'
            })
        
        # 添加V45及变体
        v45_base = 0.7105  # V45基准值
        for method, metrics in self.v45_results.items():
            cindex = metrics['val_cindex']
            method_type = 'V45_Variant'
            
            # 确定实验类型
            if 'rankevent' in method and not any(x in method for x in ['eventloss', 'rankcox', 'globalres', 'epsanneal']):
                method_type = 'V45_Original'
            elif 'eventloss' in method:
                method_type = 'V46_EventLoss'
            elif 'rankcox' in method:
                method_type = 'V47_RankOnly'
            elif 'globalres' in method:
                method_type = 'V48_GlobalOnly'
            elif 'epsanneal' in method:
                method_type = 'V49_EpsAnneal'
            elif 'v45ipcw' in method:
                method_type = 'V50_IPCW'
            elif 'seed' in method:
                method_type = 'V45_MultiSeed'
                
            comparison_rows.append({
                'Method': method,
                'Type': method_type,
                'C-index': cindex,
                'Delta_vs_V45': cindex - v45_base,
                'Delta_vs_Baseline': cindex - 0.7014,
                'Std': metrics.get('std', np.nan),
                'Note': self._get_experiment_note(method, cindex, v45_base)
            })
        
        # 创建DataFrame
        df = pd.DataFrame(comparison_rows)
        df = df.sort_values('C-index', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        # 保存到文件
        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"对比表格已保存到: {save_path}")
            
            # 同时保存Markdown格式
            md_path = save_path.replace('.csv', '.md')
            self._save_markdown_table(df, md_path)
        
        self.comparison_data = df
        return df
    
    def _get_experiment_note(self, method: str, cindex: float, v45_base: float) -> str:
        """生成实验说明"""
        delta = cindex - v45_base
        
        if method == 'otehv2_rankevent':
            return "V45原始版本（SOTA）"
        elif delta > 0.001:
            return f"✅ 优于V45 (+{delta:.4f})"
        elif delta > -0.001:
            return f"≈ 与V45相当"
        elif delta > -0.005:
            return f"稍逊于V45 ({delta:.4f})"
        else:
            return f"❌ 显著差于V45 ({delta:.4f})"
    
    def _save_markdown_table(self, df: pd.DataFrame, save_path: str):
        """保存Markdown格式表格"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# V45实验结果对比\n\n")
            f.write("| 排名 | 方法 | 类型 | C-index | Δ vs V45 | Δ vs Baseline | 标准差 | 说明 |\n")
            f.write("|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|\n")
            
            for _, row in df.iterrows():
                cindex = f"{row['C-index']:.4f}" if not pd.isna(row['C-index']) else "N/A"
                delta_v45 = f"{row['Delta_vs_V45']:+.4f}" if not pd.isna(row['Delta_vs_V45']) else "N/A"
                delta_base = f"{row['Delta_vs_Baseline']:+.4f}" if not pd.isna(row['Delta_vs_Baseline']) else "N/A"
                std = f"{row['Std']:.4f}" if 'Std' in row and not pd.isna(row['Std']) else "N/A"
                
                f.write(f"| {int(row['Rank'])} | {row['Method']} | {row['Type']} | {cindex} | {delta_v45} | {delta_base} | {std} | {row['Note']} |\n")
    
    def plot_comparison_chart(self, save_path: Optional[str] = None):
        """
        绘制对比图表
        
        Args:
            save_path: 保存路径（可选）
        """
        if self.comparison_data.empty:
            self.create_comparison_table()
            
        df = self.comparison_data.copy()
        
        # 设置图表样式
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        
        # 创建分组
        df['Group'] = df['Type'].apply(lambda x: 'Baseline' if 'Baseline' in str(x) else 'V45_Family')
        
        # 创建颜色映射
        colors = {'Baseline': 'gray', 'V45_Family': 'steelblue'}
        group_colors = [colors[g] for g in df['Group']]
        
        # 绘制条形图
        bars = plt.barh(range(len(df)), df['C-index'], color=group_colors, edgecolor='black')
        
        # 添加数值标签
        for i, (bar, cindex) in enumerate(zip(bars, df['C-index'])):
            plt.text(cindex + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{cindex:.4f}', va='center', fontsize=9)
        
        # 添加V45基准线
        plt.axvline(x=0.7105, color='red', linestyle='--', alpha=0.7, label='V45基准 (0.7105)')
        plt.axvline(x=0.7014, color='green', linestyle='--', alpha=0.7, label='Baseline (0.7014)')
        
        # 设置图表属性
        plt.yticks(range(len(df)), df['Method'], fontsize=10)
        plt.xlabel('C-index', fontsize=12)
        plt.title('V45家族与基线方法对比', fontsize=14, fontweight='bold')
        plt.xlim(0.65, 0.75)
        plt.legend()
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_improvement_trend(self, save_path: Optional[str] = None):
        """
        绘制改进趋势图
        
        Args:
            save_path: 保存路径（可选）
        """
        # V45家族消融实验结果
        ablation_data = {
            'Method': ['V45全开', 'V46仅NLL', 'V47仅Ranking', 'V48仅Global', 'V49仅ε-Anneal'],
            'C-index': [0.7105, 0.6977, 0.7050, 0.6947, 0.7026],
            'Delta_vs_V45': [0.0000, -0.0128, -0.0055, -0.0158, -0.0079]
        }
        
        df_ablation = pd.DataFrame(ablation_data)
        
        plt.figure(figsize=(12, 6))
        
        # 子图1: C-index对比
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(df_ablation['Method'], df_ablation['C-index'], color='lightblue', edgecolor='black')
        plt.axhline(y=0.7105, color='red', linestyle='--', alpha=0.7, label='V45基准')
        plt.axhline(y=0.7014, color='green', linestyle='--', alpha=0.7, label='Baseline')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('C-index')
        plt.title('V45消融实验C-index对比')
        plt.legend()
        
        # 添加数值标签
        for bar, cindex in zip(bars1, df_ablation['C-index']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{cindex:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 子图2: 相对于V45的改进
        plt.subplot(1, 2, 2)
        colors = ['green' if x >= 0 else 'red' for x in df_ablation['Delta_vs_V45']]
        bars2 = plt.bar(df_ablation['Method'], df_ablation['Delta_vs_V45'], color=colors, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Δ vs V45')
        plt.title('相对于V45的改进幅度')
        
        # 添加数值标签
        for bar, delta in zip(bars2, df_ablation['Delta_vs_V45']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.0003 if delta >= 0 else -0.0005),
                    f'{delta:+.4f}', ha='center', va='bottom' if delta >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"趋势图已保存到: {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir: str = "./reports"):
        """
        生成完整分析报告
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("生成V45分析报告中...")
        
        # 1. 扫描实验结果
        experiments = self.scan_v45_experiments()
        print(f"找到 {len(experiments)} 个V45相关实验")
        
        # 2. 创建对比表格
        csv_path = output_path / "v45_comparison.csv"
        df = self.create_comparison_table(str(csv_path))
        
        # 3. 生成图表
        chart1_path = output_path / "v45_comparison_chart.png"
        self.plot_comparison_chart(str(chart1_path))
        
        chart2_path = output_path / "v45_improvement_trend.png"
        self.plot_improvement_trend(str(chart2_path))
        
        # 4. 生成文本报告
        report_path = output_path / "v45_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# V45实验结果分析报告\n\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. 实验概况\n")
            f.write(f"- 扫描的实验数量: {len(experiments)}\n")
            f.write(f"- 结果目录: {self.result_dir}\n\n")
            
            f.write("## 2. 关键发现\n")
            f.write("### 2.1 当前最佳结果\n")
            
            # 找出最佳结果
            if not df.empty:
                best_row = df.iloc[0]
                f.write(f"- **最佳方法**: {best_row['Method']}\n")
                f.write(f"- **C-index**: {best_row['C-index']:.4f}\n")
                f.write(f"- **相对于V45**: {best_row['Delta_vs_V45']:+.4f}\n")
                f.write(f"- **相对于Baseline**: {best_row['Delta_vs_Baseline']:+.4f}\n\n")
            
            f.write("### 2.2 V45家族表现\n")
            v45_family = df[df['Type'].str.contains('V45')]
            if not v45_family.empty:
                for _, row in v45_family.iterrows():
                    status = "✅" if row['Delta_vs_V45'] >= 0 else "❌"
                    f.write(f"- {status} {row['Method']}: {row['C-index']:.4f} (Δ vs V45: {row['Delta_vs_V45']:+.4f})\n")
            
            f.write("\n## 3. 改进建议\n")
            f.write("### 3.1 立即执行的改进\n")
            f.write("1. **多seed集成**：验证随机种子稳定性\n")
            f.write("2. **IPCW加权**：已通过V50实验验证\n")
            f.write("3. **排序参数调优**：调整margin和max_pairs\n\n")
            
            f.write("### 3.2 中期改进方向\n")
            f.write("1. **门控机制优化**：注意力门控或稀疏门控\n")
            f.write("2. **排序损失增强**：加权排序或分层排序\n")
            f.write("3. **训练策略优化**：课程学习或自适应损失权重\n\n")
            
            f.write("## 4. 文件清单\n")
            f.write(f"- 对比表格: `{csv_path.name}`\n")
            f.write(f"- 对比图表: `{chart1_path.name}`\n")
            f.write(f"- 趋势图表: `{chart2_path.name}`\n")
            f.write(f"- 本报告: `{report_path.name}`\n")
        
        print(f"报告已生成到: {output_dir}")
        print(f"- 对比表格: {csv_path}")
        print(f"- 对比图表: {chart1_path}")
        print(f"- 趋势图表: {chart2_path}")
        print(f"- 分析报告: {report_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V45实验结果分析工具')
    parser.add_argument('--result_dir', type=str, default='/data1/sweep_results_30ep',
                       help='结果目录路径')
    parser.add_argument('--output_dir', type=str, default='./reports',
                       help='输出目录路径')
    parser.add_argument('--scan_only', action='store_true',
                       help='仅扫描实验结果，不生成报告')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = V45Analysis(result_dir=args.result_dir)
    
    if args.scan_only:
        # 仅扫描
        experiments = analyzer.scan_v45_experiments()
        print(f"找到 {len(experiments)} 个V45相关实验:")
        for method, metrics in experiments.items():
            print(f"  {method}: C-index={metrics['val_cindex']:.4f}, std={metrics.get('std', 'N/A')}")
    else:
        # 生成完整报告
        analyzer.generate_report(args.output_dir)


if __name__ == '__main__':
    main()