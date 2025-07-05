import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AIModelAnalyzer:
    """
    Phân tích và so sánh hiệu suất của các mô hình AI
    """
    
    def __init__(self):
        self.models_data = self._load_models_data()
        self.df = pd.DataFrame(self.models_data)
        
    def _load_models_data(self) -> Dict:
        """Load dữ liệu các mô hình AI"""
        return {
            'Model': ['Claude Sonnet 4', 'Gemini 2.5 Pro', 'GPT-4.1', 'GPT-o3', 'GPT-o1'],
            
            # Performance metrics
            'SWE_bench': [72.7, 63.8, 54.6, 71.7, 48.9],
            'AIME_2024': [68.0, 92.0, 55.0, 96.7, 83.0],  # Updated with estimated values
            'GPQA_Diamond': [78.0, 84.0, 72.0, 87.7, 75.0],  # Updated with estimated values
            'Instruction_Following': [85, 75, 95, 80, 75],
            
            # Additional benchmark data
            'HumanEval_Coding': [85.2, 78.5, 80.1, 88.3, 76.4],
            'MBPP_Coding': [82.7, 75.8, 78.2, 86.1, 74.5],
            'Math_Competition': [72.5, 88.2, 68.3, 94.1, 81.6],
            'Science_QA': [76.8, 82.4, 74.2, 85.9, 78.1],
            
            # Speed metrics
            'Output_Speed_tokens_per_sec': [50.1, 143.3, 116.7, 140.5, 30.0],
            'Latency_TTFT_seconds': [1.49, 36.39, 0.44, 19.0, 45.0],
            'Processing_Time_seconds': [2.8, 8.2, 1.5, 4.7, 12.3],
            
            # Cost metrics (per million tokens)
            'Input_Cost_USD': [3.0, 1.25, 2.0, 2.0, 12.0],
            'Output_Cost_USD': [15.0, 10.0, 8.0, 8.0, 48.0],
            
            # Reliability scores (1-5 scale)
            'Consistency': [5, 5, 5, 5, 4],
            'Safety_Features': [5, 5, 5, 5, 4],
            'Error_Rate_Score': [5, 5, 5, 5, 3],
            
            # Context window (in thousands)
            'Context_Window_K': [200, 1000, 1000, 130, 128],
            
            # Overall ratings (1-5 scale)
            'Performance_Rating': [5, 5, 4, 5, 4],
            'Speed_Rating': [2, 3, 4, 3, 1],
            'Cost_Rating': [3, 4, 4, 4, 2],
            'Reliability_Rating': [5, 5, 5, 5, 4]
        }
    
    def calculate_overall_scores(self) -> pd.DataFrame:
        """Tính điểm tổng thể cho mỗi mô hình"""
        df_copy = self.df.copy()
        
        # Normalize performance metrics (0-100 scale)
        df_copy['Performance_Score'] = (
            df_copy['SWE_bench'] * 0.4 +  # Coding weight
            df_copy['Instruction_Following'] * 0.6  # Instruction following weight
        )
        
        # Speed score (inverse of latency + output speed)
        df_copy['Speed_Score'] = (
            (1 / df_copy['Latency_TTFT_seconds']) * 20 +  # Lower latency = better
            (df_copy['Output_Speed_tokens_per_sec'] / 10)  # Higher throughput = better
        )
        
        # Cost score (inverse of total cost)
        df_copy['Cost_Score'] = 100 / (df_copy['Input_Cost_USD'] + df_copy['Output_Cost_USD'])
        
        # Reliability score
        df_copy['Reliability_Score'] = (
            df_copy['Consistency'] * 20 +
            df_copy['Safety_Features'] * 20 +
            df_copy['Error_Rate_Score'] * 20
        )
        
        # Overall weighted score
        df_copy['Overall_Score'] = (
            df_copy['Performance_Score'] * 0.3 +
            df_copy['Speed_Score'] * 0.25 +
            df_copy['Cost_Score'] * 0.25 +
            df_copy['Reliability_Score'] * 0.2
        )
        
        return df_copy
    
    def create_performance_comparison(self) -> go.Figure:
        """Tạo biểu đồ so sánh hiệu suất"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('SWE-bench Performance', 'AIME 2024 Math', 
                          'Speed vs Latency', 'Cost Comparison'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # SWE-bench performance
        fig.add_trace(
            go.Bar(x=self.df['Model'], y=self.df['SWE_bench'], 
                   name='SWE-bench %', marker_color='lightblue'),
            row=1, col=1
        )
        
        # AIME 2024 (filter out None values)
        aime_data = self.df.dropna(subset=['AIME_2024'])
        fig.add_trace(
            go.Bar(x=aime_data['Model'], y=aime_data['AIME_2024'],
                   name='AIME 2024 %', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Speed vs Latency scatter
        fig.add_trace(
            go.Scatter(
                x=self.df['Latency_TTFT_seconds'],
                y=self.df['Output_Speed_tokens_per_sec'],
                mode='markers+text',
                text=self.df['Model'],
                textposition='top center',
                marker=dict(size=10, color='red'),
                name='Speed vs Latency'
            ),
            row=2, col=1
        )
        
        # Cost comparison (total cost)
        total_cost = self.df['Input_Cost_USD'] + self.df['Output_Cost_USD']
        fig.add_trace(
            go.Bar(x=self.df['Model'], y=total_cost,
                   name='Total Cost ($/M tokens)', marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="AI Models Performance Analysis")
        return fig
    
    def create_radar_chart(self) -> go.Figure:
        """Tạo biểu đồ radar so sánh tổng thể"""
        categories = ['Performance', 'Speed', 'Cost', 'Reliability']
        
        fig = go.Figure()
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, model in enumerate(self.df['Model']):
            values = [
                self.df.loc[i, 'Performance_Rating'],
                self.df.loc[i, 'Speed_Rating'],
                self.df.loc[i, 'Cost_Rating'],
                self.df.loc[i, 'Reliability_Rating']
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=categories + [categories[0]],
                fill='toself',
                name=model,
                line_color=colors[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )),
            showlegend=True,
            title="AI Models Radar Comparison (1-5 scale)"
        )
        
        return fig
    
    def calculate_value_for_money(self) -> pd.DataFrame:
        """Tính toán giá trị đồng tiền"""
        df_scores = self.calculate_overall_scores()
        
        # Value = Performance / Cost
        df_scores['Value_Score'] = df_scores['Performance_Score'] / (
            df_scores['Input_Cost_USD'] + df_scores['Output_Cost_USD']
        )
        
        return df_scores[['Model', 'Performance_Score', 'Cost_Score', 'Value_Score']].round(2)
    
    def generate_recommendations(self) -> Dict[str, str]:
        """Tạo khuyến nghị cho từng use case"""
        df_scores = self.calculate_overall_scores()
        df_value = self.calculate_value_for_money()
        
        recommendations = {}
        
        # Best for coding
        best_coding_idx = df_scores['SWE_bench'].idxmax()
        recommendations['Best for Coding'] = df_scores.loc[best_coding_idx, 'Model']
        
        # Best for speed
        best_speed_idx = df_scores['Speed_Rating'].idxmax()
        recommendations['Best for Speed'] = df_scores.loc[best_speed_idx, 'Model']
        
        # Best value for money
        best_value_idx = df_value['Value_Score'].idxmax()
        recommendations['Best Value'] = df_value.loc[best_value_idx, 'Model']
        
        # Most reliable
        best_reliability_idx = df_scores['Reliability_Rating'].idxmax()
        recommendations['Most Reliable'] = df_scores.loc[best_reliability_idx, 'Model']
        
        # Most cost-effective
        best_cost_idx = df_scores['Cost_Rating'].idxmax()
        recommendations['Most Cost-Effective'] = df_scores.loc[best_cost_idx, 'Model']
        
        return recommendations
    
    def create_cost_analysis(self) -> go.Figure:
        """Phân tích chi phí chi tiết"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Input vs Output Costs', 'Cost per Performance'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # Input vs Output costs
        fig.add_trace(
            go.Scatter(
                x=self.df['Input_Cost_USD'],
                y=self.df['Output_Cost_USD'],
                mode='markers+text',
                text=self.df['Model'],
                textposition='top center',
                marker=dict(size=12, color='blue'),
                name='Input vs Output Cost'
            ),
            row=1, col=1
        )
        
        # Cost per performance
        df_scores = self.calculate_overall_scores()
        cost_per_perf = (self.df['Input_Cost_USD'] + self.df['Output_Cost_USD']) / self.df['SWE_bench']
        
        fig.add_trace(
            go.Bar(
                x=self.df['Model'],
                y=cost_per_perf,
                name='Cost per Performance Point',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Cost Analysis")
        return fig
    
    def print_detailed_analysis(self):
        """In ra phân tích chi tiết"""
        df_scores = self.calculate_overall_scores()
        
        print("=== AI MODELS DETAILED ANALYSIS ===\n")
        
        # Top performers by category
        print("🏆 TOP PERFORMERS BY CATEGORY:")
        print(f"Best Coding Performance: {self.df.loc[self.df['SWE_bench'].idxmax(), 'Model']} ({self.df['SWE_bench'].max()}%)")
        print(f"Fastest Output: {self.df.loc[self.df['Output_Speed_tokens_per_sec'].idxmax(), 'Model']} ({self.df['Output_Speed_tokens_per_sec'].max()} tokens/s)")
        print(f"Lowest Latency: {self.df.loc[self.df['Latency_TTFT_seconds'].idxmin(), 'Model']} ({self.df['Latency_TTFT_seconds'].min()}s)")
        print(f"Most Cost-Effective: {self.df.loc[self.df['Cost_Rating'].idxmax(), 'Model']}")
        
        print("\n💰 COST COMPARISON (per 1M tokens):")
        for i, model in enumerate(self.df['Model']):
            total_cost = self.df.loc[i, 'Input_Cost_USD'] + self.df.loc[i, 'Output_Cost_USD']
            print(f"{model}: ${self.df.loc[i, 'Input_Cost_USD']}/{self.df.loc[i, 'Output_Cost_USD']} (Total: ${total_cost})")
        
        print("\n⚡ SPEED ANALYSIS:")
        for i, model in enumerate(self.df['Model']):
            print(f"{model}: {self.df.loc[i, 'Output_Speed_tokens_per_sec']} tokens/s, {self.df.loc[i, 'Latency_TTFT_seconds']}s latency")
        
        print("\n🎯 RECOMMENDATIONS:")
        recommendations = self.generate_recommendations()
        for use_case, model in recommendations.items():
            print(f"{use_case}: {model}")
    
    def export_to_csv(self, filename: str = "ai_models_analysis.csv"):
        """Xuất dữ liệu ra file CSV"""
        df_scores = self.calculate_overall_scores()
        df_scores.to_csv(filename, index=False)
        print(f"Data exported to {filename}")

    def create_swe_bench_detailed(self) -> go.Figure:
        """Tạo biểu đồ chi tiết cho SWE-bench"""
        fig = go.Figure()
        
        # Màu sắc cho từng model
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        # Tạo bar chart với gradient
        fig.add_trace(go.Bar(
            x=self.df['Model'],
            y=self.df['SWE_bench'],
            marker=dict(
                color=colors,
                line=dict(color='black', width=1)
            ),
            text=self.df['SWE_bench'],
            textposition='auto',
            name='SWE-bench Score'
        ))
        
        # Thêm đường trung bình
        avg_score = self.df['SWE_bench'].mean()
        fig.add_hline(y=avg_score, line_dash="dash", line_color="red", 
                      annotation_text=f"Average: {avg_score:.1f}%")
        
        fig.update_layout(
            title={
                'text': "🔧 SWE-bench Performance - Software Engineering Benchmark",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="AI Models",
            yaxis_title="Success Rate (%)",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        # Thêm annotations
        fig.add_annotation(
            text="Higher is better - Tests real-world software engineering tasks",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        return fig
    
    def create_aime_math_detailed(self) -> go.Figure:
        """Tạo biểu đồ chi tiết cho AIME 2024 Math"""
        fig = go.Figure()
        
        # Tạo bar chart với pattern
        fig.add_trace(go.Bar(
            x=self.df['Model'],
            y=self.df['AIME_2024'],
            marker=dict(
                color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'],
                pattern_shape=["/", "\\", "|", "-", "+"],
                line=dict(color='black', width=1)
            ),
            text=self.df['AIME_2024'],
            textposition='auto',
            name='AIME 2024 Score'
        ))
        
        # Thêm benchmark line
        human_expert_score = 85  # Typical human expert score
        fig.add_hline(y=human_expert_score, line_dash="dot", line_color="green", 
                      annotation_text=f"Human Expert: {human_expert_score}%")
        
        fig.update_layout(
            title={
                'text': "🧮 AIME 2024 Math Competition - Advanced Mathematics",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="AI Models",
            yaxis_title="Problems Solved (%)",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_speed_latency_detailed(self) -> go.Figure:
        """Tạo biểu đồ chi tiết cho Speed vs Latency"""
        fig = go.Figure()
        
        # Scatter plot với size dựa trên processing time
        fig.add_trace(go.Scatter(
            x=self.df['Latency_TTFT_seconds'],
            y=self.df['Output_Speed_tokens_per_sec'],
            mode='markers+text',
            text=self.df['Model'],
            textposition='top center',
            marker=dict(
                size=self.df['Processing_Time_seconds'] * 3,  # Scale for visibility
                color=self.df['Speed_Rating'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Speed Rating"),
                line=dict(width=2, color='black')
            ),
            name='Models Performance'
        ))
        
        # Thêm ideal zone
        fig.add_shape(
            type="rect",
            x0=0, y0=100, x1=5, y1=150,
            fillcolor="rgba(0,255,0,0.1)",
            line=dict(color="green", width=2, dash="dash"),
        )
        
        fig.add_annotation(
            x=2.5, y=125,
            text="Ideal Zone<br>Low Latency + High Speed",
            showarrow=False,
            font=dict(color="green")
        )
        
        fig.update_layout(
            title={
                'text': "⚡ Speed vs Latency Analysis - Performance Trade-offs",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Latency - Time to First Token (seconds)",
            yaxis_title="Output Speed (tokens/second)",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_cost_comparison_detailed(self) -> go.Figure:
        """Tạo biểu đồ chi tiết cho Cost Comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cost Breakdown', 'Cost per Performance', 
                          'Value for Money', 'Cost Efficiency'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Cost breakdown
        fig.add_trace(
            go.Bar(x=self.df['Model'], y=self.df['Input_Cost_USD'],
                   name='Input Cost', marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=self.df['Model'], y=self.df['Output_Cost_USD'],
                   name='Output Cost', marker_color='lightcoral'),
            row=1, col=1
        )
        
        # Cost per performance
        cost_per_perf = (self.df['Input_Cost_USD'] + self.df['Output_Cost_USD']) / self.df['SWE_bench']
        fig.add_trace(
            go.Scatter(
                x=self.df['Model'],
                y=cost_per_perf,
                mode='markers+lines',
                name='Cost per Performance',
                marker=dict(size=10, color='red')
            ),
            row=1, col=2
        )
        
        # Value for money (Performance / Total Cost)
        total_cost = self.df['Input_Cost_USD'] + self.df['Output_Cost_USD']
        value_score = self.df['SWE_bench'] / total_cost
        fig.add_trace(
            go.Bar(x=self.df['Model'], y=value_score,
                   name='Value Score', marker_color='green'),
            row=2, col=1
        )
        
        # Cost efficiency (tokens per dollar)
        efficiency = 1000 / total_cost  # tokens per dollar (scaled)
        fig.add_trace(
            go.Bar(x=self.df['Model'], y=efficiency,
                   name='Cost Efficiency', marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="💰 Comprehensive Cost Analysis",
            showlegend=True
        )
        
        return fig
    
    def create_comprehensive_benchmark_comparison(self) -> go.Figure:
        """Tạo biểu đồ so sánh toàn diện các benchmark"""
        benchmarks = ['SWE_bench', 'AIME_2024', 'GPQA_Diamond', 'HumanEval_Coding', 'MBPP_Coding']
        
        fig = go.Figure()
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, model in enumerate(self.df['Model']):
            values = [self.df.loc[i, benchmark] for benchmark in benchmarks]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=benchmarks,
                fill='toself',
                name=model,
                line=dict(color=colors[i], width=2),
                fillcolor=colors[i],
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title={
                'text': "🎯 Comprehensive Benchmark Comparison",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=600
        )
        
        return fig

    def show_all_benchmark_charts(self):
        """Hiển thị tất cả biểu đồ benchmark"""
        print("🚀 Generating detailed benchmark visualizations...")
        
        # SWE-bench Performance
        fig1 = self.create_swe_bench_detailed()
        fig1.show()
        
        # AIME 2024 Math
        fig2 = self.create_aime_math_detailed()
        fig2.show()
        
        # Speed vs Latency
        fig3 = self.create_speed_latency_detailed()
        fig3.show()
        
        # Cost Comparison
        fig4 = self.create_cost_comparison_detailed()
        fig4.show()
        
        # Comprehensive Benchmark
        fig5 = self.create_comprehensive_benchmark_comparison()
        fig5.show()
        
        print("✅ All benchmark charts generated!")
    
    def show_individual_chart(self, chart_type: str):
        """Hiển thị biểu đồ riêng lẻ theo yêu cầu"""
        chart_type = chart_type.lower()
        
        if chart_type == 'swe-bench' or chart_type == 'swe':
            fig = self.create_swe_bench_detailed()
            fig.show()
        elif chart_type == 'aime' or chart_type == 'math':
            fig = self.create_aime_math_detailed()
            fig.show()
        elif chart_type == 'speed' or chart_type == 'latency':
            fig = self.create_speed_latency_detailed()
            fig.show()
        elif chart_type == 'cost':
            fig = self.create_cost_comparison_detailed()
            fig.show()
        elif chart_type == 'comprehensive' or chart_type == 'all':
            fig = self.create_comprehensive_benchmark_comparison()
            fig.show()
        else:
            print(f"❌ Unknown chart type: {chart_type}")
            print("Available types: swe-bench, aime, speed, cost, comprehensive")
    
    def create_benchmark_summary_table(self) -> pd.DataFrame:
        """Tạo bảng tóm tắt các benchmark"""
        summary_data = {
            'Model': self.df['Model'],
            'SWE-bench (%)': self.df['SWE_bench'],
            'AIME 2024 (%)': self.df['AIME_2024'],
            'GPQA Diamond (%)': self.df['GPQA_Diamond'],
            'HumanEval (%)': self.df['HumanEval_Coding'],
            'Speed (tokens/s)': self.df['Output_Speed_tokens_per_sec'],
            'Latency (s)': self.df['Latency_TTFT_seconds'],
            'Total Cost ($)': self.df['Input_Cost_USD'] + self.df['Output_Cost_USD']
        }
        
        return pd.DataFrame(summary_data)

# Usage example
if __name__ == "__main__":
    # Khởi tạo analyzer
    analyzer = AIModelAnalyzer()
    
    # In phân tích chi tiết
    analyzer.print_detailed_analysis()
    
    # Tạo bảng tóm tắt benchmark
    print("\n📊 BENCHMARK SUMMARY TABLE:")
    summary_table = analyzer.create_benchmark_summary_table()
    print(summary_table.to_string(index=False))
    
    # Hiển thị tất cả biểu đồ benchmark chi tiết
    print("\n" + "="*50)
    print("📈 DETAILED BENCHMARK VISUALIZATIONS")
    print("="*50)
    analyzer.show_all_benchmark_charts()
    
    # Tạo và hiển thị biểu đồ gốc (để so sánh)
    print("\n" + "="*50)
    print("📊 ORIGINAL COMPARISON CHARTS")
    print("="*50)
    
    fig1 = analyzer.create_performance_comparison()
    fig1.show()
    
    fig2 = analyzer.create_radar_chart()
    fig2.show()
    
    fig3 = analyzer.create_cost_analysis()
    fig3.show()
    
    # Tính toán giá trị đồng tiền
    value_analysis = analyzer.calculate_value_for_money()
    print("\n💎 VALUE FOR MONEY ANALYSIS:")
    print(value_analysis.to_string(index=False))
    
    # Xuất dữ liệu
    analyzer.export_to_csv()
    
    print("\n✅ Analysis complete! Check the generated CSV file for detailed data.")
    
    # Hướng dẫn sử dụng
    print("\n📋 HOW TO USE INDIVIDUAL CHARTS:")
    print("analyzer.show_individual_chart('swe-bench')  # Show SWE-bench chart")
    print("analyzer.show_individual_chart('aime')       # Show AIME math chart")
    print("analyzer.show_individual_chart('speed')      # Show speed/latency chart")
    print("analyzer.show_individual_chart('cost')       # Show cost comparison chart")
    print("analyzer.show_individual_chart('comprehensive') # Show comprehensive benchmark")