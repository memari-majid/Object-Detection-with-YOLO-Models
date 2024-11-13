"""
HTML Report Generator for Defect Analysis
"""

import jinja2
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_html_report(report_data, visualizations, output_path):
    """Generate comprehensive HTML report."""
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wind Turbine Blade Defect Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .section { margin-bottom: 30px; }
            .visualization { margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Wind Turbine Blade Defect Analysis Report</h1>
        
        <div class="section">
            <h2>Dataset Overview</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {% for key, value in overview.items() %}
                <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
                {% endfor %}
            </table>
        </div>

        <div class="section">
            <h2>Size Analysis</h2>
            <img src="data:image/png;base64,{{ size_distribution_plot }}" />
            <table>
                {% for category, stats in size_analysis.items() %}
                <tr><td>{{ category }}</td><td>{{ stats }}</td></tr>
                {% endfor %}
            </table>
        </div>

        <div class="section">
            <h2>Difficulty Analysis</h2>
            <img src="data:image/png;base64,{{ difficulty_plot }}" />
        </div>

        <div class="section">
            <h2>Recommendations</h2>
            <ul>
            {% for rec in recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Generate HTML
    html = jinja2.Template(template).render(
        overview=report_data['dataset_overview'],
        size_analysis=report_data['size_analysis'],
        recommendations=report_data['recommendations']
    )
    
    # Save HTML report
    with open(output_path, 'w') as f:
        f.write(html) 