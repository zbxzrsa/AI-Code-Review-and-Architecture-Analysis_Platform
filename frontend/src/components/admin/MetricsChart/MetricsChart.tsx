import React, { useRef, useEffect } from 'react';
import * as echarts from 'echarts';
import { Card, Empty, Spin } from 'antd';
import { useTranslation } from 'react-i18next';
import { useUIStore } from '../../../store/uiStore';
import './MetricsChart.css';

type EChartsOption = echarts.EChartsOption;

export interface MetricDataPoint {
  timestamp: string;
  value: number;
}

export interface MetricSeries {
  name: string;
  data: MetricDataPoint[];
  color?: string;
  type?: 'line' | 'bar' | 'area';
}

interface MetricsChartProps {
  title?: string;
  series: MetricSeries[];
  loading?: boolean;
  height?: number;
  xAxisLabel?: string;
  yAxisLabel?: string;
  yAxisMin?: number;
  yAxisMax?: number;
  showLegend?: boolean;
  showTooltip?: boolean;
  showGrid?: boolean;
}

export const MetricsChart: React.FC<MetricsChartProps> = ({
  title,
  series,
  loading = false,
  height = 300,
  xAxisLabel,
  yAxisLabel,
  yAxisMin,
  yAxisMax,
  showLegend = true,
  showTooltip = true,
  showGrid = true
}) => {
  const { t } = useTranslation();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const { resolvedTheme } = useUIStore();

  // Initialize chart
  useEffect(() => {
    if (!chartRef.current) return;

    chartInstance.current = echarts.init(chartRef.current, resolvedTheme);

    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, [resolvedTheme]);

  // Update chart options
  useEffect(() => {
    if (!chartInstance.current || series.length === 0) return;

    const isDark = resolvedTheme === 'dark';
    const textColor = isDark ? '#a0a0a0' : '#666';
    const gridColor = isDark ? '#303030' : '#e0e0e0';

    // Get all timestamps
    const allTimestamps = [...new Set(
      series.flatMap((s) => s.data.map((d) => d.timestamp))
    )].sort((a, b) => a - b); // Numeric sort instead of alphabetical

    const option: EChartsOption = {
      title: title ? {
        text: title,
        left: 'center',
        textStyle: {
          color: isDark ? '#e0e0e0' : '#262626',
          fontSize: 16
        }
      } : undefined,
      tooltip: showTooltip ? {
        trigger: 'axis',
        backgroundColor: isDark ? '#1f1f1f' : '#fff',
        borderColor: isDark ? '#303030' : '#d9d9d9',
        textStyle: {
          color: isDark ? '#e0e0e0' : '#262626'
        },
        axisPointer: {
          type: 'cross',
          label: {
            backgroundColor: isDark ? '#303030' : '#6a7985'
          }
        }
      } : undefined,
      legend: showLegend ? {
        data: series.map((s) => s.name),
        bottom: 0,
        textStyle: {
          color: textColor
        }
      } : undefined,
      grid: showGrid ? {
        left: '3%',
        right: '4%',
        bottom: showLegend ? '15%' : '3%',
        top: title ? '15%' : '3%',
        containLabel: true
      } : undefined,
      xAxis: {
        type: 'category',
        data: allTimestamps,
        name: xAxisLabel,
        nameLocation: 'middle',
        nameGap: 30,
        axisLine: {
          lineStyle: {
            color: gridColor
          }
        },
        axisLabel: {
          color: textColor,
          formatter: (value: string) => {
            const date = new Date(value);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          }
        },
        splitLine: {
          show: false
        }
      },
      yAxis: {
        type: 'value',
        name: yAxisLabel,
        min: yAxisMin,
        max: yAxisMax,
        axisLine: {
          lineStyle: {
            color: gridColor
          }
        },
        axisLabel: {
          color: textColor
        },
        splitLine: {
          lineStyle: {
            color: gridColor,
            type: 'dashed'
          }
        }
      },
      series: series.map((s) => ({
        name: s.name,
        type: s.type === 'area' ? 'line' : (s.type || 'line'),
        data: allTimestamps.map((ts) => {
          const point = s.data.find((d) => d.timestamp === ts);
          return point?.value ?? null;
        }),
        smooth: true,
        itemStyle: s.color ? { color: s.color } : undefined,
        lineStyle: s.color ? { color: s.color } : undefined,
        areaStyle: s.type === 'area' ? {
          opacity: 0.3,
          color: s.color
        } : undefined,
        connectNulls: true
      }))
    };

    chartInstance.current.setOption(option, true);
  }, [series, title, xAxisLabel, yAxisLabel, yAxisMin, yAxisMax, showLegend, showTooltip, showGrid, resolvedTheme]);

  if (loading) {
    return (
      <Card className="metrics-chart-loading">
        <Spin />
      </Card>
    );
  }

  if (series.length === 0) {
    return (
      <Card className="metrics-chart-empty">
        <Empty description={t('metrics.no_data', 'No data available')} />
      </Card>
    );
  }

  return (
    <div 
      ref={chartRef} 
      className="metrics-chart"
      style={{ height }}
    />
  );
};

export default MetricsChart;
