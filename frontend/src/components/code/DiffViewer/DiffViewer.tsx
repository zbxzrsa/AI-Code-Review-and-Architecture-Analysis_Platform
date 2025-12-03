import React, { useMemo } from 'react';
import { html } from 'diff2html';
import { useTranslation } from 'react-i18next';
import { Button, Space, Segmented, Tooltip } from 'antd';
import { 
  SwapOutlined, 
  ColumnWidthOutlined, 
  CopyOutlined,
  DownloadOutlined 
} from '@ant-design/icons';
import './DiffViewer.css';

interface DiffViewerProps {
  oldCode: string;
  newCode: string;
  oldFileName?: string;
  newFileName?: string;
  language?: string;
  outputFormat?: 'side-by-side' | 'line-by-line';
  onFormatChange?: (format: 'side-by-side' | 'line-by-line') => void;
}

export const DiffViewer: React.FC<DiffViewerProps> = ({
  oldCode,
  newCode,
  oldFileName = 'original',
  newFileName = 'modified',
  language: _language = 'plaintext',
  outputFormat = 'side-by-side',
  onFormatChange
}) => {
  const { t } = useTranslation();
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Generate unified diff
  const unifiedDiff = useMemo(() => {
    const oldLines = oldCode.split('\n');
    const newLines = newCode.split('\n');
    
    let diff = `--- ${oldFileName}\n+++ ${newFileName}\n`;
    
    // Simple diff generation (for production, use a proper diff library)
    const maxLines = Math.max(oldLines.length, newLines.length);
    let hunkStart = -1;
    // Track hunk state
    const _hunkLines: string[] = [];
    
    for (let i = 0; i < maxLines; i++) {
      const oldLine = oldLines[i] || '';
      const newLine = newLines[i] || '';
      
      if (oldLine !== newLine) {
        if (hunkStart === -1) {
          hunkStart = i;
          diff += `@@ -${i + 1},${oldLines.length} +${i + 1},${newLines.length} @@\n`;
        }
        if (oldLines[i] !== undefined) {
          diff += `-${oldLine}\n`;
        }
        if (newLines[i] !== undefined) {
          diff += `+${newLine}\n`;
        }
      } else {
        diff += ` ${oldLine}\n`;
      }
    }
    
    return diff;
  }, [oldCode, newCode, oldFileName, newFileName]);

  // Generate HTML diff
  const diffHtml = useMemo(() => {
    return html(unifiedDiff, {
      drawFileList: false,
      matching: 'lines',
      outputFormat: outputFormat === 'side-by-side' ? 'side-by-side' : 'line-by-line',
      renderNothingWhenEmpty: false,
    } as any);
  }, [unifiedDiff, outputFormat]);

  // Copy diff to clipboard
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(unifiedDiff);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  // Download diff as file
  const handleDownload = () => {
    const blob = new Blob([unifiedDiff], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${newFileName}.diff`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="diff-viewer">
      <div className="diff-viewer-toolbar">
        <Space>
          <Segmented
            value={outputFormat}
            onChange={(value) => onFormatChange?.(value as 'side-by-side' | 'line-by-line')}
            options={[
              {
                label: (
                  <Tooltip title={t('diff.side_by_side', 'Side by Side')}>
                    <ColumnWidthOutlined />
                  </Tooltip>
                ),
                value: 'side-by-side'
              },
              {
                label: (
                  <Tooltip title={t('diff.line_by_line', 'Line by Line')}>
                    <SwapOutlined />
                  </Tooltip>
                ),
                value: 'line-by-line'
              }
            ]}
          />
        </Space>
        <Space>
          <Button
            size="small"
            icon={<CopyOutlined />}
            onClick={handleCopy}
          >
            {t('common.copy', 'Copy')}
          </Button>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            onClick={handleDownload}
          >
            {t('common.download', 'Download')}
          </Button>
        </Space>
      </div>
      <div 
        ref={containerRef}
        className="diff-viewer-content"
        dangerouslySetInnerHTML={{ __html: diffHtml }}
      />
    </div>
  );
};

export default DiffViewer;
