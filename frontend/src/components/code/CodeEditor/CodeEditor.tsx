import React, { useRef, useEffect, useCallback } from 'react';
import * as monaco from 'monaco-editor';
import { useTranslation } from 'react-i18next';

export interface Issue {
  id: string;
  type: string;
  severity: 'error' | 'warning' | 'info' | 'hint';
  line_start: number;
  line_end?: number;
  column_start?: number;
  column_end?: number;
  description: string;
  fix?: string;
  rule?: string;
}

interface CodeEditorProps {
  value: string;
  language: string;
  onChange?: (value: string) => void;
  readOnly?: boolean;
  issues?: Issue[];
  height?: string | number;
  theme?: 'vs-dark' | 'vs-light' | 'hc-black';
  onSave?: (value: string) => void;
}

export const CodeEditor: React.FC<CodeEditorProps> = ({
  value,
  language,
  onChange,
  readOnly = false,
  issues = [],
  height = '100%',
  theme = 'vs-dark',
  onSave
}) => {
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const decorationsRef = useRef<string[]>([]);
  const { t } = useTranslation();

  // Initialize Monaco Editor
  useEffect(() => {
    if (!containerRef.current) return;

    const editor = monaco.editor.create(containerRef.current, {
      value,
      language,
      theme,
      readOnly,
      minimap: { enabled: true },
      scrollBeyondLastLine: false,
      fontSize: 14,
      lineNumbers: 'on',
      glyphMargin: true,
      folding: true,
      automaticLayout: true,
      wordWrap: 'on',
      tabSize: 2,
      insertSpaces: true,
      renderWhitespace: 'selection',
      bracketPairColorization: { enabled: true },
      guides: {
        bracketPairs: true,
        indentation: true
      },
      suggest: {
        showKeywords: true,
        showSnippets: true
      }
    });
    
    editorRef.current = editor;

    // Listen for changes
    if (onChange) {
      editor.onDidChangeModelContent(() => {
        onChange(editor.getValue());
      });
    }

    // Add save command (Ctrl+S / Cmd+S)
    if (onSave) {
      editor.addCommand(
        monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS,
        () => {
          onSave(editor.getValue());
        }
      );
    }

    return () => {
      editor.dispose();
    };
    // onChange, onSave, value are intentionally excluded to prevent editor recreation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language, theme, readOnly]);

  // Update value when prop changes
  useEffect(() => {
    if (editorRef.current && value !== editorRef.current.getValue()) {
      editorRef.current.setValue(value);
    }
  }, [value]);

  // Add decorations for issues
  useEffect(() => {
    if (!editorRef.current) return;

    const severityToClass: Record<string, string> = {
      error: 'line-decoration-error',
      warning: 'line-decoration-warning',
      info: 'line-decoration-info',
      hint: 'line-decoration-hint'
    };

    const decorations: monaco.editor.IModelDeltaDecoration[] = issues.map(issue => ({
      range: new monaco.Range(
        issue.line_start,
        issue.column_start || 1,
        issue.line_end || issue.line_start,
        issue.column_end || 1
      ),
      options: {
        isWholeLine: !issue.column_start,
        className: severityToClass[issue.severity] || 'line-decoration-info',
        glyphMarginClassName: `glyph-${issue.severity}`,
        hoverMessage: { 
          value: `**${issue.type}** (${issue.severity})\n\n${issue.description}${issue.rule ? `\n\nRule: ${issue.rule}` : ''}`
        },
        overviewRuler: {
          color: issue.severity === 'error' ? '#ff0000' : 
                 issue.severity === 'warning' ? '#ffcc00' : '#0099ff',
          position: monaco.editor.OverviewRulerLane.Right
        }
      }
    }));

    decorationsRef.current = editorRef.current.deltaDecorations(
      decorationsRef.current,
      decorations
    );
  }, [issues]);

  // Register code action provider for quick fixes
  useEffect(() => {
    if (!editorRef.current || issues.length === 0) return;

    const disposable = monaco.languages.registerCodeActionProvider(language, {
      provideCodeActions: (model, range) => {
        const issueAtCursor = issues.find(
          issue => 
            range.startLineNumber >= issue.line_start &&
            range.endLineNumber <= (issue.line_end || issue.line_start)
        );

        if (!issueAtCursor || !issueAtCursor.fix) {
          return { actions: [], dispose: () => {} };
        }

        return {
          actions: [{
            title: t('code_editor.apply_fix', 'Apply fix'),
            kind: 'quickfix',
            diagnostics: [],
            isPreferred: true,
            edit: {
              edits: [{
                resource: model.uri,
                textEdit: {
                  range: new monaco.Range(
                    issueAtCursor.line_start,
                    1,
                    issueAtCursor.line_end || issueAtCursor.line_start,
                    model.getLineMaxColumn(issueAtCursor.line_end || issueAtCursor.line_start)
                  ),
                  text: issueAtCursor.fix
                },
                versionId: undefined
              }]
            }
          }],
          dispose: () => {}
        };
      }
    });

    return () => disposable.dispose();
  }, [issues, language, t]);

  // Jump to specific line (exposed for parent components)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _jumpToLine = useCallback((line: number) => {
    if (editorRef.current) {
      editorRef.current.revealLineInCenter(line);
      editorRef.current.setPosition({ lineNumber: line, column: 1 });
      editorRef.current.focus();
    }
  }, []);

  // Format document (exposed for parent components)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _formatDocument = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.getAction('editor.action.formatDocument')?.run();
    }
  }, []);

  return (
    <div 
      ref={containerRef} 
      style={{ 
        height: typeof height === 'number' ? `${height}px` : height, 
        width: '100%' 
      }} 
    />
  );
};

export default CodeEditor;
