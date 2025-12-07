/**
 * 代码编辑器组件 (Code Editor Component)
 * 
 * 功能描述:
 *   提供支持多种编程语言的代码编辑器，支持 Monaco 和 CodeMirror 两种引擎。
 * 
 * 主要特性:
 *   - 多语言语法高亮（Python, JavaScript, TypeScript, Java, Rust, C++, Go）
 *   - 可切换编辑器引擎
 *   - 只读模式支持
 *   - 可配置高度
 * 
 * 最后修改日期: 2024-12-07
 */

import { useState } from 'react'
import { Segmented, Alert } from 'antd'
import MonacoEditor from '@monaco-editor/react'
import CodeMirrorEditor from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { javascript } from '@codemirror/lang-javascript'
import { java } from '@codemirror/lang-java'
import { rust } from '@codemirror/lang-rust'
import { cpp } from '@codemirror/lang-cpp'
import { go } from '@codemirror/lang-go'
import './CodeEditor.css'

interface CodeEditorProps {
  value: string
  onChange: (value: string) => void
  language: string
  readOnly?: boolean
  height?: string
}

export default function CodeEditor({
  value,
  onChange,
  language,
  readOnly = false,
  height = '400px',
}: CodeEditorProps) {
  const [editor, setEditor] = useState<'monaco' | 'codemirror'>('monaco')
  const [isMonacoReady, setIsMonacoReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const getLanguageExtension = () => {
    switch (language.toLowerCase()) {
      case 'python':
        return python()
      case 'javascript':
      case 'typescript':
        return javascript({ typescript: true })
      case 'java':
        return java()
      case 'rust':
        return rust()
      case 'cpp':
      case 'c':
        return cpp()
      case 'go':
        return go()
      default:
        return javascript()
    }
  }

  const handleEditorChange = (value: string | undefined) => {
    if (value !== undefined) {
      onChange(value)
      setError(null)
    }
  }

  return (
    <div className="code-editor-container">
      <div className="code-editor-toolbar">
        <Segmented
          value={editor}
          onChange={(value) => setEditor(value as 'monaco' | 'codemirror')}
          options={[
            { label: 'Monaco', value: 'monaco' },
            { label: 'CodeMirror', value: 'codemirror' },
          ]}
          disabled={!isMonacoReady}
        />
      </div>

      {error && <Alert message={error} type="error" showIcon closable />}

      {editor === 'monaco' ? (
        <div className="code-editor-wrapper" style={{ height }}>
          <MonacoEditor
            height="100%"
            language={language.toLowerCase()}
            value={value}
            onChange={handleEditorChange}
            onMount={() => setIsMonacoReady(true)}
            options={{
              minimap: { enabled: false },
              readOnly,
              fontSize: 14,
              fontFamily: 'Fira Code, monospace',
              lineNumbers: 'on',
              scrollBeyondLastLine: false,
              automaticLayout: true,
              formatOnPaste: true,
              formatOnType: true,
            }}
            theme="vs-dark"
          />
        </div>
      ) : (
        <div className="code-editor-wrapper" style={{ height }}>
          <CodeMirrorEditor
            value={value}
            onChange={handleEditorChange}
            extensions={[getLanguageExtension()]}
            height={height}
            readOnly={readOnly}
            className="codemirror-editor"
          />
        </div>
      )}
    </div>
  )
}
