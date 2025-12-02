import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ConfigProvider, theme } from 'antd'
import { useThemeStore } from './stores/theme'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import CodeReview from './pages/CodeReview'
import Experiments from './pages/Experiments'
import ExperimentDetail from './pages/ExperimentDetail'
import Quarantine from './pages/Quarantine'
import Settings from './pages/Settings'
import CommandPalette from './components/CommandPalette'
import './App.css'

export default function App() {
  const { mode } = useThemeStore()

  const themeConfig = {
    token: {
      colorPrimary: '#1890ff',
      borderRadius: 6,
    },
    algorithm: mode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
  }

  return (
    <ConfigProvider theme={themeConfig}>
      <Router>
        <CommandPalette />
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/code-review" element={<CodeReview />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/experiments/:id" element={<ExperimentDetail />} />
            <Route path="/quarantine" element={<Quarantine />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Layout>
      </Router>
    </ConfigProvider>
  )
}
