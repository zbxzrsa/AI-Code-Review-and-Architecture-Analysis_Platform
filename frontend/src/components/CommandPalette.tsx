import { useEffect, useState } from 'react'
import { Input, Modal, List, Empty, Badge } from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import './CommandPalette.css'

interface Command {
  id: string
  label: string
  description: string
  shortcut: string
  category: string
  action: () => void
}

export default function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false)
  const [searchText, setSearchText] = useState('')
  const navigate = useNavigate()
  const { t } = useTranslation()

  const commands: Command[] = [
    {
      id: 'projects',
      label: t('commands.projects'),
      description: t('commands.projectsDesc'),
      shortcut: 'g+p',
      category: 'Navigation',
      action: () => {
        navigate('/')
        setIsOpen(false)
      },
    },
    {
      id: 'reviews',
      label: t('commands.reviews'),
      description: t('commands.reviewsDesc'),
      shortcut: 'g+r',
      category: 'Navigation',
      action: () => {
        navigate('/code-review')
        setIsOpen(false)
      },
    },
    {
      id: 'experiments',
      label: t('commands.experiments'),
      description: t('commands.experimentsDesc'),
      shortcut: 'g+e',
      category: 'Navigation',
      action: () => {
        navigate('/experiments')
        setIsOpen(false)
      },
    },
    {
      id: 'quarantine',
      label: t('commands.quarantine'),
      description: t('commands.quarantineDesc'),
      shortcut: 'g+q',
      category: 'Navigation',
      action: () => {
        navigate('/quarantine')
        setIsOpen(false)
      },
    },
    {
      id: 'settings',
      label: t('commands.settings'),
      description: t('commands.settingsDesc'),
      shortcut: 'g+s',
      category: 'Navigation',
      action: () => {
        navigate('/settings')
        setIsOpen(false)
      },
    },
    {
      id: 'analyze',
      label: t('commands.analyze'),
      description: t('commands.analyzeDesc'),
      shortcut: 'Cmd+Shift+A',
      category: 'Actions',
      action: () => {
        // Trigger AI analysis
        window.dispatchEvent(new CustomEvent('trigger-analysis'))
        setIsOpen(false)
      },
    },
  ]

  const filteredCommands = commands.filter(
    (cmd) =>
      cmd.label.toLowerCase().includes(searchText.toLowerCase()) ||
      cmd.description.toLowerCase().includes(searchText.toLowerCase())
  )

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl+K to open command palette
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setIsOpen(true)
      }

      // Escape to close
      if (e.key === 'Escape') {
        setIsOpen(false)
      }

      // Handle shortcut combinations
      if (e.key === 'g') {
        const nextKey = (e: KeyboardEvent) => {
          if (e.key === 'p') {
            navigate('/')
            window.removeEventListener('keydown', nextKey)
          } else if (e.key === 'r') {
            navigate('/code-review')
            window.removeEventListener('keydown', nextKey)
          } else if (e.key === 'e') {
            navigate('/experiments')
            window.removeEventListener('keydown', nextKey)
          } else if (e.key === 'q') {
            navigate('/quarantine')
            window.removeEventListener('keydown', nextKey)
          } else if (e.key === 's') {
            navigate('/settings')
            window.removeEventListener('keydown', nextKey)
          }
        }
        window.addEventListener('keydown', nextKey)
        setTimeout(() => window.removeEventListener('keydown', nextKey), 2000)
      }

      // Cmd/Ctrl+Shift+A to trigger analysis
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'A') {
        e.preventDefault()
        window.dispatchEvent(new CustomEvent('trigger-analysis'))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [navigate])

  return (
    <>
      <Modal
        title={null}
        open={isOpen}
        onCancel={() => setIsOpen(false)}
        footer={null}
        className="command-palette-modal"
        centered
        width={600}
      >
        <Input
          placeholder={t('commands.placeholder')}
          prefix={<SearchOutlined />}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          autoFocus
          size="large"
          className="command-palette-input"
        />

        <div className="command-palette-list">
          {filteredCommands.length === 0 ? (
            <Empty description={t('common.noData')} />
          ) : (
            <List
              dataSource={filteredCommands}
              renderItem={(cmd) => (
                <List.Item
                  onClick={() => {
                    cmd.action()
                    setIsOpen(false)
                  }}
                  className="command-palette-item"
                >
                  <div className="command-palette-content">
                    <div className="command-palette-label">{cmd.label}</div>
                    <div className="command-palette-description">
                      {cmd.description}
                    </div>
                  </div>
                  <Badge
                    count={cmd.shortcut}
                    className="command-palette-shortcut"
                  />
                </List.Item>
              )}
            />
          )}
        </div>
      </Modal>
    </>
  )
}
