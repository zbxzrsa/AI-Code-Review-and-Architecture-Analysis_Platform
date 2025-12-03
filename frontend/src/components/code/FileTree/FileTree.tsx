import React, { useState, useMemo } from 'react';
import { Tree, Input, Empty, Spin } from 'antd';
import type { DataNode, TreeProps } from 'antd/es/tree';
import {
  FileOutlined,
  FolderOutlined,
  FolderOpenOutlined,
  FileTextOutlined,
  CodeOutlined,
  PictureOutlined,
  FileMarkdownOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import './FileTree.css';

const { Search } = Input;

export interface FileNode {
  path: string;
  name: string;
  type: 'file' | 'directory';
  children?: FileNode[];
  size?: number;
  language?: string;
}

interface FileTreeProps {
  files: FileNode[];
  selectedPath?: string;
  onSelect?: (path: string, node: FileNode) => void;
  loading?: boolean;
  searchable?: boolean;
  expandedKeys?: string[];
  onExpand?: (keys: string[]) => void;
}

// Get icon based on file extension
const getFileIcon = (name: string): React.ReactNode => {
  const ext = name.split('.').pop()?.toLowerCase();
  
  const iconMap: Record<string, React.ReactNode> = {
    // Code files
    js: <CodeOutlined style={{ color: '#f7df1e' }} />,
    jsx: <CodeOutlined style={{ color: '#61dafb' }} />,
    ts: <CodeOutlined style={{ color: '#3178c6' }} />,
    tsx: <CodeOutlined style={{ color: '#3178c6' }} />,
    py: <CodeOutlined style={{ color: '#3776ab' }} />,
    java: <CodeOutlined style={{ color: '#b07219' }} />,
    go: <CodeOutlined style={{ color: '#00add8' }} />,
    rs: <CodeOutlined style={{ color: '#dea584' }} />,
    cpp: <CodeOutlined style={{ color: '#f34b7d' }} />,
    c: <CodeOutlined style={{ color: '#555555' }} />,
    cs: <CodeOutlined style={{ color: '#178600' }} />,
    rb: <CodeOutlined style={{ color: '#701516' }} />,
    php: <CodeOutlined style={{ color: '#4f5d95' }} />,
    swift: <CodeOutlined style={{ color: '#ffac45' }} />,
    kt: <CodeOutlined style={{ color: '#a97bff' }} />,
    
    // Config files
    json: <SettingOutlined style={{ color: '#cbcb41' }} />,
    yaml: <SettingOutlined style={{ color: '#cb171e' }} />,
    yml: <SettingOutlined style={{ color: '#cb171e' }} />,
    toml: <SettingOutlined style={{ color: '#9c4221' }} />,
    xml: <SettingOutlined style={{ color: '#e34c26' }} />,
    
    // Documentation
    md: <FileMarkdownOutlined style={{ color: '#083fa1' }} />,
    mdx: <FileMarkdownOutlined style={{ color: '#083fa1' }} />,
    txt: <FileTextOutlined />,
    
    // Images
    png: <PictureOutlined style={{ color: '#a074c4' }} />,
    jpg: <PictureOutlined style={{ color: '#a074c4' }} />,
    jpeg: <PictureOutlined style={{ color: '#a074c4' }} />,
    gif: <PictureOutlined style={{ color: '#a074c4' }} />,
    svg: <PictureOutlined style={{ color: '#ffb13b' }} />,
    
    // Web
    html: <CodeOutlined style={{ color: '#e34c26' }} />,
    css: <CodeOutlined style={{ color: '#563d7c' }} />,
    scss: <CodeOutlined style={{ color: '#c6538c' }} />,
    less: <CodeOutlined style={{ color: '#1d365d' }} />
  };
  
  return iconMap[ext || ''] || <FileOutlined />;
};

export const FileTree: React.FC<FileTreeProps> = ({
  files,
  selectedPath,
  onSelect,
  loading = false,
  searchable = true,
  expandedKeys: controlledExpandedKeys,
  onExpand
}) => {
  const { t } = useTranslation();
  const [searchValue, setSearchValue] = useState('');
  const [internalExpandedKeys, setInternalExpandedKeys] = useState<string[]>([]);
  
  const expandedKeys = controlledExpandedKeys ?? internalExpandedKeys;
  const setExpandedKeys = onExpand ?? setInternalExpandedKeys;

  // Convert FileNode to Ant Design TreeNode
  const convertToTreeData = (nodes: FileNode[], parentPath = ''): DataNode[] => {
    return nodes
      .sort((a, b) => {
        // Directories first, then alphabetically
        if (a.type !== b.type) {
          return a.type === 'directory' ? -1 : 1;
        }
        return a.name.localeCompare(b.name);
      })
      .map((node) => {
        const fullPath = parentPath ? `${parentPath}/${node.name}` : node.name;
        const isDirectory = node.type === 'directory';
        
        const treeNode: DataNode = {
          key: fullPath,
          title: node.name,
          icon: isDirectory 
            ? (props: any) => 
                props?.expanded ? <FolderOpenOutlined /> : <FolderOutlined />
            : getFileIcon(node.name),
          isLeaf: !isDirectory,
          children: node.children 
            ? convertToTreeData(node.children, fullPath) 
            : undefined
        };
        
        return treeNode;
      });
  };

  // Filter tree based on search
  const filterTree = (nodes: DataNode[], search: string): DataNode[] => {
    if (!search) return nodes;
    
    const searchLower = search.toLowerCase();
    
    return nodes
      .map((node) => {
        const title = String(node.title).toLowerCase();
        const matchesSearch = title.includes(searchLower);
        
        if (node.children) {
          const filteredChildren = filterTree(node.children, search);
          if (filteredChildren.length > 0 || matchesSearch) {
            return { ...node, children: filteredChildren };
          }
        } else if (matchesSearch) {
          return node;
        }
        
        return null;
      })
      .filter((node): node is DataNode => node !== null);
  };

  // Get all parent keys for expanded state (reserved for future use)
  const _getParentKeys = (nodes: DataNode[], targetKey: string, parents: string[] = []): string[] => {
    for (const node of nodes) {
      if (node.key === targetKey) {
        return parents;
      }
      if (node.children) {
        const found = _getParentKeys(node.children, targetKey, [...parents, String(node.key)]);
        if (found.length > 0) {
          return found;
        }
      }
    }
    return [];
  };

  const treeData = useMemo(() => {
    const data = convertToTreeData(files);
    return filterTree(data, searchValue);
    // convertToTreeData and filterTree are stable functions defined above
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files, searchValue]);

  // Auto-expand when searching
  const autoExpandedKeys = useMemo(() => {
    if (!searchValue) return expandedKeys;
    
    const keys: string[] = [];
    const collectKeys = (nodes: DataNode[]) => {
      nodes.forEach((node) => {
        if (node.children && node.children.length > 0) {
          keys.push(String(node.key));
          collectKeys(node.children);
        }
      });
    };
    collectKeys(treeData);
    return keys;
  }, [treeData, searchValue, expandedKeys]);

  const handleSelect: TreeProps['onSelect'] = (selectedKeys, info) => {
    if (selectedKeys.length > 0 && info.node.isLeaf) {
      const path = String(selectedKeys[0]);
      // Find the original node
      const findNode = (nodes: FileNode[], targetPath: string): FileNode | null => {
        for (const node of nodes) {
          const fullPath = node.path || node.name;
          if (fullPath === targetPath) return node;
          if (node.children) {
            const found = findNode(node.children, targetPath);
            if (found) return found;
          }
        }
        return null;
      };
      const node = findNode(files, path);
      if (node) {
        onSelect?.(path, node);
      }
    }
  };

  const handleExpand: TreeProps['onExpand'] = (keys) => {
    setExpandedKeys(keys as string[]);
  };

  if (loading) {
    return (
      <div className="file-tree-loading">
        <Spin />
      </div>
    );
  }

  return (
    <div className="file-tree">
      {searchable && (
        <div className="file-tree-search">
          <Search
            placeholder={t('file_tree.search', 'Search files...')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            allowClear
          />
        </div>
      )}
      
      {treeData.length === 0 ? (
        <Empty
          description={
            searchValue
              ? t('file_tree.no_results', 'No files found')
              : t('file_tree.empty', 'No files')
          }
        />
      ) : (
        <Tree
          showIcon
          blockNode
          treeData={treeData}
          selectedKeys={selectedPath ? [selectedPath] : []}
          expandedKeys={searchValue ? autoExpandedKeys : expandedKeys}
          onSelect={handleSelect}
          onExpand={handleExpand}
          className="file-tree-content"
        />
      )}
    </div>
  );
};

export default FileTree;
