# Frontend Technical Stack

## Overview

Modern, accessible, and performant React 18 + TypeScript frontend with comprehensive code editing, real-time collaboration, and advanced visualization capabilities.

---

## Technology Stack

### Core Framework

- **React 18.2.0** - Latest React with concurrent features
- **TypeScript 5.3** - Type-safe development
- **Vite 5.0** - Lightning-fast build tool
- **React Router 6** - Client-side routing with lazy loading

### UI Framework & Components

- **Ant Design 5** - Enterprise-grade UI components
  - Customizable theme system
  - Dark/Light/High-Contrast modes
  - WCAG 2.1 AA compliance
  - 50+ components

### Code Editors

- **Monaco Editor** - Primary editor (VS Code engine)
  - Syntax highlighting for 100+ languages
  - IntelliSense support
  - Multi-cursor editing
  - Minimap and code folding
- **CodeMirror 6** - Fallback editor
  - Lightweight alternative
  - Language extensions for Python, JavaScript, Java, Rust, C++, Go
  - Extensible architecture

### State Management

- **Zustand** - Lightweight state management
  - Minimal boilerplate
  - Persistence middleware
  - DevTools integration
- **TanStack Query (React Query)** - Server state management
  - Automatic caching
  - Background refetching
  - Optimistic updates
  - Infinite queries

### Data Visualization

- **ECharts** - Metrics dashboards

  - 30+ chart types
  - Real-time data updates
  - Interactive legends
  - Export capabilities

- **Diff2Html** - Code comparisons

  - Side-by-side diffs
  - Syntax highlighting
  - Line-by-line annotations
  - Unified diff format

- **vis.js** - Dependency graphs
  - Network visualization
  - Interactive nodes
  - Physics simulation
  - Export to PNG/SVG

### Real-time Communication

- **Server-Sent Events (SSE)** - Streaming AI responses

  - One-way server-to-client communication
  - Automatic reconnection
  - Event multiplexing
  - Lower overhead than WebSocket

- **WebSocket** - Collaborative editing
  - Cursor sharing
  - Live comments
  - Presence awareness
  - Conflict resolution

### Internationalization

- **react-i18next** - Multi-language support

  - Lazy-loaded translation bundles
  - Pluralization support
  - Namespace management
  - Language detection

- **i18next-browser-languagedetector** - Auto language detection
- **i18next-http-backend** - Remote translation loading

### Accessibility

- **WCAG 2.1 AA Compliance**
  - Keyboard navigation
  - ARIA labels and roles
  - Screen reader support
  - Color contrast ratios
  - Focus management

### HTTP Client

- **Axios** - HTTP requests
  - Request/response interceptors
  - Timeout handling
  - Request cancellation
  - Progress tracking

### Utilities

- **date-fns** - Date manipulation
- **lodash-es** - Utility functions
- **clsx** - Conditional CSS classes
- **react-helmet-async** - Document head management

---

## Key Features

### 1. Command Palette (Cmd/Ctrl+K)

Quick navigation and action execution:

```typescript
// Keyboard shortcuts
g+p → Projects
g+r → Reviews
g+e → Experiments
g+q → Quarantine
g+s → Settings
Cmd+Shift+A → Trigger AI analysis
```

**Features**:

- Fuzzy search
- Category grouping
- Keyboard-only navigation
- Custom command registration

### 2. Code Editor Integration

**Monaco Editor (Primary)**:

- Syntax highlighting for 100+ languages
- IntelliSense and autocomplete
- Multi-cursor editing
- Minimap and code folding
- Bracket matching
- Line numbers and gutter decorations

**CodeMirror (Fallback)**:

- Lightweight alternative
- Language extensions
- Automatic fallback on Monaco error
- Consistent API

**Features**:

- Language detection
- Theme switching
- Read-only mode
- Line highlighting
- Error markers

### 3. Real-time Collaboration

**WebSocket Features**:

- Cursor position sharing
- Live comment threads
- Presence awareness (who's editing)
- Conflict-free collaborative editing
- Automatic reconnection

**SSE Features**:

- Streaming AI responses
- Real-time metrics updates
- Event multiplexing
- Automatic retry

### 4. Themes

**Three Theme Modes**:

1. **Light** - Default light theme
2. **Dark** - Dark theme for reduced eye strain
3. **High-Contrast** - Enhanced contrast for accessibility

**Theme Customization**:

- Primary color
- Border radius
- Font family
- Component-level overrides

### 5. Offline Mode

**Service Worker**:

- Offline request queuing
- Background sync
- Cache-first strategy for assets
- Network status detection

**IndexedDB Storage**:

- Pending operations
- Draft code snippets
- User preferences
- Cached API responses

### 6. Responsive Design

**Breakpoints**:

- Mobile: < 576px
- Tablet: 576px - 992px
- Desktop: > 992px

**Features**:

- Fluid layouts
- Touch-friendly interactions
- Adaptive navigation
- Optimized for all screen sizes

---

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── CodeEditor.tsx          # Code editor wrapper
│   │   ├── CommandPalette.tsx       # Command palette
│   │   ├── Layout.tsx              # Main layout
│   │   ├── CodeDiff.tsx            # Diff viewer
│   │   ├── DependencyGraph.tsx      # Dependency visualization
│   │   └── MetricsDashboard.tsx     # Metrics charts
│   ├── pages/
│   │   ├── Dashboard.tsx           # Main dashboard
│   │   ├── CodeReview.tsx          # Code review page
│   │   ├── Experiments.tsx         # Experiments list
│   │   ├── ExperimentDetail.tsx    # Experiment details
│   │   ├── Quarantine.tsx          # Quarantine records
│   │   └── Settings.tsx            # Settings page
│   ├── stores/
│   │   ├── theme.ts               # Theme store
│   │   ├── auth.ts                # Authentication store
│   │   └── ui.ts                  # UI state store
│   ├── hooks/
│   │   ├── useCodeReview.ts       # Code review hook
│   │   ├── useExperiments.ts      # Experiments hook
│   │   ├── useWebSocket.ts        # WebSocket hook
│   │   └── useSSE.ts              # SSE hook
│   ├── services/
│   │   ├── api.ts                 # API client
│   │   ├── auth.ts                # Authentication
│   │   └── storage.ts             # Local storage
│   ├── lib/
│   │   ├── query-client.ts        # React Query setup
│   │   ├── axios-instance.ts      # Axios configuration
│   │   └── utils.ts               # Utility functions
│   ├── i18n/
│   │   ├── index.ts               # i18n setup
│   │   └── locales/
│   │       ├── en.json            # English translations
│   │       ├── es.json            # Spanish translations
│   │       └── zh.json            # Chinese translations
│   ├── styles/
│   │   ├── index.css              # Global styles
│   │   ├── variables.css           # CSS variables
│   │   └── theme.css              # Theme styles
│   ├── App.tsx                    # Main app component
│   └── main.tsx                   # Entry point
├── public/
│   ├── index.html
│   └── manifest.json
├── package.json
├── tsconfig.json
├── vite.config.ts
└── vitest.config.ts
```

---

## API Integration

### Axios Configuration

```typescript
// Interceptors for authentication
// Automatic token refresh
// Error handling
// Request/response transformation
```

### React Query Setup

```typescript
// Query caching
// Automatic background refetching
// Optimistic updates
// Infinite queries for pagination
```

### Real-time Updates

```typescript
// SSE for streaming responses
// WebSocket for collaborative features
// Automatic reconnection
// Event multiplexing
```

---

## Performance Optimization

### Code Splitting

- Route-based lazy loading
- Component-level code splitting
- Dynamic imports for heavy libraries

### Caching

- HTTP caching headers
- Service Worker caching
- React Query caching
- Browser cache

### Bundle Optimization

- Tree shaking
- Minification
- Gzip compression
- Image optimization

### Rendering Optimization

- Memoization
- Virtual scrolling for large lists
- Debounced search
- Lazy component loading

---

## Accessibility Features

### Keyboard Navigation

- Tab navigation through all interactive elements
- Enter/Space to activate buttons
- Arrow keys for list navigation
- Escape to close modals

### Screen Reader Support

- ARIA labels for all interactive elements
- ARIA roles for custom components
- ARIA live regions for dynamic content
- Semantic HTML

### Visual Accessibility

- Minimum 4.5:1 contrast ratio
- High-contrast theme option
- Focus indicators
- Color-blind friendly palette

### Motor Accessibility

- Large touch targets (44x44px minimum)
- Keyboard-only operation
- No time-based interactions
- Customizable animations

---

## Development Workflow

### Setup

```bash
cd frontend
npm install
npm run dev
```

### Build

```bash
npm run build
npm run preview
```

### Testing

```bash
npm run test
npm run test:ui
npm run test:coverage
```

### Linting

```bash
npm run lint
npm run type-check
```

---

## Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile browsers: Latest versions

---

## Security

### Content Security Policy

- Restricts script sources
- Prevents inline scripts
- Blocks framing
- Controls resource loading

### HTTPS/TLS

- Enforced via HSTS header
- Certificate pinning ready
- Secure cookie flags

### Input Validation

- Client-side validation
- XSS prevention
- CSRF token handling
- Sanitization of user input

---

## Monitoring

### Error Tracking

- Sentry integration ready
- Error boundaries
- Console error logging
- User session tracking

### Performance Monitoring

- Web Vitals tracking
- Performance API integration
- Network monitoring
- Memory profiling

---

## Future Enhancements

- [ ] PWA support (install as app)
- [ ] Offline-first architecture
- [ ] Advanced code analysis visualization
- [ ] Custom theme builder
- [ ] Plugin system
- [ ] Advanced search with filters
- [ ] Saved code snippets
- [ ] Code templates
