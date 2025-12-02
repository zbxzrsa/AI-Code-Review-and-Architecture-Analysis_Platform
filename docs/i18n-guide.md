# Internationalization (i18n) Guide / 国际化指南

This document describes the internationalization implementation for the AI Code Review Platform.

本文档描述了 AI 代码审查平台的国际化实现。

---

## Table of Contents / 目录

1. [Overview / 概述](#overview--概述)
2. [Configuration / 配置](#configuration--配置)
3. [Supported Languages / 支持的语言](#supported-languages--支持的语言)
4. [Adding Translations / 添加翻译](#adding-translations--添加翻译)
5. [Using Translations / 使用翻译](#using-translations--使用翻译)
6. [Language Selector / 语言选择器](#language-selector--语言选择器)
7. [RTL Support / 从右到左支持](#rtl-support--从右到左支持)
8. [Testing / 测试](#testing--测试)

---

## Overview / 概述

The platform uses **i18next** for internationalization with the following features:

平台使用 **i18next** 进行国际化，具有以下功能：

- ✅ **Default Language**: English (en) / 默认语言：英语
- ✅ **Multiple Languages**: English, Simplified Chinese, Traditional Chinese, Arabic / 多语言支持
- ✅ **Persistence**: localStorage for user preference / 持久化：localStorage 用于用户偏好
- ✅ **Dynamic Loading**: Load language packs on demand / 动态加载：按需加载语言包
- ✅ **RTL Support**: Right-to-left layout for Arabic / 从右到左支持
- ✅ **Fallback**: English fallback for missing translations / 回退：缺失翻译回退到英语

---

## Configuration / 配置

### File Structure / 文件结构

```
frontend/src/i18n/
├── index.ts              # Main i18n initialization / 主初始化文件
├── config.ts             # Language configuration / 语言配置
└── locales/              # Translation files / 翻译文件
    ├── en/
    │   └── translation.json
    ├── zh-CN/
    │   └── translation.json
    ├── zh-TW/
    │   └── translation.json
    └── ar/
        └── translation.json
```

### Configuration Options / 配置选项

```typescript
// config.ts
export const DEFAULT_LANGUAGE = "en"; // Default language / 默认语言
export const FALLBACK_LANGUAGE = "en"; // Fallback language / 回退语言
export const LANGUAGE_STORAGE_KEY = "app-language"; // localStorage key / 存储键
```

---

## Supported Languages / 支持的语言

| Code    | Native Name | English Name        | Direction | Flag |
| ------- | ----------- | ------------------- | --------- | ---- |
| `en`    | English     | English             | LTR       | 🇺🇸   |
| `zh-CN` | 简体中文    | Simplified Chinese  | LTR       | 🇨🇳   |
| `zh-TW` | 繁體中文    | Traditional Chinese | LTR       | 🇹🇼   |
| `ar`    | العربية     | Arabic              | RTL       | 🇸🇦   |

---

## Adding Translations / 添加翻译

### 1. Add Translation File / 添加翻译文件

Create a new folder and `translation.json` in `locales/`:

在 `locales/` 中创建新文件夹和 `translation.json`：

```json
// locales/ja/translation.json
{
  "common": {
    "loading": "読み込み中...",
    "error": "エラー"
  }
}
```

### 2. Update Configuration / 更新配置

Add the language to `config.ts`:

在 `config.ts` 中添加语言：

```typescript
export const SUPPORTED_LANGUAGES = {
  // ... existing languages
  ja: {
    code: "ja",
    nativeName: "日本語",
    englishName: "Japanese",
    direction: "ltr",
    flag: "🇯🇵",
    dateFormat: "YYYY/MM/DD",
    numberLocale: "ja-JP",
  },
};
```

### 3. Add to i18n Initialization / 添加到 i18n 初始化

Update `index.ts` to include the new language:

更新 `index.ts` 以包含新语言：

```typescript
import ja from "./locales/ja/translation.json";

const bundledResources = {
  // ... existing resources
  ja: { translation: ja },
};
```

---

## Using Translations / 使用翻译

### In Components / 在组件中

```tsx
import { useTranslation } from "react-i18next";

function MyComponent() {
  const { t } = useTranslation();

  return (
    <div>
      <h1>{t("dashboard.welcome")}</h1>
      <p>{t("dashboard.subtitle")}</p>
    </div>
  );
}
```

### With Interpolation / 带插值

```tsx
// Translation: "Showing {{count}} projects"
<p>{t("projects.showing", { count: 10 })}</p>
// Output: "Showing 10 projects"
```

### With Plurals / 带复数

```json
{
  "items": "{{count}} item",
  "items_plural": "{{count}} items"
}
```

```tsx
t("items", { count: 1 }); // "1 item"
t("items", { count: 5 }); // "5 items"
```

### Using the Hook / 使用钩子

```tsx
import { useLanguage } from "../hooks/useLanguage";

function MyComponent() {
  const { currentLanguage, setLanguage, isRTL, formatDate, formatNumber } =
    useLanguage();

  return (
    <div dir={isRTL ? "rtl" : "ltr"}>
      <p>Current: {currentLanguage}</p>
      <button onClick={() => setLanguage("zh-CN")}>Switch to Chinese</button>
      <p>Date: {formatDate(new Date())}</p>
      <p>Number: {formatNumber(1234.56)}</p>
    </div>
  );
}
```

---

## Language Selector / 语言选择器

### Basic Usage / 基本用法

```tsx
import { LanguageSelector } from '../components/common/LanguageSelector';

// Dropdown mode (default) / 下拉模式（默认）
<LanguageSelector />

// Inline mode (show all options) / 内联模式（显示所有选项）
<LanguageSelector mode="inline" />

// Icon only mode / 仅图标模式
<LanguageSelector mode="icon-only" />
```

### Props / 属性

| Prop               | Type                                    | Default      | Description               |
| ------------------ | --------------------------------------- | ------------ | ------------------------- |
| `mode`             | `'dropdown' \| 'inline' \| 'icon-only'` | `'dropdown'` | Display mode              |
| `size`             | `'small' \| 'middle' \| 'large'`        | `'middle'`   | Size                      |
| `showFlag`         | `boolean`                               | `true`       | Show flag emoji           |
| `showNativeName`   | `boolean`                               | `true`       | Show native language name |
| `onLanguageChange` | `(lang: string) => void`                | -            | Callback on change        |

---

## RTL Support / 从右到左支持

### Automatic Direction / 自动方向

The layout automatically adjusts for RTL languages:

布局自动适配从右到左语言：

```css
/* Automatic CSS classes / 自动CSS类 */
body.lang-ltr {
  direction: ltr;
}
body.lang-rtl {
  direction: rtl;
}
```

### Manual RTL Styles / 手动 RTL 样式

```css
/* RTL-specific styles / RTL特定样式 */
body.lang-rtl .sidebar {
  right: 0;
  left: auto;
}

body.lang-rtl .icon {
  margin-right: 0;
  margin-left: 8px;
}
```

### Using RTL in Components / 在组件中使用 RTL

```tsx
import { useLanguage } from "../hooks/useLanguage";

function MyComponent() {
  const { isRTL } = useLanguage();

  return (
    <div
      style={{
        textAlign: isRTL ? "right" : "left",
        direction: isRTL ? "rtl" : "ltr",
      }}
    >
      Content
    </div>
  );
}
```

---

## Testing / 测试

### Running Tests / 运行测试

```bash
# Run i18n tests / 运行国际化测试
npm run test src/i18n/__tests__/i18n.test.ts

# Run with coverage / 运行并生成覆盖率报告
npm run test:coverage
```

### Test Cases / 测试用例

1. **Initial Language / 初始语言**

   - ✅ Default is English
   - ✅ English translations load correctly

2. **Language Switching / 语言切换**

   - ✅ Switch to Chinese works
   - ✅ Persistence in localStorage

3. **Translation Coverage / 翻译覆盖**

   - ✅ All required keys exist
   - ✅ No missing translations

4. **RTL Support / 从右到左支持**
   - ✅ Arabic is detected as RTL
   - ✅ Document direction updates

### Manual Testing Checklist / 手动测试清单

- [ ] Load page - should be in English
- [ ] Click language selector - should show all options
- [ ] Switch to Chinese - UI should update immediately
- [ ] Refresh page - Chinese should persist
- [ ] Check all pages for untranslated text
- [ ] Test mobile responsive design
- [ ] Test keyboard navigation

---

## Best Practices / 最佳实践

### 1. Key Naming / 键命名

```json
{
  "module": {
    "feature": {
      "element": "Translation"
    }
  }
}
```

Example / 示例:

```json
{
  "dashboard": {
    "stats": {
      "total_projects": "Total Projects"
    }
  }
}
```

### 2. Fallback Values / 回退值

Always provide fallback values:

始终提供回退值：

```tsx
t("key.that.might.not.exist", { defaultValue: "Fallback Text" });
```

### 3. Dynamic Content / 动态内容

Use interpolation for dynamic values:

对动态值使用插值：

```tsx
// ✅ Good
t("welcome.message", { name: userName })// ❌ Bad
`Welcome, ${userName}`; // Not translatable
```

### 4. Avoid Concatenation / 避免拼接

```tsx
// ❌ Bad
t("hello") + " " + t("world");

// ✅ Good
t("hello_world");
```

---

## Troubleshooting / 故障排除

### Language Not Changing / 语言不切换

1. Check if language is in `SUPPORTED_LANGUAGES`
2. Check browser console for errors
3. Clear localStorage and refresh

### Missing Translations / 缺失翻译

1. Check if key exists in translation file
2. Check for typos in key name
3. Enable debug mode: `debug: true` in i18n config

### RTL Not Working / 从右到左不工作

1. Check if `isRTL()` returns true for the language
2. Check if `dir` attribute is set on `<html>`
3. Check CSS for RTL-specific styles

---

## Resources / 资源

- [i18next Documentation](https://www.i18next.com/)
- [react-i18next Documentation](https://react.i18next.com/)
- [Ant Design Internationalization](https://ant.design/docs/react/i18n)

---

_Last Updated / 最后更新: 2024-12-02_
