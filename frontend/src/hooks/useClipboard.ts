/**
 * Clipboard Hook
 *
 * Clipboard operations with:
 * - Copy to clipboard
 * - Read from clipboard
 * - Copy notifications
 * - Fallback support
 */

import { useState, useCallback } from "react";
import { message } from "antd";

interface ClipboardOptions {
  successMessage?: string;
  errorMessage?: string;
  showNotification?: boolean;
}

interface ClipboardState {
  copied: boolean;
  error: Error | null;
}

export function useClipboard(options: ClipboardOptions = {}) {
  const {
    successMessage = "Copied to clipboard!",
    errorMessage = "Failed to copy",
    showNotification = true,
  } = options;

  const [state, setState] = useState<ClipboardState>({
    copied: false,
    error: null,
  });

  const copy = useCallback(
    async (text: string): Promise<boolean> => {
      try {
        // Try modern clipboard API
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          // Fallback for older browsers
          const textArea = document.createElement("textarea");
          textArea.value = text;
          textArea.style.position = "fixed";
          textArea.style.left = "-999999px";
          textArea.style.top = "-999999px";
          document.body.appendChild(textArea);
          textArea.focus();
          textArea.select();

          const success = document.execCommand("copy");
          document.body.removeChild(textArea);

          if (!success) {
            throw new Error("execCommand failed");
          }
        }

        setState({ copied: true, error: null });

        if (showNotification) {
          message.success(successMessage);
        }

        // Reset copied state after 2 seconds
        setTimeout(() => {
          setState((prev) => ({ ...prev, copied: false }));
        }, 2000);

        return true;
      } catch (error) {
        const err = error as Error;
        setState({ copied: false, error: err });

        if (showNotification) {
          message.error(errorMessage);
        }

        return false;
      }
    },
    [successMessage, errorMessage, showNotification]
  );

  const read = useCallback(async (): Promise<string | null> => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        return await navigator.clipboard.readText();
      }
      return null;
    } catch (error) {
      setState((prev) => ({ ...prev, error: error as Error }));
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ copied: false, error: null });
  }, []);

  return {
    copy,
    read,
    reset,
    copied: state.copied,
    error: state.error,
  };
}

/**
 * Copy code block with syntax
 */
export function useCopyCode() {
  const { copy, copied } = useClipboard({
    successMessage: "Code copied!",
  });

  const copyCode = useCallback(
    async (code: string, _language?: string) => {
      // Clean up code (remove line numbers if present)
      const cleanCode = code
        .split("\n")
        .map((line) => line.replace(/^\s*\d+\s*\|\s*/, ""))
        .join("\n")
        .trim();

      return copy(cleanCode);
    },
    [copy]
  );

  return { copyCode, copied };
}

/**
 * Copy formatted JSON
 */
export function useCopyJson() {
  const { copy, copied } = useClipboard({
    successMessage: "JSON copied!",
  });

  const copyJson = useCallback(
    async (data: any, pretty: boolean = true) => {
      const json = pretty
        ? JSON.stringify(data, null, 2)
        : JSON.stringify(data);
      return copy(json);
    },
    [copy]
  );

  return { copyJson, copied };
}

/**
 * Copy URL with optional parameters
 */
export function useCopyUrl() {
  const { copy, copied } = useClipboard({
    successMessage: "URL copied!",
  });

  const copyUrl = useCallback(
    async (path?: string, params?: Record<string, string>) => {
      const url = new URL(
        path || window.location.pathname,
        window.location.origin
      );

      if (params) {
        Object.entries(params).forEach(([key, value]) => {
          url.searchParams.set(key, value);
        });
      }

      return copy(url.toString());
    },
    [copy]
  );

  const copyCurrentUrl = useCallback(async () => {
    return copy(window.location.href);
  }, [copy]);

  return { copyUrl, copyCurrentUrl, copied };
}

export default useClipboard;
