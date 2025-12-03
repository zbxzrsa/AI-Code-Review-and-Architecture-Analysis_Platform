/**
 * Media Query Hooks
 *
 * Responsive design utilities:
 * - Breakpoint detection
 * - Device detection
 * - Orientation
 * - Reduced motion preference
 */

import { useState, useEffect, useMemo } from "react";

// ============================================
// Breakpoints
// ============================================

export const breakpoints = {
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600,
} as const;

export type Breakpoint = keyof typeof breakpoints;

// ============================================
// Media Query Hook
// ============================================

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === "undefined") return;

    const mediaQuery = window.matchMedia(query);
    setMatches(mediaQuery.matches);

    const handler = (event: MediaQueryListEvent) => {
      setMatches(event.matches);
    };

    // Modern browsers
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", handler);
      return () => mediaQuery.removeEventListener("change", handler);
    }
    // Legacy browsers
    mediaQuery.addListener(handler);
    return () => mediaQuery.removeListener(handler);
  }, [query]);

  return matches;
}

// ============================================
// Breakpoint Hooks
// ============================================

export function useBreakpoint(): Breakpoint {
  const isXs = useMediaQuery(`(max-width: ${breakpoints.sm - 1}px)`);
  const isSm = useMediaQuery(
    `(min-width: ${breakpoints.sm}px) and (max-width: ${breakpoints.md - 1}px)`
  );
  const isMd = useMediaQuery(
    `(min-width: ${breakpoints.md}px) and (max-width: ${breakpoints.lg - 1}px)`
  );
  const isLg = useMediaQuery(
    `(min-width: ${breakpoints.lg}px) and (max-width: ${breakpoints.xl - 1}px)`
  );
  const isXl = useMediaQuery(
    `(min-width: ${breakpoints.xl}px) and (max-width: ${breakpoints.xxl - 1}px)`
  );

  if (isXs) return "xs";
  if (isSm) return "sm";
  if (isMd) return "md";
  if (isLg) return "lg";
  if (isXl) return "xl";
  return "xxl";
}

export function useBreakpointUp(breakpoint: Breakpoint): boolean {
  return useMediaQuery(`(min-width: ${breakpoints[breakpoint]}px)`);
}

export function useBreakpointDown(breakpoint: Breakpoint): boolean {
  const nextBreakpoint = getNextBreakpoint(breakpoint);
  // Always call the hook with a valid query to satisfy Rules of Hooks
  const query = nextBreakpoint
    ? `(max-width: ${breakpoints[nextBreakpoint] - 1}px)`
    : "(min-width: 0px)";
  const matches = useMediaQuery(query);
  // If no next breakpoint, always return true (largest breakpoint)
  return nextBreakpoint ? matches : true;
}

function getNextBreakpoint(breakpoint: Breakpoint): Breakpoint | null {
  const keys = Object.keys(breakpoints) as Breakpoint[];
  const index = keys.indexOf(breakpoint);
  return index < keys.length - 1 ? keys[index + 1] : null;
}

// ============================================
// Responsive Values Hook
// ============================================

export function useResponsive<T>(
  values: Partial<Record<Breakpoint, T>>,
  defaultValue: T
): T {
  const breakpoint = useBreakpoint();

  return useMemo(() => {
    const breakpointOrder: Breakpoint[] = ["xxl", "xl", "lg", "md", "sm", "xs"];
    const currentIndex = breakpointOrder.indexOf(breakpoint);

    // Find the first defined value at or below current breakpoint
    for (let i = currentIndex; i < breakpointOrder.length; i++) {
      const bp = breakpointOrder[i];
      if (values[bp] !== undefined) {
        return values[bp] as T;
      }
    }

    return defaultValue;
  }, [breakpoint, values, defaultValue]);
}

// ============================================
// Device Detection Hooks
// ============================================

export function useIsMobile(): boolean {
  return useBreakpointDown("md");
}

export function useIsTablet(): boolean {
  const isMd = useMediaQuery(`(min-width: ${breakpoints.md}px)`);
  const isLg = useMediaQuery(`(max-width: ${breakpoints.lg - 1}px)`);
  return isMd && isLg;
}

export function useIsDesktop(): boolean {
  return useBreakpointUp("lg");
}

export function useIsTouchDevice(): boolean {
  const [isTouch, setIsTouch] = useState(false);

  useEffect(() => {
    const hasTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;
    setIsTouch(hasTouch);
  }, []);

  return isTouch;
}

// ============================================
// Orientation Hook
// ============================================

export function useOrientation(): "portrait" | "landscape" {
  const isPortrait = useMediaQuery("(orientation: portrait)");
  return isPortrait ? "portrait" : "landscape";
}

// ============================================
// Preference Hooks
// ============================================

export function usePrefersColorScheme(): "light" | "dark" {
  const isDark = useMediaQuery("(prefers-color-scheme: dark)");
  return isDark ? "dark" : "light";
}

export function usePrefersReducedMotion(): boolean {
  return useMediaQuery("(prefers-reduced-motion: reduce)");
}

export function usePrefersContrast(): "no-preference" | "more" | "less" {
  const prefersMore = useMediaQuery("(prefers-contrast: more)");
  const prefersLess = useMediaQuery("(prefers-contrast: less)");

  if (prefersMore) return "more";
  if (prefersLess) return "less";
  return "no-preference";
}

// ============================================
// Window Size Hook
// ============================================

export function useWindowSize(): { width: number; height: number } {
  const [size, setSize] = useState({
    width: typeof window !== "undefined" ? window.innerWidth : 0,
    height: typeof window !== "undefined" ? window.innerHeight : 0,
  });

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return size;
}

// ============================================
// Viewport Hook
// ============================================

export function useViewport() {
  const size = useWindowSize();
  const breakpoint = useBreakpoint();
  const orientation = useOrientation();
  const isMobile = useIsMobile();
  const isTablet = useIsTablet();
  const isDesktop = useIsDesktop();
  const isTouch = useIsTouchDevice();

  return {
    ...size,
    breakpoint,
    orientation,
    isMobile,
    isTablet,
    isDesktop,
    isTouch,
  };
}

// ============================================
// Container Query Hook (CSS Container Queries Fallback)
// ============================================

export function useContainerSize(containerRef: React.RefObject<HTMLElement>): {
  width: number;
  height: number;
} {
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [containerRef]);

  return size;
}

// ============================================
// Export
// ============================================

export default {
  useMediaQuery,
  useBreakpoint,
  useBreakpointUp,
  useBreakpointDown,
  useResponsive,
  useIsMobile,
  useIsTablet,
  useIsDesktop,
  useIsTouchDevice,
  useOrientation,
  usePrefersColorScheme,
  usePrefersReducedMotion,
  usePrefersContrast,
  useWindowSize,
  useViewport,
  useContainerSize,
  breakpoints,
};
