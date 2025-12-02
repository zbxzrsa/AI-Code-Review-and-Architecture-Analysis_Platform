/**
 * ErrorBoundary Component Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorBoundary } from '../ErrorBoundary';

// Mock error logging service
vi.mock('../../../services/errorLogging', () => ({
  errorLoggingService: {
    logComponentError: vi.fn(() => ({ id: 'test-error-id' })),
  },
  ErrorCategory: {
    CLIENT: 'client',
  },
}));

// Component that throws an error
const ThrowError: React.FC<{ shouldThrow?: boolean }> = ({ shouldThrow = true }) => {
  if (shouldThrow) {
    throw new Error('Test error message');
  }
  return <div data-testid="normal">Normal content</div>;
};

// Suppress console.error for cleaner test output
const originalConsoleError = console.error;

describe('ErrorBoundary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalConsoleError;
  });

  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <div data-testid="child">Child content</div>
      </ErrorBoundary>
    );

    expect(screen.getByTestId('child')).toBeInTheDocument();
    expect(screen.getByText('Child content')).toBeInTheDocument();
  });

  it('renders error UI when error occurs', () => {
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.queryByTestId('normal')).not.toBeInTheDocument();
  });

  it('renders custom fallback when provided', () => {
    const fallback = <div data-testid="custom-fallback">Custom Error</div>;

    render(
      <ErrorBoundary fallback={fallback}>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();
  });

  it('calls onError callback when error occurs', () => {
    const onError = vi.fn();

    render(
      <ErrorBoundary onError={onError}>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(onError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        componentStack: expect.any(String),
      })
    );
  });

  it('logs error to error logging service', async () => {
    const { errorLoggingService } = await import('../../../services/errorLogging');

    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(errorLoggingService.logComponentError).toHaveBeenCalled();
  });

  it('shows Try Again button', () => {
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('shows Reload Page button', () => {
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByRole('button', { name: /reload page/i })).toBeInTheDocument();
  });

  it('shows Go to Dashboard button', () => {
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByRole('button', { name: /go to dashboard/i })).toBeInTheDocument();
  });

  it('recovers when Try Again is clicked and error is resolved', () => {
    let shouldThrow = true;

    const ThrowOnce: React.FC = () => {
      if (shouldThrow) {
        throw new Error('Test error');
      }
      return <div data-testid="recovered">Recovered</div>;
    };

    const { rerender } = render(
      <ErrorBoundary>
        <ThrowOnce />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();

    // Fix the error
    shouldThrow = false;

    // Click Try Again
    fireEvent.click(screen.getByRole('button', { name: /try again/i }));

    // Re-render with fixed component
    rerender(
      <ErrorBoundary>
        <ThrowOnce />
      </ErrorBoundary>
    );

    expect(screen.getByTestId('recovered')).toBeInTheDocument();
  });

  it('shows error details in development mode', () => {
    const originalEnv = process.env.NODE_ENV;
    
    // Mock development mode
    vi.stubEnv('NODE_ENV', 'development');

    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByText(/error details/i)).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();

    vi.unstubAllEnvs();
  });
});

describe('ErrorBoundary Accessibility', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalConsoleError;
  });

  it('has accessible error message', () => {
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    const heading = screen.getByRole('heading');
    expect(heading).toHaveTextContent('Something went wrong');
  });

  it('has accessible buttons', () => {
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    const buttons = screen.getAllByRole('button');
    buttons.forEach(button => {
      expect(button).toHaveAccessibleName();
    });
  });

  it('error UI is keyboard navigable', () => {
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    const buttons = screen.getAllByRole('button');
    
    buttons[0].focus();
    expect(document.activeElement).toBe(buttons[0]);
    
    buttons[1].focus();
    expect(document.activeElement).toBe(buttons[1]);
  });
});

describe('ErrorBoundary Error Categories', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalConsoleError;
  });

  it('handles TypeError', () => {
    const ThrowTypeError: React.FC = () => {
      throw new TypeError('Type error');
    };

    render(
      <ErrorBoundary>
        <ThrowTypeError />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('handles SyntaxError', () => {
    const ThrowSyntaxError: React.FC = () => {
      throw new SyntaxError('Syntax error');
    };

    render(
      <ErrorBoundary>
        <ThrowSyntaxError />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('handles ReferenceError', () => {
    const ThrowReferenceError: React.FC = () => {
      throw new ReferenceError('Reference error');
    };

    render(
      <ErrorBoundary>
        <ThrowReferenceError />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });
});

describe('ErrorBoundary Nested', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalConsoleError;
  });

  it('inner boundary catches error first', () => {
    const outerFallback = <div data-testid="outer-fallback">Outer</div>;
    const innerFallback = <div data-testid="inner-fallback">Inner</div>;

    render(
      <ErrorBoundary fallback={outerFallback}>
        <div>
          <ErrorBoundary fallback={innerFallback}>
            <ThrowError />
          </ErrorBoundary>
        </div>
      </ErrorBoundary>
    );

    expect(screen.getByTestId('inner-fallback')).toBeInTheDocument();
    expect(screen.queryByTestId('outer-fallback')).not.toBeInTheDocument();
  });

  it('outer boundary catches if inner is not present', () => {
    const outerFallback = <div data-testid="outer-fallback">Outer</div>;

    render(
      <ErrorBoundary fallback={outerFallback}>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByTestId('outer-fallback')).toBeInTheDocument();
  });
});
