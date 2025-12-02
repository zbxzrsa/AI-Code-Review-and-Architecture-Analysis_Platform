/**
 * Jest configuration for frontend tests
 */
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/setupTests.ts'],
  moduleNameMapper: {
    // Handle CSS imports
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    // Handle image imports
    '\\.(jpg|jpeg|png|gif|webp|svg)$': '<rootDir>/__mocks__/fileMock.js',
    // Handle module aliases
    '^@/(.*)$': '<rootDir>/../frontend/src/$1',
  },
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: '<rootDir>/../frontend/tsconfig.json',
    }],
  },
  testMatch: [
    '<rootDir>/**/*.test.{ts,tsx}',
    '<rootDir>/**/*.spec.{ts,tsx}',
  ],
  collectCoverageFrom: [
    '../frontend/src/**/*.{ts,tsx}',
    '!../frontend/src/**/*.d.ts',
    '!../frontend/src/main.tsx',
    '!../frontend/src/vite-env.d.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 50,
      functions: 50,
      lines: 50,
      statements: 50,
    },
  },
  coverageReporters: ['text', 'lcov', 'html'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
};
