/**
 * useProjects Hook Tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

// Mock API
vi.mock("@/services/api", () => ({
  apiService: {
    projects: {
      list: vi.fn(),
      get: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
    },
  },
}));

import {
  useProjects,
  useProject,
  useCreateProject,
  useUpdateProject,
  useDeleteProject,
} from "@/hooks/useProjects";
import { apiService } from "@/services/api";

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
};

const mockProjects = [
  { id: "1", name: "Project 1", description: "Desc 1", language: "python" },
  { id: "2", name: "Project 2", description: "Desc 2", language: "javascript" },
  { id: "3", name: "Project 3", description: "Desc 3", language: "typescript" },
];

describe("useProjects", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("useProjects (list)", () => {
    it("fetches projects successfully", async () => {
      (apiService.projects.list as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { items: mockProjects, total: 3 },
      });

      const { result } = renderHook(() => useProjects(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data?.items).toEqual(mockProjects);
      expect(result.current.data?.total).toBe(3);
    });

    it("handles loading state", () => {
      // Create a pending promise that never resolves
      const pendingPromise = new Promise(() => {});
      (apiService.projects.list as ReturnType<typeof vi.fn>).mockReturnValue(
        pendingPromise
      );

      const { result } = renderHook(() => useProjects(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
    });

    it("handles error state", async () => {
      (apiService.projects.list as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error("Failed to fetch")
      );

      const { result } = renderHook(() => useProjects(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toBe("Failed to fetch");
    });

    it("supports pagination", async () => {
      (apiService.projects.list as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { items: mockProjects.slice(0, 2), total: 3 },
      });

      const { result } = renderHook(() => useProjects({ page: 1, limit: 2 }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(apiService.projects.list).toHaveBeenCalledWith(
        expect.objectContaining({ page: 1, limit: 2 })
      );
    });

    it("supports filtering", async () => {
      (apiService.projects.list as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { items: [mockProjects[0]], total: 1 },
      });

      const { result } = renderHook(() => useProjects({ language: "python" }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(apiService.projects.list).toHaveBeenCalledWith(
        expect.objectContaining({ language: "python" })
      );
    });

    it("supports search", async () => {
      (apiService.projects.list as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { items: [mockProjects[0]], total: 1 },
      });

      const { result } = renderHook(
        () => useProjects({ search: "Project 1" }),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(apiService.projects.list).toHaveBeenCalledWith(
        expect.objectContaining({ search: "Project 1" })
      );
    });
  });

  describe("useProject (single)", () => {
    it("fetches single project", async () => {
      (apiService.projects.get as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: mockProjects[0],
      });

      const { result } = renderHook(() => useProject("1"), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockProjects[0]);
    });

    it("does not fetch when id is undefined", () => {
      const { result } = renderHook(() => useProject(undefined), {
        wrapper: createWrapper(),
      });

      expect(result.current.isFetching).toBe(false);
      expect(apiService.projects.get).not.toHaveBeenCalled();
    });

    it("handles 404 error", async () => {
      (apiService.projects.get as ReturnType<typeof vi.fn>).mockRejectedValue({
        response: { status: 404 },
        message: "Not found",
      });

      const { result } = renderHook(() => useProject("nonexistent"), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });
    });
  });

  describe("useCreateProject", () => {
    it("creates project successfully", async () => {
      const newProject = { name: "New Project", description: "New Desc" };
      const createdProject = { id: "4", ...newProject };

      (
        apiService.projects.create as ReturnType<typeof vi.fn>
      ).mockResolvedValue({
        data: createdProject,
      });

      const { result } = renderHook(() => useCreateProject(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.mutateAsync(newProject);
      });

      expect(apiService.projects.create).toHaveBeenCalledWith(newProject);
    });

    it("handles creation error", async () => {
      (
        apiService.projects.create as ReturnType<typeof vi.fn>
      ).mockRejectedValue(new Error("Validation error"));

      const { result } = renderHook(() => useCreateProject(), {
        wrapper: createWrapper(),
      });

      await expect(
        act(async () => {
          await result.current.mutateAsync({ name: "" });
        })
      ).rejects.toThrow("Validation error");
    });

    it("shows loading state during creation", async () => {
      // Create delayed promise helper
      const createDelayedResponse = () =>
        new Promise((resolve) => setTimeout(() => resolve({ data: {} }), 100));
      (
        apiService.projects.create as ReturnType<typeof vi.fn>
      ).mockImplementation(createDelayedResponse);

      const { result } = renderHook(() => useCreateProject(), {
        wrapper: createWrapper(),
      });

      act(() => {
        result.current.mutate({ name: "Test" });
      });

      expect(result.current.isPending).toBe(true);
    });
  });

  describe("useUpdateProject", () => {
    it("updates project successfully", async () => {
      const updates = { name: "Updated Name" };
      const updatedProject = { id: "1", ...mockProjects[0], ...updates };

      (
        apiService.projects.update as ReturnType<typeof vi.fn>
      ).mockResolvedValue({
        data: updatedProject,
      });

      const { result } = renderHook(() => useUpdateProject(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.mutateAsync({ id: "1", data: updates });
      });

      expect(apiService.projects.update).toHaveBeenCalledWith("1", updates);
    });

    it("handles update error", async () => {
      (
        apiService.projects.update as ReturnType<typeof vi.fn>
      ).mockRejectedValue(new Error("Update failed"));

      const { result } = renderHook(() => useUpdateProject(), {
        wrapper: createWrapper(),
      });

      await expect(
        act(async () => {
          await result.current.mutateAsync({ id: "1", data: {} });
        })
      ).rejects.toThrow("Update failed");
    });
  });

  describe("useDeleteProject", () => {
    it("deletes project successfully", async () => {
      (
        apiService.projects.delete as ReturnType<typeof vi.fn>
      ).mockResolvedValue({});

      const { result } = renderHook(() => useDeleteProject(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.mutateAsync("1");
      });

      expect(apiService.projects.delete).toHaveBeenCalledWith("1");
    });

    it("handles delete error", async () => {
      (
        apiService.projects.delete as ReturnType<typeof vi.fn>
      ).mockRejectedValue(new Error("Cannot delete"));

      const { result } = renderHook(() => useDeleteProject(), {
        wrapper: createWrapper(),
      });

      await expect(
        act(async () => {
          await result.current.mutateAsync("1");
        })
      ).rejects.toThrow("Cannot delete");
    });
  });

  describe("Cache Invalidation", () => {
    it("invalidates list cache after create", async () => {
      const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });

      (apiService.projects.list as ReturnType<typeof vi.fn>).mockResolvedValue({
        data: { items: mockProjects, total: 3 },
      });

      (
        apiService.projects.create as ReturnType<typeof vi.fn>
      ).mockResolvedValue({
        data: { id: "4", name: "New" },
      });

      // Prefetch list
      await queryClient.prefetchQuery({
        queryKey: ["projects"],
        queryFn: () => apiService.projects.list({}),
      });

      const wrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(
          QueryClientProvider,
          { client: queryClient },
          children
        );

      const { result } = renderHook(() => useCreateProject(), { wrapper });

      await act(async () => {
        await result.current.mutateAsync({ name: "New" });
      });

      // List query should be invalidated
      const listState = queryClient.getQueryState(["projects"]);
      expect(listState?.isInvalidated).toBe(true);
    });
  });
});
