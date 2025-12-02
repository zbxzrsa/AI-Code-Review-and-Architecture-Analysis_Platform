/**
 * React Query hooks for project management
 * 
 * Provides caching, optimistic updates, and mutations for projects.
 */

import { useQuery, useMutation, useQueryClient, UseQueryOptions } from '@tanstack/react-query';
import { message } from 'antd';
import { apiService } from '../services/api';
import type { 
  Project, 
  ProjectFilters, 
  PaginationState,
  TeamMember,
  Webhook,
  APIKey,
  ActivityLog,
} from '../store/projectStore';

/** Query keys for cache management */
export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  list: (filters: ProjectFilters, pagination: PaginationState) => 
    [...projectKeys.lists(), { filters, pagination }] as const,
  details: () => [...projectKeys.all, 'detail'] as const,
  detail: (id: string) => [...projectKeys.details(), id] as const,
  stats: (id: string) => [...projectKeys.detail(id), 'stats'] as const,
  activity: (id: string) => [...projectKeys.detail(id), 'activity'] as const,
  team: (id: string) => [...projectKeys.detail(id), 'team'] as const,
  webhooks: (id: string) => [...projectKeys.detail(id), 'webhooks'] as const,
  apiKeys: (id: string) => [...projectKeys.detail(id), 'api-keys'] as const,
};

/** Response type for paginated list */
interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
}

/**
 * Hook to fetch projects list with pagination and filtering
 */
export function useProjects(
  filters: ProjectFilters, 
  pagination: PaginationState,
  options?: Omit<UseQueryOptions<PaginatedResponse<Project>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: projectKeys.list(filters, pagination),
    queryFn: async () => {
      const response = await apiService.projects.list({
        page: pagination.page,
        limit: pagination.pageSize,
        search: filters.search || undefined,
        status: filters.status !== 'all' ? filters.status : undefined,
        language: filters.language !== 'all' ? filters.language : undefined,
        sort_field: filters.sortField,
        sort_order: filters.sortOrder,
      });
      return response.data as PaginatedResponse<Project>;
    },
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

/**
 * Hook to fetch a single project by ID
 */
export function useProject(
  id: string | undefined,
  options?: Omit<UseQueryOptions<Project>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: projectKeys.detail(id!),
    queryFn: async () => {
      const response = await apiService.projects.get(id!);
      return response.data as Project;
    },
    enabled: !!id,
    staleTime: 60 * 1000, // 1 minute
    ...options,
  });
}

/**
 * Hook to fetch project statistics
 */
export function useProjectStats(id: string | undefined) {
  return useQuery({
    queryKey: projectKeys.stats(id!),
    queryFn: async () => {
      const response = await apiService.projects.getStats(id!);
      return response.data;
    },
    enabled: !!id,
    staleTime: 60 * 1000,
  });
}

/**
 * Hook to fetch project activity log
 */
export function useProjectActivity(
  id: string | undefined,
  params?: { page?: number; limit?: number }
) {
  return useQuery({
    queryKey: [...projectKeys.activity(id!), params],
    queryFn: async () => {
      const response = await apiService.projects.getActivity(id!, params);
      return response.data as PaginatedResponse<ActivityLog>;
    },
    enabled: !!id,
    staleTime: 30 * 1000,
  });
}

/**
 * Hook to fetch project team members
 */
export function useProjectTeam(id: string | undefined) {
  return useQuery({
    queryKey: projectKeys.team(id!),
    queryFn: async () => {
      const response = await apiService.projects.getTeam(id!);
      return response.data as TeamMember[];
    },
    enabled: !!id,
  });
}

/**
 * Hook to fetch project webhooks
 */
export function useProjectWebhooks(id: string | undefined) {
  return useQuery({
    queryKey: projectKeys.webhooks(id!),
    queryFn: async () => {
      const response = await apiService.projects.getWebhooks(id!);
      return response.data as Webhook[];
    },
    enabled: !!id,
  });
}

/**
 * Hook to fetch project API keys
 */
export function useProjectApiKeys(id: string | undefined) {
  return useQuery({
    queryKey: projectKeys.apiKeys(id!),
    queryFn: async () => {
      const response = await apiService.projects.getApiKeys(id!);
      return response.data as APIKey[];
    },
    enabled: !!id,
  });
}

// ============== Mutations ==============

/**
 * Hook to create a new project
 */
export function useCreateProject() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: {
      name: string;
      description?: string;
      repository_url?: string;
      language: string;
      framework?: string;
      settings?: unknown;
    }) => {
      const response = await apiService.projects.create(data);
      return response.data as Project;
    },
    onSuccess: (newProject) => {
      // Invalidate list queries to refetch
      queryClient.invalidateQueries({ queryKey: projectKeys.lists() });
      message.success('Project created successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to create project: ${error.message}`);
    },
  });
}

/**
 * Hook to update a project
 */
export function useUpdateProject() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ id, data }: {
      id: string;
      data: Partial<{
        name: string;
        description: string;
        status: string;
        settings: unknown;
      }>;
    }) => {
      const response = await apiService.projects.update(id, data);
      return response.data as Project;
    },
    onMutate: async ({ id, data }) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: projectKeys.detail(id) });
      
      // Snapshot the previous value
      const previousProject = queryClient.getQueryData<Project>(projectKeys.detail(id));
      
      // Optimistically update to the new value
      if (previousProject) {
        queryClient.setQueryData<Project>(projectKeys.detail(id), {
          ...previousProject,
          ...data,
        });
      }
      
      return { previousProject };
    },
    onError: (error, { id }, context) => {
      // Rollback on error
      if (context?.previousProject) {
        queryClient.setQueryData(projectKeys.detail(id), context.previousProject);
      }
      message.error(`Failed to update project: ${error.message}`);
    },
    onSettled: (_, __, { id }) => {
      // Always refetch after error or success
      queryClient.invalidateQueries({ queryKey: projectKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: projectKeys.lists() });
    },
    onSuccess: () => {
      message.success('Project updated successfully');
    },
  });
}

/**
 * Hook to delete a project
 */
export function useDeleteProject() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (id: string) => {
      await apiService.projects.delete(id);
      return id;
    },
    onSuccess: (deletedId) => {
      queryClient.invalidateQueries({ queryKey: projectKeys.lists() });
      queryClient.removeQueries({ queryKey: projectKeys.detail(deletedId) });
      message.success('Project deleted successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to delete project: ${error.message}`);
    },
  });
}

/**
 * Hook to archive a project
 */
export function useArchiveProject() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (id: string) => {
      const response = await apiService.projects.archive(id);
      return response.data as Project;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: projectKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: projectKeys.lists() });
      message.success('Project archived successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to archive project: ${error.message}`);
    },
  });
}

/**
 * Hook to restore an archived project
 */
export function useRestoreProject() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (id: string) => {
      const response = await apiService.projects.restore(id);
      return response.data as Project;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: projectKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: projectKeys.lists() });
      message.success('Project restored successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to restore project: ${error.message}`);
    },
  });
}

// ============== Team Mutations ==============

/**
 * Hook to invite a team member
 */
export function useInviteTeamMember(projectId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: { email: string; role: string }) => {
      const response = await apiService.projects.inviteMember(projectId, data);
      return response.data as TeamMember;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.team(projectId) });
      message.success('Team member invited successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to invite team member: ${error.message}`);
    },
  });
}

/**
 * Hook to update team member role
 */
export function useUpdateMemberRole(projectId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ memberId, role }: { memberId: string; role: string }) => {
      const response = await apiService.projects.updateMemberRole(projectId, memberId, role);
      return response.data as TeamMember;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.team(projectId) });
      message.success('Member role updated successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to update member role: ${error.message}`);
    },
  });
}

/**
 * Hook to remove a team member
 */
export function useRemoveTeamMember(projectId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (memberId: string) => {
      await apiService.projects.removeMember(projectId, memberId);
      return memberId;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.team(projectId) });
      message.success('Team member removed successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to remove team member: ${error.message}`);
    },
  });
}

// ============== Webhook Mutations ==============

/**
 * Hook to create a webhook
 */
export function useCreateWebhook(projectId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: { url: string; events: string[]; secret?: string }) => {
      const response = await apiService.projects.createWebhook(projectId, data);
      return response.data as Webhook;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.webhooks(projectId) });
      message.success('Webhook created successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to create webhook: ${error.message}`);
    },
  });
}

/**
 * Hook to update a webhook
 */
export function useUpdateWebhook(projectId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ webhookId, data }: {
      webhookId: string;
      data: { url?: string; events?: string[]; is_active?: boolean };
    }) => {
      const response = await apiService.projects.updateWebhook(projectId, webhookId, data);
      return response.data as Webhook;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.webhooks(projectId) });
      message.success('Webhook updated successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to update webhook: ${error.message}`);
    },
  });
}

/**
 * Hook to delete a webhook
 */
export function useDeleteWebhook(projectId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (webhookId: string) => {
      await apiService.projects.deleteWebhook(projectId, webhookId);
      return webhookId;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.webhooks(projectId) });
      message.success('Webhook deleted successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to delete webhook: ${error.message}`);
    },
  });
}

/**
 * Hook to test a webhook
 */
export function useTestWebhook(projectId: string) {
  return useMutation({
    mutationFn: async (webhookId: string) => {
      const response = await apiService.projects.testWebhook(projectId, webhookId);
      return response.data;
    },
    onSuccess: () => {
      message.success('Webhook test sent successfully');
    },
    onError: (error: Error) => {
      message.error(`Webhook test failed: ${error.message}`);
    },
  });
}

// ============== API Key Mutations ==============

/**
 * Hook to create an API key
 */
export function useCreateApiKey(projectId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: { name: string; permissions: string[]; expires_at?: string }) => {
      const response = await apiService.projects.createApiKey(projectId, data);
      return response.data as APIKey & { key: string }; // Full key returned only on creation
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.apiKeys(projectId) });
      message.success('API key created successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to create API key: ${error.message}`);
    },
  });
}

/**
 * Hook to revoke an API key
 */
export function useRevokeApiKey(projectId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (keyId: string) => {
      await apiService.projects.revokeApiKey(projectId, keyId);
      return keyId;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.apiKeys(projectId) });
      message.success('API key revoked successfully');
    },
    onError: (error: Error) => {
      message.error(`Failed to revoke API key: ${error.message}`);
    },
  });
}

export default {
  useProjects,
  useProject,
  useProjectStats,
  useProjectActivity,
  useProjectTeam,
  useProjectWebhooks,
  useProjectApiKeys,
  useCreateProject,
  useUpdateProject,
  useDeleteProject,
  useArchiveProject,
  useRestoreProject,
  useInviteTeamMember,
  useUpdateMemberRole,
  useRemoveTeamMember,
  useCreateWebhook,
  useUpdateWebhook,
  useDeleteWebhook,
  useTestWebhook,
  useCreateApiKey,
  useRevokeApiKey,
};
