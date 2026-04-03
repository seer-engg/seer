export interface SessionsStatus {
  isPending: boolean;
  isError: boolean;
  error: unknown;
  hasNextPage?: boolean;
  fetchNextPage: () => void | Promise<unknown>;
  isFetchingNextPage: boolean;
}

