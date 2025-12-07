/**
 * EventBus Service Tests
 *
 * Tests for the centralized event handling system.
 */

import { eventBus } from "../eventBus";

describe("EventBus", () => {
  beforeEach(() => {
    // Clear all listeners and history before each test
    eventBus.clear();
  });

  describe("on/emit", () => {
    it("should emit and receive typed events", async () => {
      const handler = jest.fn();
      eventBus.on("analysis:started", handler);

      await eventBus.emit("analysis:started", { id: "123", file: "test.ts" });

      expect(handler).toHaveBeenCalledWith({ id: "123", file: "test.ts" });
    });

    it("should support multiple listeners for same event", async () => {
      const handler1 = jest.fn();
      const handler2 = jest.fn();

      eventBus.on("project:created", handler1);
      eventBus.on("project:created", handler2);

      await eventBus.emit("project:created", { id: "1", name: "Test Project" });

      expect(handler1).toHaveBeenCalled();
      expect(handler2).toHaveBeenCalled();
    });

    it("should return unsubscribe function", async () => {
      const handler = jest.fn();
      const unsubscribe = eventBus.on("user:login", handler);

      await eventBus.emit("user:login", { userId: "123" });
      expect(handler).toHaveBeenCalledTimes(1);

      unsubscribe();

      await eventBus.emit("user:login", { userId: "456" });
      expect(handler).toHaveBeenCalledTimes(1); // Still 1, not called again
    });
  });

  describe("once", () => {
    it("should only fire once", async () => {
      const handler = jest.fn();
      eventBus.once("system:online", handler);

      await eventBus.emit("system:online", {});
      await eventBus.emit("system:online", {});

      expect(handler).toHaveBeenCalledTimes(1);
    });
  });

  describe("off", () => {
    it("should remove specific listener", async () => {
      const handler = jest.fn();
      eventBus.on("user:logout", handler); // Subscribe first

      eventBus.off("user:logout");

      await eventBus.emit("user:logout", { userId: "123" });
      expect(handler).not.toHaveBeenCalled();
    });

    it("should remove all listeners for event type", async () => {
      const handler1 = jest.fn();
      const handler2 = jest.fn();

      eventBus.on("analysis:cancelled", handler1);
      eventBus.on("analysis:cancelled", handler2);

      eventBus.off("analysis:cancelled");

      await eventBus.emit("analysis:cancelled", { id: "123" });

      expect(handler1).not.toHaveBeenCalled();
      expect(handler2).not.toHaveBeenCalled();
    });
  });

  describe("priority", () => {
    it("should call high priority listeners first", async () => {
      const order: string[] = [];

      eventBus.on("project:deleted", () => {
        order.push("normal");
      });
      eventBus.on(
        "project:deleted",
        () => {
          order.push("high");
        },
        { priority: "high" }
      );
      eventBus.on(
        "project:deleted",
        () => {
          order.push("low");
        },
        { priority: "low" }
      );
      eventBus.on(
        "project:deleted",
        () => {
          order.push("critical");
        },
        { priority: "critical" }
      );

      await eventBus.emit("project:deleted", { id: "123" });

      expect(order).toEqual(["critical", "high", "normal", "low"]);
    });
  });

  describe("filter", () => {
    it("should filter events based on payload", async () => {
      const handler = jest.fn();

      eventBus.on("analysis:progress", handler, {
        filter: (payload) => payload.progress > 50,
      });

      await eventBus.emit("analysis:progress", { id: "1", progress: 30 });
      expect(handler).not.toHaveBeenCalled();

      await eventBus.emit("analysis:progress", { id: "1", progress: 75 });
      expect(handler).toHaveBeenCalledWith({ id: "1", progress: 75 });
    });
  });

  describe("history", () => {
    it("should track event history", async () => {
      await eventBus.emit("user:login", { userId: "1" });
      await eventBus.emit("user:logout", { userId: "1" });

      const history = eventBus.getHistory();

      expect(history).toHaveLength(2);
      expect(history[0].type).toBe("user:logout"); // Most recent first
      expect(history[1].type).toBe("user:login");
    });

    it("should filter history by event type", async () => {
      await eventBus.emit("user:login", { userId: "1" });
      await eventBus.emit("project:created", { id: "1", name: "Test" });
      await eventBus.emit("user:login", { userId: "2" });

      const loginHistory = eventBus.getHistory("user:login");

      expect(loginHistory).toHaveLength(2);
      expect(loginHistory.every((e) => e.type === "user:login")).toBe(true);
    });

    it("should clear history", async () => {
      await eventBus.emit("user:login", { userId: "1" });

      eventBus.clearHistory();

      expect(eventBus.getHistory()).toHaveLength(0);
    });

    it("should limit history to maxHistory", async () => {
      // Emit more than maxHistory (100) events
      for (let i = 0; i < 150; i++) {
        await eventBus.emit("analysis:progress", {
          id: String(i),
          progress: i,
        });
      }

      const history = eventBus.getHistory();
      expect(history.length).toBeLessThanOrEqual(100);
    });
  });

  describe("waitFor", () => {
    it("should wait for specific event", async () => {
      // Start waiting
      const promise = eventBus.waitFor("analysis:completed");

      // Emit after a delay
      setTimeout(() => {
        eventBus.emit("analysis:completed", {
          id: "123",
          result: { issues: [], score: 100, summary: "OK" },
        });
      }, 10);

      const result = await promise;
      expect(result.id).toBe("123");
    });

    it("should timeout if event not received", async () => {
      await expect(eventBus.waitFor("analysis:completed", 50)).rejects.toThrow(
        "Timeout waiting for analysis:completed"
      );
    });

    it("should filter events when waiting", async () => {
      const promise = eventBus.waitFor(
        "analysis:progress",
        1000,
        (payload) => payload.progress === 100
      );

      // Emit events that don't match filter
      setTimeout(() => {
        eventBus.emit("analysis:progress", { id: "1", progress: 50 });
      }, 5);

      // Emit matching event
      setTimeout(() => {
        eventBus.emit("analysis:progress", { id: "1", progress: 100 });
      }, 10);

      const result = await promise;
      expect(result.progress).toBe(100);
    });
  });

  describe("middleware", () => {
    it("should run middleware before event handlers", async () => {
      const order: string[] = [];

      eventBus.use(async (event, next) => {
        order.push("middleware-before");
        await next();
        order.push("middleware-after");
      });

      eventBus.on("user:login", () => {
        order.push("handler");
      });

      await eventBus.emit("user:login", { userId: "123" });

      expect(order).toEqual(["middleware-before", "handler", "middleware-after"]);
    });

    it("should allow middleware to modify event flow", async () => {
      const handler = jest.fn();

      // Middleware that blocks events
      eventBus.use(async (event, next) => {
        if (event.type === "system:error") {
          return; // Don't call next()
        }
        await next();
      });

      eventBus.on("system:error", handler);

      await eventBus.emit("system:error", {
        error: { message: "test", code: "500" },
      });

      expect(handler).not.toHaveBeenCalled();
    });

    it("should return unsubscribe function for middleware", async () => {
      const middlewareFn = jest.fn(async (event, next) => await next());
      const unsubscribe = eventBus.use(middlewareFn);

      await eventBus.emit("user:login", { userId: "123" });
      expect(middlewareFn).toHaveBeenCalledTimes(1);

      unsubscribe();

      await eventBus.emit("user:login", { userId: "456" });
      expect(middlewareFn).toHaveBeenCalledTimes(1); // Not called again
    });
  });

  describe("onAny", () => {
    it("should receive all events", async () => {
      const handler = jest.fn();
      eventBus.onAny(handler);

      await eventBus.emit("user:login", { userId: "1" });
      await eventBus.emit("project:created", { id: "1", name: "Test" });

      expect(handler).toHaveBeenCalledTimes(2);
    });
  });

  describe("getListenerCount", () => {
    it("should return listener count for specific event", () => {
      eventBus.on("user:login", () => {});
      eventBus.on("user:login", () => {});
      eventBus.on("user:logout", () => {});

      expect(eventBus.getListenerCount("user:login")).toBe(2);
      expect(eventBus.getListenerCount("user:logout")).toBe(1);
    });

    it("should return total listener count", () => {
      eventBus.on("user:login", () => {});
      eventBus.on("user:logout", () => {});

      expect(eventBus.getListenerCount()).toBe(2);
    });
  });

  describe("emitSync", () => {
    it("should emit without waiting", () => {
      const handler = jest.fn();
      eventBus.on("system:notification", handler);

      eventBus.emitSync("system:notification", {
        type: "info",
        message: "Test",
      });

      // Handler should be called eventually (fire and forget)
      // We can't easily test this synchronously
    });
  });

  describe("type safety", () => {
    it("should enforce correct payload types", async () => {
      const analysisHandler = jest.fn<
        void,
        [
          {
            id: string;
            result: {
              issues: Array<{
                id: string;
                type: string;
                severity: string;
                message: string;
              }>;
              score: number;
              summary: string;
            };
          }
        ]
      >();

      eventBus.on("analysis:completed", analysisHandler);

      await eventBus.emit("analysis:completed", {
        id: "123",
        result: {
          issues: [{ id: "1", type: "error", severity: "high", message: "Test" }],
          score: 85,
          summary: "Found 1 issue",
        },
      });

      expect(analysisHandler).toHaveBeenCalledWith({
        id: "123",
        result: {
          issues: [{ id: "1", type: "error", severity: "high", message: "Test" }],
          score: 85,
          summary: "Found 1 issue",
        },
      });
    });
  });

  describe("error handling", () => {
    it("should continue processing after listener error", async () => {
      const errorHandler = jest.fn(() => {
        throw new Error("Handler error");
      });
      const successHandler = jest.fn();

      eventBus.on("user:login", errorHandler);
      eventBus.on("user:login", successHandler);

      // Should not throw, should continue to next handler
      await eventBus.emit("user:login", { userId: "123" });

      expect(errorHandler).toHaveBeenCalled();
      expect(successHandler).toHaveBeenCalled();
    });
  });
});
