import { afterEach, describe, expect, it, vi } from "vitest";
import {
  SCAN_SETTINGS_EVENT,
  isGdtScanEnabled,
  setGdtScanEnabled,
} from "@/lib/scanSettings";

function stubBrowser(storage: Storage | null) {
  const events = new EventTarget();
  vi.stubGlobal("window", events);
  vi.stubGlobal("CustomEvent", class extends Event {});
  vi.stubGlobal(
    "localStorage",
    storage ??
      new Proxy({} as Storage, {
        get: () => () => {
          throw new Error("storage blocked");
        },
      })
  );
  return events;
}

function memoryStorage(): Storage {
  const items = new Map<string, string>();
  return {
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => void items.set(key, value),
  } as Storage;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("GD&T scan setting", () => {
  it("is on until it is turned off, and stays as set", () => {
    stubBrowser(memoryStorage());
    expect(isGdtScanEnabled()).toBe(true);
    setGdtScanEnabled(false);
    expect(isGdtScanEnabled()).toBe(false);
    setGdtScanEnabled(true);
    expect(isGdtScanEnabled()).toBe(true);
  });

  it("tells open menus that it changed", () => {
    const events = stubBrowser(memoryStorage());
    const heard = vi.fn();
    events.addEventListener(SCAN_SETTINGS_EVENT, heard);
    setGdtScanEnabled(false);
    expect(heard).toHaveBeenCalledOnce();
  });

  it("still holds for the page when storage is blocked", () => {
    stubBrowser(null);
    setGdtScanEnabled(false);
    expect(isGdtScanEnabled()).toBe(false);
    setGdtScanEnabled(true);
  });
});

