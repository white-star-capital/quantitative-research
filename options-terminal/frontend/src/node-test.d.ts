// Minimal types for Node's built-in test runner. @types/node is outside the approved
// dependency set (building_secure_front_ends.md §2), and tests only need this much.
declare module "node:test" {
  type Fn = () => void | Promise<void>;
  export function describe(name: string, fn: Fn): void;
  export function it(name: string, fn: Fn): void;
  export function test(name: string, fn: Fn): void;
}

declare module "node:assert/strict" {
  interface Assert {
    (value: unknown, message?: string): asserts value;
    equal<T>(actual: unknown, expected: T, message?: string): asserts actual is T;
    notEqual(actual: unknown, expected: unknown, message?: string): void;
    deepEqual<T>(actual: unknown, expected: T, message?: string): asserts actual is T;
    match(value: string, pattern: RegExp, message?: string): void;
    ok(value: unknown, message?: string): asserts value;
  }
  const assert: Assert;
  export default assert;
}

declare module "node:fs" {
  export function existsSync(path: URL): boolean;
  export function readFileSync(path: URL, encoding: "utf8"): string;
  export function readdirSync(path: URL, options: { recursive: true }): string[];
}
