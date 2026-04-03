---
paths: src/**/*.{ts,tsx}
---

# TypeScript Patterns and Conventions

These rules apply to all TypeScript and TypeScript React files in the project.

## Component Props Patterns

### Extending HTML Attributes

Components should extend appropriate React HTML attributes:

```tsx
// Button component
export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "destructive" | "outline";
  size?: "default" | "sm" | "lg";
}

// Input component
export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  // Additional props
}

// Div-based component
export interface CardProps
  extends React.HTMLAttributes<HTMLDivElement> {
  // Additional props
}
```

### CVA Component Props

For components using Class Variance Authority, use `VariantProps`:

```tsx
import { cva, type VariantProps } from "class-variance-authority";

const buttonVariants = cva("base-classes", {
  variants: {
    variant: { default: "...", destructive: "..." },
    size: { default: "...", sm: "...", lg: "..." },
  },
});

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}
```

### Wrapping Native Elements

Use `React.ComponentProps` to extract props from native elements:

```tsx
// Extract input props
type InputProps = React.ComponentProps<"input">;

// With additional custom props
interface CustomInputProps extends React.ComponentProps<"input"> {
  label?: string;
  error?: string;
}
```

### Always Export Props Interfaces

```tsx
// Export the props interface
export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

// Export the component
export const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant, ...props }, ref) => {
    return <div className={cn(badgeVariants({ variant, className }))} ref={ref} {...props} />;
  }
);
```

## Export Conventions

### Named Exports (Preferred)

Use named exports for all components, types, and utilities:

```tsx
// Component
export { Button, buttonVariants };
export type { ButtonProps };

// Utility
export { cn } from "./utils";

// Types
export type { User, Workflow, AgentTrace };
```

### Default Exports (Avoid)

Avoid default exports except for:
- Page components (route-level components)
- Lazy-loaded components

```tsx
// Page component - default export OK
export default function WorkflowsPage() {
  return <div>Workflows</div>;
}

// Lazy loading - default export OK
const WorkflowCanvas = React.lazy(() => import("./WorkflowCanvas"));
```

### Barrel Exports

Use index files to re-export from a directory:

```tsx
// src/components/ui/index.ts
export { Button, buttonVariants } from "./button";
export { Badge, badgeVariants } from "./badge";
export { Input } from "./input";
export type { ButtonProps, BadgeProps, InputProps };
```

## Type Safety

### Strict Mode

TypeScript strict mode is enabled. Follow these rules:

- No implicit `any` types
- Proper null/undefined handling
- No unused variables or parameters
- Proper return types

### Avoid `any` Types

Use proper types instead of `any`:

```tsx
// Bad
function processData(data: any) {
  return data.value;
}

// Good - use unknown and type guard
function processData(data: unknown) {
  if (isValidData(data)) {
    return data.value;
  }
  throw new Error("Invalid data");
}

// Good - use generics
function processData<T extends { value: string }>(data: T) {
  return data.value;
}

// Good - define proper type
interface DataType {
  value: string;
  count: number;
}

function processData(data: DataType) {
  return data.value;
}
```

### Null/Undefined Handling

Handle null and undefined explicitly:

```tsx
// Bad
function getUserName(user: User) {
  return user.name; // Error if user is null
}

// Good - optional chaining
function getUserName(user: User | null) {
  return user?.name ?? "Anonymous";
}

// Good - type guard
function getUserName(user: User | null): string {
  if (!user) {
    return "Anonymous";
  }
  return user.name;
}
```

### Use Type Inference

Let TypeScript infer types when possible:

```tsx
// Type is inferred as string
const name = "Seer";

// Type is inferred as number[]
const numbers = [1, 2, 3];

// Type is inferred from function return
const result = calculateTotal(items); // result type inferred from calculateTotal return type
```

## Generics

### When to Use Generics

Use generics for reusable, type-safe components and functions:

```tsx
// Generic component
interface SelectProps<T> {
  options: T[];
  value: T;
  onChange: (value: T) => void;
  renderOption: (option: T) => React.ReactNode;
}

function Select<T>({ options, value, onChange, renderOption }: SelectProps<T>) {
  return (
    <select onChange={(e) => onChange(options[Number(e.target.value)])}>
      {options.map((option, index) => (
        <option key={index} value={index}>
          {renderOption(option)}
        </option>
      ))}
    </select>
  );
}

// Generic utility
function groupBy<T, K extends keyof T>(items: T[], key: K): Record<string, T[]> {
  return items.reduce((acc, item) => {
    const groupKey = String(item[key]);
    (acc[groupKey] = acc[groupKey] || []).push(item);
    return acc;
  }, {} as Record<string, T[]>);
}
```

## React-Specific Patterns

### forwardRef with Generics

```tsx
import * as React from "react";

interface ComponentProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary";
}

const Component = React.forwardRef<HTMLDivElement, ComponentProps>(
  ({ className, variant = "default", ...props }, ref) => {
    return <div ref={ref} className={className} {...props} />;
  }
);
Component.displayName = "Component";

export { Component };
export type { ComponentProps };
```

### Event Handlers

Use proper React event types:

```tsx
// Click handlers
const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
  event.preventDefault();
  console.log("Clicked!");
};

// Change handlers
const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
  setValue(event.target.value);
};

// Form submit
const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
  event.preventDefault();
  // Handle form submission
};

// Generic handler
const handleEvent = (event: React.SyntheticEvent) => {
  // Handle any React event
};
```

### Children Props

Use `React.ReactNode` for children:

```tsx
interface CardProps {
  children: React.ReactNode;
  title?: string;
}

function Card({ children, title }: CardProps) {
  return (
    <div>
      {title && <h2>{title}</h2>}
      {children}
    </div>
  );
}
```

## State and Hooks

### useState with Types

Provide type arguments when TypeScript can't infer:

```tsx
// Type inferred as string
const [name, setName] = useState("Initial");

// Explicit type when initial value is null
const [user, setUser] = useState<User | null>(null);

// Explicit type for arrays
const [items, setItems] = useState<Item[]>([]);

// Union types
const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
```

### Custom Hooks

Type custom hooks properly:

```tsx
function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = (value: T) => {
    setStoredValue(value);
    window.localStorage.setItem(key, JSON.stringify(value));
  };

  return [storedValue, setValue];
}
```

## Type Utilities

### Common TypeScript Utilities

Use built-in TypeScript utilities:

```tsx
// Partial - make all properties optional
type PartialUser = Partial<User>;

// Required - make all properties required
type RequiredUser = Required<User>;

// Pick - select specific properties
type UserNameAndEmail = Pick<User, "name" | "email">;

// Omit - exclude specific properties
type UserWithoutPassword = Omit<User, "password">;

// Record - create object type with specific keys
type UserMap = Record<string, User>;

// Extract - extract types from union
type Status = "idle" | "loading" | "success" | "error";
type SuccessOrError = Extract<Status, "success" | "error">;

// Exclude - exclude types from union
type NonErrorStatus = Exclude<Status, "error">;
```

## Import Organization

Organize imports in this order:

```tsx
// 1. External libraries
import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Link } from "react-router-dom";

// 2. Internal utilities and hooks
import { cn } from "@/lib/utils";
import { useWorkflow } from "@/hooks/useWorkflow";

// 3. Component imports
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

// 4. Type imports (if not inlined above)
import type { User, Workflow } from "@/types";
```

## Summary Checklist

When writing TypeScript:

- [ ] Extend appropriate React HTML attributes for component props
- [ ] Use `VariantProps` for CVA-based components
- [ ] Export both component and Props type
- [ ] Use named exports (avoid default exports except for pages/lazy components)
- [ ] Avoid `any` types - use proper types or generics
- [ ] Handle null/undefined explicitly with optional chaining or type guards
- [ ] Use `React.forwardRef` for components that need ref forwarding
- [ ] Set `displayName` for debugging
- [ ] Provide proper types for event handlers
- [ ] Use `React.ReactNode` for children props
- [ ] Type `useState` when TypeScript can't infer the type
- [ ] Organize imports: external → internal → components → types
