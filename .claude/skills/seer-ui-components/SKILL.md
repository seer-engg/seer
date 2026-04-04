---
name: seer-ui-components
description: Apply Seer design system patterns when creating or modifying UI components. Auto-triggers for badge, button, input, and form components in src/components/.
allowed-tools: Read, Edit, Glob, Grep
---

# Seer UI Components Skill

This skill automatically applies when Claude Code works with UI components in the Seer Frontend project. It ensures consistency with the design system defined in `src/index.css` and component patterns in `src/components/ui/`.

## When This Skill Applies

Claude Code automatically uses this skill when:

- Creating new UI components in `src/components/ui/`
- Modifying existing UI components (Badge, Button, Input, etc.)
- Asked to style badges, pills, buttons, or inputs
- Working with component variants or theming
- Implementing design system patterns

## Implementation Steps

### 1. Check Existing Components First

Always reference existing implementations before creating or modifying components:

**Base components to check:**
- `src/components/ui/badge.tsx` - Badge component with CVA variants
- `src/components/ui/button.tsx` - Button component with variants and sizes
- `src/components/ui/input.tsx` - Input component with consistent styling

**Reference implementations:**
- `src/components/workflows/build/CategoryBadge.tsx:10-34` - Preferred minimal badge pattern with semantic colors
- `src/components/workflows/blocks/ToolBlockNode.tsx:126-151` - Status badges with icons (critical indicators only)
- `src/components/workflows/build/ToolsSection.tsx:106-108` - Count badges

### 2. Apply Design System Patterns

**For Badge Components:**

Use the **preferred minimal pattern** for labels, counts, and non-critical indicators:

```tsx
// Preferred: Minimal, subtle, no icons
<Badge
  variant="secondary"
  className="h-4 px-1 text-[9px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
>
  Ready
</Badge>
```

**Reserve icon-based badges for critical status only:**

```tsx
// Only for important status indicators
<Badge
  variant="secondary"
  className="flex items-center gap-1 text-[10px] px-1.5 h-5 bg-emerald-500/10 text-emerald-600 border-emerald-500/20"
>
  <CheckCircle2 className="w-3 h-3" />
  Connected
</Badge>
```

**Color combinations:**
- Success/Ready: `bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20`
- Warning/Attention: `bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20`
- Error/Bug: `bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20`

**For Button Components:**

Select appropriate variant:
- `default`: Primary actions
- `destructive`: Delete, remove, dangerous actions
- `outline`: Secondary actions, filters
- `secondary`: Less prominent actions
- `ghost`: Icon buttons, subtle actions
- `link`: Navigation links
- `brand`: Special CTAs with Seer purple

```tsx
<Button variant="brand" size="lg">Get Started</Button>
<Button variant="destructive">Delete Workflow</Button>
```

**For Input Components:**

Maintain consistent styling:

```tsx
<Input
  type="text"
  placeholder="Enter workflow name"
  className="h-10 px-3 py-2 border border-input focus-visible:ring-2"
/>
```

### 3. Use CVA Pattern for Variants

All UI components must use Class Variance Authority:

```tsx
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const componentVariants = cva(
  "base-classes",
  {
    variants: {
      variant: {
        default: "variant-specific",
        secondary: "variant-specific",
      },
      size: {
        default: "h-10 px-4",
        sm: "h-9 px-3",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);
```

### 4. Ensure TypeScript Exports

Always export:
- Component (with `React.forwardRef`)
- Props interface
- Variants (CVA)
- Set `displayName`

```tsx
export interface ComponentProps
  extends React.HTMLAttributes<HTMLElement>,
    VariantProps<typeof componentVariants> {
  asChild?: boolean;
}

const Component = React.forwardRef<HTMLElement, ComponentProps>(
  ({ className, variant, ...props }, ref) => {
    return (
      <element
        className={cn(componentVariants({ variant, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Component.displayName = "Component";

export { Component, componentVariants };
```

### 5. Support Dark Mode

All components MUST support dark mode:

```tsx
className="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
```

Test in both light and dark modes before finalizing.

## Badge Pattern Enforcement

### Preferred Badge Pattern

**Reference:** `src/components/workflows/build/CategoryBadge.tsx` and `src/components/workflows/blocks/ToolBlockNode.tsx`

**Characteristics:**
- No icons (minimal design)
- Subtle background transparency: `bg-[color]-500/10`
- Matching text and border: `text-[color]-600 border-[color]-500/20`
- Small text: `text-[9px]` or `text-[10px]`
- Compact padding: `px-1` or `px-1.5`
- Fixed heights: `h-4` or `h-5`

### Status Badge Pattern (Use Sparingly)

**Reference:** `src/components/workflows/blocks/ToolBlockNode.tsx:126-151`

Only use icons for:
- Connection status
- Authentication state
- Critical errors or warnings

## Color Usage

Reference: `src/index.css`

**Seer custom colors:**
- `seer`: Purple brand (`239 84% 67%`)
- `success`: Emerald green (`160 84% 39%`)
- `warning`: Amber orange (`38 92% 50%`)
- `bug`: Red for bugs/errors (`347 77% 50%`)
- `destructive`: Red for dangerous actions (`0 84% 60%`)

**When to use:**
- Success states → `emerald-500/600` or `success`
- Warnings → `amber-500/600` or `warning`
- Errors → `bug` or `destructive`
- Brand elements → `seer`

## Examples

### Good Badge Usage

```tsx
// Simple label badge (preferred)
<Badge variant="outline" className="h-5 px-1.5 text-[10px]">
  {count}
</Badge>

// Success state badge (preferred)
<Badge className="h-4 px-1 text-[9px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20">
  Ready
</Badge>

// Warning state badge (preferred)
<Badge className="h-4 px-1 text-[9px] bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20">
  Needs Auth
</Badge>
```

### Bad Badge Usage

```tsx
// Too many icons for simple labels (avoid)
<Badge>
  <Icon /> Label <Icon />
</Badge>

// Inconsistent transparency (avoid)
<Badge className="bg-emerald-500/50 text-emerald-300 border-emerald-400">
  Ready
</Badge>

// Missing dark mode support (avoid)
<Badge className="bg-emerald-500/10 text-emerald-600">
  Ready
</Badge>
```

### Button Variant Selection

```tsx
// Primary action
<Button variant="default">Save Workflow</Button>

// Dangerous action
<Button variant="destructive">Delete</Button>

// Secondary action
<Button variant="outline">Cancel</Button>

// Brand CTA
<Button variant="brand">Get Started</Button>

// Subtle action
<Button variant="ghost" size="icon">
  <MoreVertical />
</Button>
```

### Input Consistency

```tsx
// Standard input
<Input
  type="email"
  placeholder="Enter your email"
  className="h-10 px-3 py-2"
/>

// With error state
<Input
  type="text"
  className="border-destructive focus-visible:ring-destructive"
/>
```

## Checklist

Before completing work on UI components:

- [ ] Referenced existing component in `src/components/ui/`
- [ ] Used CVA for variants
- [ ] Applied preferred badge pattern (minimal, no icons unless status)
- [ ] Used correct color combinations (`bg-[color]-500/10 text-[color]-600 border-[color]-500/20`)
- [ ] Included dark mode support with `dark:` variants
- [ ] Exported component, Props type, and variants
- [ ] Used `React.forwardRef` and set `displayName`
- [ ] Merged classes with `cn()` utility
- [ ] Tested in both light and dark modes
- [ ] Ensured accessibility (keyboard navigation, focus states)

## Additional Resources

- **Component patterns:** `.claude/rules/component-design.md`
- **Color system:** `.claude/rules/color-theming.md`
- **TypeScript patterns:** `.claude/rules/typescript-patterns.md`
- **Design system:** `src/index.css`
