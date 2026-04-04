---
paths: src/components/**/*.tsx
---

# Component Design Guidelines

These rules apply to all React components in the `src/components/` directory. Follow these patterns to ensure consistency across the Seer UI.

## Badge/Pill Components

### Preferred Pattern (Minimal Design)

Use this pattern for labels, counts, and non-critical indicators:

**Styling characteristics:**
- No icons (keep it minimal)
- Subtle background with transparency: `bg-[color]-500/10`
- Matching text and border: `text-[color]-600 border-[color]-500/20`
- Small text: `text-[9px]` or `text-[10px]`
- Compact padding: `px-1` or `px-1.5`
- Fixed heights: `h-4` or `h-5`
- Use `variant="outline"` or `variant="secondary"`

**Reference:** `src/components/workflows/build/CategoryBadge.tsx:10-34`

**Examples:**

```tsx
// Success/Ready state
<Badge
  variant="secondary"
  className="h-4 px-1 text-[9px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
>
  Ready
</Badge>

// Warning/Needs attention
<Badge
  variant="secondary"
  className="h-4 px-1 text-[9px] bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
>
  Needs Auth
</Badge>

// Count badge
<Badge variant="outline" className="h-5 px-1.5 text-[10px] ml-auto">
  {count}
</Badge>
```

### Status Badge Pattern (Use Sparingly)

Only use icon-based badges for critical status indicators:

**When to use:**
- Connection status
- Authentication state
- Critical errors or warnings

**Reference:** `src/components/workflows/blocks/ToolBlockNode.tsx:126-151`

**Example:**

```tsx
// Only for important status with icons
<Badge
  variant="secondary"
  className="flex items-center gap-1 text-[10px] px-1.5 py-0 h-5 bg-emerald-500/10 text-emerald-600 border-emerald-500/20"
>
  <CheckCircle2 className="w-3 h-3" />
  Connected
</Badge>
```

### Color Coding for Badges

- **Success/Ready:** `emerald-500/10`, `emerald-600`, `emerald-500/20`
- **Warning/Attention:** `amber-500/10`, `amber-600`, `amber-500/20`
- **Error/Bug:** Use `bug` or `destructive` color
- **Neutral:** `secondary` variant without custom colors

## Button Components

### Available Variants

Reference: `src/components/ui/button.tsx`

**Variants:**
- `default`: Primary action button (dark background)
- `destructive`: Dangerous actions (delete, remove)
- `outline`: Secondary action with border
- `secondary`: Less prominent action (light gray)
- `ghost`: Minimal button without background
- `link`: Text link with underline
- `brand`: Seer purple brand color for special CTAs

**Sizes:**
- `default`: `h-10 px-4 py-2` (standard)
- `sm`: `h-9 px-3` (compact)
- `lg`: `h-11 px-8` (prominent)
- `icon`: `h-10 w-10` (icon-only)

### Button Composition Pattern

Always use `asChild` when wrapping other components:

```tsx
// Wrapping a Link component
<Button asChild>
  <Link to="/workflows">View Workflows</Link>
</Button>

// Wrapping custom components
<Button asChild variant="brand">
  <CustomLink href="/signup">Get Started</CustomLink>
</Button>
```

### When to Use Each Variant

- **default:** Primary actions (submit, save, create)
- **destructive:** Delete, remove, cancel subscription
- **outline:** Secondary actions, filters, toggles
- **secondary:** Tertiary actions, less important buttons
- **ghost:** Icon buttons, menu items, subtle actions
- **link:** Navigation, inline links
- **brand:** Marketing CTAs, special promotions

## Form Input Components

Reference: `src/components/ui/input.tsx`

### Standard Input Pattern

All inputs MUST follow this pattern:

**Required styling:**
- Height: `h-10`
- Padding: `px-3 py-2`
- Border: `border border-input`
- Focus ring: `focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2`
- Responsive text: `text-base md:text-sm`
- Disabled state: `disabled:cursor-not-allowed disabled:opacity-50`

**Example:**

```tsx
<Input
  type="text"
  placeholder="Enter workflow name"
  className="additional-custom-classes"
/>
```

### Input States

- **Default:** Standard border and background
- **Focus:** Ring offset with theme ring color
- **Disabled:** Reduced opacity (50%), cursor not-allowed
- **Error:** Add `border-destructive` and `focus-visible:ring-destructive`

## Component Structure Pattern

All UI components MUST follow this structure:

### 1. Use Class Variance Authority (CVA)

```tsx
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const componentVariants = cva(
  "base classes here", // Base classes applied to all variants
  {
    variants: {
      variant: {
        default: "variant-specific classes",
        secondary: "variant-specific classes",
      },
      size: {
        default: "size-specific classes",
        sm: "size-specific classes",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);
```

### 2. Define Props Interface

```tsx
export interface ComponentProps
  extends React.HTMLAttributes<HTMLElement>,
    VariantProps<typeof componentVariants> {
  asChild?: boolean;
}
```

### 3. Use React.forwardRef

```tsx
const Component = React.forwardRef<HTMLElement, ComponentProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "element";
    return (
      <Comp
        className={cn(componentVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
```

### 4. Set displayName

```tsx
Component.displayName = "Component";
```

### 5. Export Pattern

```tsx
export { Component, componentVariants };
```

## Dark Mode Support

ALL components MUST support both light and dark modes:

- Use Tailwind's `dark:` prefix for dark mode variants
- Reference CSS variables from `src/index.css` (`:root` and `.dark`)
- Test in both modes before finalizing

**Example:**

```tsx
<div className="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
  <Badge className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
    Ready
  </Badge>
</div>
```

## Accessibility Requirements

- All interactive elements must be keyboard accessible
- Use proper ARIA attributes when needed
- Focus states must be visible (`focus-visible:ring-2`)
- Disabled states must be properly indicated
- Color alone should not convey information (use icons/text too)

## Summary Checklist

When creating or modifying UI components:

- [ ] Use CVA for variants
- [ ] Export component, Props type, and variants
- [ ] Use `React.forwardRef` for ref forwarding
- [ ] Set `displayName` for debugging
- [ ] Use `cn()` utility for class merging
- [ ] Support dark mode with `dark:` variants
- [ ] Test accessibility (keyboard navigation, focus states)
- [ ] Follow badge pattern (minimal, no icons unless status)
- [ ] Use appropriate button variant and size
- [ ] Maintain consistent input styling
