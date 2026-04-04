---
paths: src/**/*.{tsx,css,ts}
---

# Color and Theming Guidelines

These rules apply to all files using colors, theming, and styling. Reference the complete design system in `src/index.css`.

## Color Palette

### Seer Custom Colors

Reference: `src/index.css:10-121`

**Light Mode (`:root`):**

| Color | HSL Value | Usage |
|-------|-----------|-------|
| `seer` | `239 84% 67%` | Purple brand color for primary branding, CTAs, special features |
| `bug` | `347 77% 50%` | Red for errors, bugs, and critical issues |
| `success` | `160 84% 39%` | Emerald green for success states, ready indicators, completed actions |
| `warning` | `38 92% 50%` | Amber orange for warnings, needs attention, pending states |
| `destructive` | `0 84% 60%` | Red for dangerous actions (delete, remove, cancel) |

**Dark Mode (`.dark`):**

All colors maintain the same HSL values in dark mode but are used against darker backgrounds for proper contrast.

### Standard Colors

| Color | Usage |
|-------|-------|
| `primary` | Main text and primary UI elements |
| `secondary` | Secondary text and less prominent elements |
| `muted` | Muted text and backgrounds |
| `accent` | Accent highlights and hover states |
| `border` | Borders and dividers |
| `input` | Form input borders |
| `ring` | Focus ring color |

## When to Use Each Color

### Brand Color (`seer`)

Use for:
- Primary CTAs and marketing buttons (`<Button variant="brand">`)
- Brand logos and headers
- Special feature highlights
- Gradient text effects (`text-gradient-seer`)

```tsx
<Button variant="brand">Get Started</Button>
<h1 className="text-gradient-seer">Welcome to Seer</h1>
```

### Success Color (`success` or `emerald`)

Use for:
- Ready states and successful connections
- Completed workflows or tasks
- Positive confirmations
- Active/online status indicators

```tsx
<Badge className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20">
  Ready
</Badge>
```

### Warning Color (`warning` or `amber`)

Use for:
- Needs authentication or attention
- Pending states
- Non-critical warnings
- Configuration required

```tsx
<Badge className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20">
  Needs Auth
</Badge>
```

### Error/Bug Color (`bug` or `destructive`)

Use for:
- Error messages and failed states
- Bug indicators
- Validation errors
- Failed connections

```tsx
<Badge className="bg-bug/10 text-bug dark:text-bug border-bug/20">
  Error
</Badge>
```

### Destructive Color (`destructive`)

Use for:
- Delete buttons
- Remove/cancel actions
- Permanent destructive operations

```tsx
<Button variant="destructive">Delete Workflow</Button>
```

## Transparency Conventions

### Standard Transparency Levels

- **Backgrounds:** `/10` (10% opacity)
- **Borders:** `/20` (20% opacity)
- **Hover states:** `/80` (80% opacity)
- **Active states:** `/90` (90% opacity)

**Examples:**

```tsx
// Background with 10% opacity
className="bg-emerald-500/10"

// Border with 20% opacity
className="border-emerald-500/20"

// Hover with 80% opacity
className="hover:bg-primary/80"

// Active with 90% opacity
className="bg-primary hover:bg-primary/90"
```

### Color Combination Pattern

For badges, pills, and status indicators:

```tsx
// Pattern: bg-[color]-500/10 text-[color]-600 border-[color]-500/20

// Emerald (success)
className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"

// Amber (warning)
className="bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"

// Red (error)
className="bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20"
```

## Custom Utilities

Reference: `src/index.css:138-194`

### Gradient Text

**Utility:** `text-gradient-seer`

Creates a purple gradient text effect for brand elements.

```tsx
<h1 className="text-gradient-seer">Seer AI Platform</h1>
```

### Glow Effects

**Available utilities:**
- `glow-seer`: Purple glow (20px blur, 0.3 opacity)
- `glow-success`: Emerald glow (20px blur, 0.3 opacity)
- `glow-bug`: Red glow (20px blur, 0.3 opacity)

**Usage:**

```tsx
// Brand element with purple glow
<div className="bg-seer text-white rounded-lg p-4 glow-seer">
  Featured Workflow
</div>

// Success indicator with green glow
<div className="bg-success text-white rounded-full p-2 glow-success">
  <CheckIcon />
</div>
```

### Scrollbar Styling

**Utility:** `scrollbar-thin`

Creates thin, styled scrollbars matching the theme.

```tsx
<div className="h-96 overflow-y-auto scrollbar-thin">
  {/* Long content */}
</div>
```

### Inner Shadows

**Available utilities:**
- `shadow-inner-right`: Subtle shadow from the right (for depth)
- `shadow-inner-left`: Subtle shadow from the left (for depth)

**Usage:**

```tsx
<div className="shadow-inner-right">
  {/* Content with subtle depth */}
</div>
```

## Dark Mode Support

### Requirements

ALL components MUST support both light and dark modes.

### Implementation

Use Tailwind's `dark:` prefix for dark mode variants:

```tsx
// Text colors
className="text-gray-900 dark:text-gray-100"

// Backgrounds
className="bg-white dark:bg-gray-900"

// Borders
className="border-gray-200 dark:border-gray-800"

// Complete badge example
className="bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
```

### CSS Variable Usage

Use CSS variables from `src/index.css` for theme-aware colors:

```tsx
// Using theme colors
className="bg-background text-foreground"
className="border-border"
className="bg-card text-card-foreground"

// Using semantic colors
className="bg-primary text-primary-foreground"
className="bg-secondary text-secondary-foreground"
className="bg-muted text-muted-foreground"
```

### Testing

Before finalizing any UI changes:

1. Test in light mode (default)
2. Switch to dark mode and verify appearance
3. Check color contrast for accessibility
4. Ensure focus states are visible in both modes

## Animation Utilities

Reference: `src/index.css:155-166`

**Available animations:**
- `animate-pulse-slow`: Slower pulse animation (3s)
- `animate-typing`: Typing indicator animation (1.5s)

**Usage:**

```tsx
<div className="animate-pulse-slow">Loading...</div>
<span className="animate-typing">...</span>
```

## Accessibility Considerations

### Color Contrast

- Ensure sufficient contrast ratios (WCAG AA: 4.5:1 for text)
- Test with browser DevTools contrast checker
- Don't rely on color alone to convey information

### Alternative Indicators

When using colors to indicate state:
- Add icons for visual distinction
- Include text labels for clarity
- Use patterns or shapes in addition to colors

**Example:**

```tsx
// Good: Color + Icon + Text
<Badge className="bg-emerald-500/10 text-emerald-600">
  <CheckCircle2 className="w-3 h-3" />
  Ready
</Badge>

// Bad: Color only
<Badge className="bg-emerald-500/10 text-emerald-600" />
```

## Summary Checklist

When working with colors and theming:

- [ ] Use semantic color names (`seer`, `success`, `warning`, `bug`, `destructive`)
- [ ] Apply correct transparency levels (`/10` for backgrounds, `/20` for borders)
- [ ] Support dark mode with `dark:` variants
- [ ] Test in both light and dark modes
- [ ] Use CSS variables for theme-aware colors
- [ ] Ensure proper contrast ratios
- [ ] Include non-color indicators (icons, text) for important states
- [ ] Apply custom utilities (`text-gradient-seer`, `glow-*`) where appropriate
- [ ] Reference `src/index.css` for complete color palette
