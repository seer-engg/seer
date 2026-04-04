import React from 'react';
import whyDidYouRender from '@welldone-software/why-did-you-render';

if (import.meta.env.DEV) {
  whyDidYouRender(React, {
    // Track all components, not just React.memo ones
    trackAllPureComponents: true,
    trackHooks: true,
    logOnDifferentValues: true,

    // Include workflow-related components by default
    include: [/^Workflow/, /^Block/, /^Canvas/, /^Node/],

    // Exclude noisy library internals
    exclude: [/^Radix/, /^Motion/, /^Presence/, /^Portal/, /^Clerk/],

    // Collapse logs by default for cleaner console
    collapseGroups: true,
  });

  console.log(
    '%c[WDYR] Why Did You Render initialized',
    'background: #6366f1; color: white; padding: 2px 6px; border-radius: 3px;'
  );
  console.log(
    'Tracking: Workflow*, Block*, Canvas*, Node* components.\n' +
    'To track a specific component, add: MyComponent.whyDidYouRender = true'
  );
}
