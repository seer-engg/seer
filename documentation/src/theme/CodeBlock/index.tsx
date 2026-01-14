import React, { useEffect } from 'react';
import CodeBlockOriginal from '@theme-original/CodeBlock';
import { trackCodeCopy } from '@site/src/utils/analytics';
import type { Props } from '@theme/CodeBlock';

export default function CodeBlock(props: Props): JSX.Element {
  useEffect(() => {
    // Add tracking to copy buttons after component renders
    const copyButtons = document.querySelectorAll(
      '[class*="copyButton"], [aria-label*="Copy"]'
    );

    const handleCopyClick = (event: Event) => {
      const button = event.target as HTMLElement;
      const codeBlock = button.closest('[class*="codeBlockContent"]');

      if (codeBlock) {
        const language = button.closest('[class*="codeBlock"]')?.getAttribute('data-language') || 'unknown';
        const codeText = codeBlock.textContent || '';
        const preview = codeText.substring(0, 50); // First 50 chars for preview

        trackCodeCopy(language, preview);
      }
    };

    copyButtons.forEach((button) => {
      button.addEventListener('click', handleCopyClick);
    });

    return () => {
      copyButtons.forEach((button) => {
        button.removeEventListener('click', handleCopyClick);
      });
    };
  }, []);

  return <CodeBlockOriginal {...props} />;
}
