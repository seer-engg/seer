import { Node, mergeAttributes } from '@tiptap/core';

/**
 * Custom TipTap inline atom node for template variables.
 *
 * Variables like ${trigger-1.data.email_addresses[0].email} are stored as
 * atomic nodes instead of plain text. This prevents the markdown serializer
 * from escaping special characters (e.g., [ ] in array accessors).
 *
 * - atom: true — editor treats the node as a single indivisible unit
 * - Markdown serialization uses state.write() which bypasses esc()
 * - HTML serialization uses a <span> with data attributes for parseHTML round-trip
 */
export const TemplateVariable = Node.create({
  name: 'templateVariable',
  group: 'inline',
  inline: true,
  atom: true,
  selectable: true,
  draggable: false,

  addAttributes() {
    return {
      path: { default: '' },
    };
  },

  parseHTML() {
    return [
      {
        tag: 'span[data-template-var]',
        getAttrs: (el) => ({
          path: (el as HTMLElement).getAttribute('data-path') ?? '',
        }),
      },
    ];
  },

  renderHTML({ node, HTMLAttributes }) {
    return [
      'span',
      mergeAttributes(HTMLAttributes, {
        'data-template-var': '',
        'data-path': node.attrs.path,
        class: 'template-variable-chip',
      }),
      `\${${node.attrs.path}}`,
    ];
  },

  renderText({ node }) {
    return `\${${node.attrs.path}}`;
  },

  addStorage() {
    return {
      markdown: {
        serialize(state: { write: (text: string) => void }, node: { attrs: { path: string } }) {
          // state.write() outputs raw text — bypasses prosemirror-markdown's esc()
          state.write(`\${${node.attrs.path}}`);
        },
        parse: {
          // Parsing is handled by toEditorContent() which converts {{...}} text
          // to <span data-template-var> HTML before feeding to TipTap
        },
      },
    };
  },
});
