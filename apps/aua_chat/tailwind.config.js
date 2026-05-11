/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:       '#f7f6f2',
        paper:    '#ffffff',
        ink:      '#18181b',
        muted:    '#6b7280',
        line:     '#e4e1da',
        soft:     '#f0ede6',
        soft2:    '#ecfdf5',
        soft3:    '#fff7ed',
        accent:   '#4338ca',
        'accent-hover': '#3730a3',
        'accent-soft': '#eef2ff',
        'accent-text': '#4338ca',
        accent2:  '#0f766e',
        accent3:  '#c2410c',
      },
      fontFamily: {
        serif: ['"DM Serif Display"', 'Georgia', 'serif'],
        sans:  ['"DM Sans"', 'system-ui', 'sans-serif'],
        mono:  ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        card: '10px',
        tag:  '999px',
      },
    },
  },
  plugins: [],
}
