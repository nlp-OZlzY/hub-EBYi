/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark': {
          900: '#0f0f1a',
          800: '#1a1a2e',
          700: '#252542',
        },
        'primary': {
          500: '#6366f1',
          600: '#4f46e5',
        },
        'secondary': {
          500: '#8b5cf6',
        }
      },
      fontFamily: {
        'inter': ['Inter', 'sans-serif'],
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}
