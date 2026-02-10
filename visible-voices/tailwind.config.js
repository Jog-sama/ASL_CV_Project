/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        beige: {
          DEFAULT: '#F5F1ED',
          light: '#FAF8F6',
        },
        mauve: {
          DEFAULT: '#E8D5D3',
          light: '#F0E3E1',
          dark: '#d9c5c3',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
