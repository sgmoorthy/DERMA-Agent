/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        auraBlack: '#000000',
        auraCyan: '#00FFD1'
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Orbitron', 'system-ui'],
        mono: ['Kode Mono', 'monospace']
      },
      boxShadow: {
        glow: '0 0 24px rgba(0, 255, 209, 0.45)'
      }
    }
  },
  plugins: []
};
