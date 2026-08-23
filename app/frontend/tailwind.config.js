/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: '#faf7f2',
        latte: '#ede8de',
        cappuccino: '#c9bba7',
        caramel: '#b07030',
        cinnamon: '#9a3412',
        roast: '#7a5030',
        espresso: '#1c0e06',
      },
    }
  },
  plugins: [],
}
