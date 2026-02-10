import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import NavBar from '../components/NavBar'

function NotFound() {
  return (
    <div className="min-h-screen bg-[#F5F1ED]">
      <NavBar />
      
      <div className="flex items-center justify-center min-h-screen px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <div className="text-9xl mb-6">🤷</div>
          <h1 className="text-6xl font-bold text-stone-800 mb-4">404</h1>
          <p className="text-2xl text-stone-600 mb-8 font-light">
            Oops! This page doesn't exist.
          </p>
          <Link
            to="/"
            className="inline-block px-8 py-4 bg-[#E8D5D3] text-stone-800 rounded-full hover:bg-[#d9c5c3] transition-all font-semibold shadow-md hover:shadow-lg transform hover:scale-105"
          >
            ← Back to Home
          </Link>
        </motion.div>
      </div>
    </div>
  )
}

export default NotFound
