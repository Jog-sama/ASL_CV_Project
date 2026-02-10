import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useState } from 'react'

function NavBar() {
  const location = useLocation()
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const isActive = (path) => location.pathname === path

  return (
    <motion.nav 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md shadow-sm"
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="text-2xl font-bold text-stone-800 hover:text-stone-600 transition-colors">
            VisibleVoices
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            <Link
              to="/"
              className={`text-lg transition-colors ${
                isActive('/') 
                  ? 'text-stone-800 font-semibold' 
                  : 'text-stone-600 hover:text-stone-800'
              }`}
            >
              Learn
            </Link>
            <Link
              to="/practice"
              className={`text-lg transition-colors ${
                isActive('/practice') 
                  ? 'text-stone-800 font-semibold' 
                  : 'text-stone-600 hover:text-stone-800'
              }`}
            >
              Practice
            </Link>
            <Link
              to="/practice"
              className="px-6 py-2 bg-[#E8D5D3] text-stone-800 rounded-full hover:bg-[#d9c5c3] transition-colors font-medium"
            >
              Start Practicing
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="md:hidden p-2 text-stone-600 hover:text-stone-800"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {isMenuOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="md:hidden mt-4 pb-4 flex flex-col gap-4"
          >
            <Link
              to="/"
              onClick={() => setIsMenuOpen(false)}
              className={`text-lg transition-colors ${
                isActive('/') 
                  ? 'text-stone-800 font-semibold' 
                  : 'text-stone-600'
              }`}
            >
              Learn
            </Link>
            <Link
              to="/practice"
              onClick={() => setIsMenuOpen(false)}
              className={`text-lg transition-colors ${
                isActive('/practice') 
                  ? 'text-stone-800 font-semibold' 
                  : 'text-stone-600'
              }`}
            >
              Practice
            </Link>
            <Link
              to="/practice"
              onClick={() => setIsMenuOpen(false)}
              className="px-6 py-2 bg-[#E8D5D3] text-stone-800 rounded-full hover:bg-[#d9c5c3] transition-colors font-medium text-center"
            >
              Start Practicing
            </Link>
          </motion.div>
        )}
      </div>
    </motion.nav>
  )
}

export default NavBar
