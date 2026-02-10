import { motion } from 'framer-motion'

function BackgroundASL() {
  // Translucent ASL hand signs scattered in background
  const signs = [
    { letter: '👌', size: 'text-9xl', top: '10%', left: '5%', rotation: -15 },
    { letter: '✋', size: 'text-8xl', top: '20%', right: '10%', rotation: 20 },
    { letter: '👍', size: 'text-7xl', top: '60%', left: '15%', rotation: -25 },
    { letter: '✌️', size: 'text-9xl', bottom: '15%', right: '8%', rotation: 15 },
    { letter: '🤙', size: 'text-8xl', top: '45%', right: '20%', rotation: -10 },
    { letter: '🤞', size: 'text-7xl', bottom: '40%', left: '8%', rotation: 25 },
    { letter: '👊', size: 'text-6xl', top: '75%', right: '25%', rotation: -20 },
    { letter: '🤘', size: 'text-8xl', top: '35%', left: '25%', rotation: 10 },
  ]

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {signs.map((sign, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ 
            opacity: 0.08, 
            scale: 1,
            y: [0, -20, 0],
          }}
          transition={{
            opacity: { duration: 1, delay: index * 0.1 },
            scale: { duration: 1, delay: index * 0.1 },
            y: {
              duration: 8 + index * 0.5,
              repeat: Infinity,
              ease: "easeInOut"
            }
          }}
          className={`absolute ${sign.size} select-none`}
          style={{
            top: sign.top,
            left: sign.left,
            right: sign.right,
            bottom: sign.bottom,
            transform: `rotate(${sign.rotation}deg)`,
            filter: 'grayscale(100%)',
          }}
        >
          {sign.letter}
        </motion.div>
      ))}
    </div>
  )
}

export default BackgroundASL
