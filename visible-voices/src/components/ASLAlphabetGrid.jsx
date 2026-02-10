import { motion } from 'framer-motion'

function ASLAlphabetGrid() {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
  
  // Placeholder emojis for ASL signs (in production, use actual ASL hand images)
  const signEmojis = {
    A: '✊', B: '🖐️', C: '🤏', D: '☝️', E: '✊', F: '👌',
    G: '👈', H: '✌️', I: '🤙', J: '🤙', K: '✌️', L: '👍',
    M: '✊', N: '✊', O: '👌', P: '👇', Q: '👇', R: '✌️',
    S: '✊', T: '✊', U: '✌️', V: '✌️', W: '🖖', X: '☝️',
    Y: '🤙', Z: '☝️'
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 w-full max-w-6xl mx-auto">
      {alphabet.map((letter, index) => (
        <motion.div
          key={letter}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: index * 0.03 }}
          whileHover={{ y: -8, scale: 1.05 }}
          className="bg-white rounded-2xl p-6 shadow-md hover:shadow-xl transition-all cursor-pointer flex flex-col items-center gap-3"
        >
          <span className="text-5xl">{signEmojis[letter]}</span>
          <span className="text-3xl font-bold text-stone-800">{letter}</span>
          <span className="text-xs text-stone-500 uppercase tracking-wide">
            Sign for {letter}
          </span>
        </motion.div>
      ))}
    </div>
  )
}

export default ASLAlphabetGrid
