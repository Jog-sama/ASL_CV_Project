import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import NavBar from '../components/NavBar'
import ASLAlphabetGrid from '../components/ASLAlphabetGrid'

function LearnPage() {
  const benefits = [
    {
      icon: '🤝',
      title: 'Connect with 500k+ ASL Users',
      description: 'Join a vibrant community and build meaningful connections with deaf and hard-of-hearing individuals.'
    },
    {
      icon: '🌍',
      title: 'Build Inclusive Communities',
      description: 'Break down communication barriers and create spaces where everyone can participate fully.'
    },
    {
      icon: '💼',
      title: 'Expand Career Opportunities',
      description: 'ASL skills are increasingly valued in healthcare, education, customer service, and many other fields.'
    }
  ]

  const steps = [
    {
      number: '1',
      title: 'Learn the Signs',
      description: 'Study the ASL alphabet and basic signs using our visual guides.',
      icon: '📚'
    },
    {
      number: '2',
      title: 'Practice with Your Camera',
      description: 'Use our AI-powered tool to practice signs and get real-time feedback.',
      icon: '📹'
    },
    {
      number: '3',
      title: 'Get Instant Feedback',
      description: 'Receive immediate accuracy scores and tips to improve your signing.',
      icon: '✨'
    }
  ]

  return (
    <div className="min-h-screen bg-[#F5F1ED]">
      <NavBar />
      
      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-5xl mx-auto text-center">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl md:text-7xl font-bold text-stone-800 mb-6 leading-tight"
          >
            Learn Sign Language,
            <br />
            <span className="text-[#9B7E7A]">Amplify Every Voice</span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl md:text-2xl text-stone-600 mb-10 max-w-3xl mx-auto font-light"
          >
            Master American Sign Language through interactive practice and AI-powered feedback. 
            Make communication accessible for everyone.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            <Link
              to="/practice"
              className="px-8 py-4 bg-[#E8D5D3] text-stone-800 rounded-full hover:bg-[#d9c5c3] transition-all text-lg font-semibold shadow-md hover:shadow-lg transform hover:scale-105"
            >
              Start Practicing Now
            </Link>
            <a
              href="#alphabet"
              className="px-8 py-4 bg-white text-stone-800 rounded-full hover:bg-stone-50 transition-all text-lg font-semibold shadow-md"
            >
              View ASL Alphabet
            </a>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="mt-16 text-8xl"
          >
            👋
          </motion.div>
        </div>
      </section>

      {/* Why Learn ASL Section */}
      <section className="py-20 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl font-bold text-stone-800 text-center mb-4"
          >
            Why Learn ASL?
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-xl text-stone-600 text-center mb-16 font-light"
          >
            Sign language opens doors to connection, understanding, and opportunity
          </motion.p>

          <div className="grid md:grid-cols-3 gap-8">
            {benefits.map((benefit, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -8 }}
                className="bg-[#F5F1ED] rounded-3xl p-8 shadow-md hover:shadow-xl transition-all"
              >
                <div className="text-5xl mb-4">{benefit.icon}</div>
                <h3 className="text-2xl font-bold text-stone-800 mb-3">
                  {benefit.title}
                </h3>
                <p className="text-stone-600 leading-relaxed">
                  {benefit.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ASL Alphabet Section */}
      <section id="alphabet" className="py-20 px-6 bg-[#F5F1ED]">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl font-bold text-stone-800 text-center mb-4"
          >
            The ASL Alphabet
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-xl text-stone-600 text-center mb-16 font-light"
          >
            Learn the foundation of ASL communication with the manual alphabet
          </motion.p>

          <ASLAlphabetGrid />
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl font-bold text-stone-800 text-center mb-4"
          >
            How It Works
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-xl text-stone-600 text-center mb-16 font-light"
          >
            Three simple steps to master ASL
          </motion.p>

          <div className="grid md:grid-cols-3 gap-12 relative">
            {/* Connector lines */}
            <div className="hidden md:block absolute top-24 left-0 right-0 h-0.5 bg-gradient-to-r from-[#E8D5D3] via-[#E8D5D3] to-[#E8D5D3] opacity-30"></div>

            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.15 }}
                className="relative text-center"
              >
                <div className="bg-[#E8D5D3] rounded-full w-20 h-20 flex items-center justify-center text-3xl font-bold text-stone-800 mx-auto mb-6 shadow-lg relative z-10">
                  {step.number}
                </div>
                <div className="text-5xl mb-4">{step.icon}</div>
                <h3 className="text-2xl font-bold text-stone-800 mb-3">
                  {step.title}
                </h3>
                <p className="text-stone-600 leading-relaxed">
                  {step.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6 bg-gradient-to-br from-[#E8D5D3] to-[#F5F1ED]">
        <div className="max-w-4xl mx-auto text-center">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl font-bold text-stone-800 mb-6"
          >
            Ready to Start Practicing?
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-xl text-stone-600 mb-10 font-light"
          >
            Put your knowledge into action with our AI-powered practice tool
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            <Link
              to="/practice"
              className="inline-block px-10 py-5 bg-stone-800 text-white rounded-full hover:bg-stone-700 transition-all text-lg font-semibold shadow-xl hover:shadow-2xl transform hover:scale-105"
            >
              Practice Now →
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-stone-800 text-white py-12 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-2xl font-bold mb-2">VisibleVoices</p>
          <p className="text-stone-400 mb-6">
            Making sign language accessible through technology
          </p>
          <div className="flex justify-center gap-8">
            <Link to="/" className="text-stone-400 hover:text-white transition-colors">
              Learn
            </Link>
            <Link to="/practice" className="text-stone-400 hover:text-white transition-colors">
              Practice
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default LearnPage
