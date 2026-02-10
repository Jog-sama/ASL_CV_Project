import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import Webcam from 'react-webcam'
import NavBar from '../components/NavBar'
import BackgroundASL from '../components/BackgroundASL'

function PracticePage() {
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const webcamRef = useRef(null)

  // Replace with your actual HuggingFace Space endpoint
  const BACKEND_URL = 'https://mg643-asl-backend.hf.space/predict'

  const startCamera = () => {
    setIsCameraOn(true)
    setCapturedImage(null)
    setPrediction(null)
  }

  const captureImage = async () => {
    const imageSrc = webcamRef.current.getScreenshot()
    setCapturedImage(imageSrc)
    setIsCameraOn(false)
    setIsProcessing(true)
    setPrediction({ processing: true })

    try {
      // Convert base64 to blob
      const base64Data = imageSrc.split(',')[1]
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/jpeg' })

      // Create FormData
      const formData = new FormData()
      formData.append('file', blob, 'capture.jpg')

      // Send to backend
      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      setPrediction(result)
    } catch (error) {
      console.error('Error processing image:', error)
      setPrediction({ 
        error: true, 
        message: 'Could not process image. Please try again.' 
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const retake = () => {
    setCapturedImage(null)
    setPrediction(null)
    setIsCameraOn(true)
  }

  return (
    <div className="min-h-screen bg-[#F5F1ED] relative overflow-hidden">
      <BackgroundASL />
      <NavBar />
      
      <div className="pt-24 pb-12 px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-7xl mx-auto"
        >
          {/* Page Title */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-stone-800 mb-3">
              Practice Your Signs
            </h1>
            <p className="text-xl text-stone-600 font-light">
              Use your camera to practice ASL and get instant feedback
            </p>
          </div>

          {/* Two Panel Layout */}
          <div className="grid md:grid-cols-2 gap-8">
            {/* Left Panel - Camera Capture */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-8 flex flex-col items-center justify-center min-h-[500px]"
            >
              <h2 className="text-2xl font-semibold text-stone-800 mb-6">
                Show Your Sign
              </h2>

              {capturedImage ? (
                <div className="w-full flex flex-col gap-4">
                  <motion.img 
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    src={capturedImage} 
                    alt="Captured" 
                    className="rounded-2xl w-full shadow-lg"
                  />
                  <button
                    onClick={retake}
                    disabled={isProcessing}
                    className="px-8 py-3 bg-stone-300 text-stone-800 rounded-full hover:bg-stone-400 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-md"
                  >
                    Retake
                  </button>
                </div>
              ) : isCameraOn ? (
                <div className="w-full flex flex-col gap-4">
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="rounded-2xl overflow-hidden shadow-lg ring-4 ring-[#E8D5D3]/30"
                  >
                    <Webcam
                      ref={webcamRef}
                      audio={false}
                      className="w-full"
                      screenshotFormat="image/jpeg"
                    />
                  </motion.div>
                  <button
                    onClick={captureImage}
                    className="px-8 py-3 bg-[#E8D5D3] text-stone-800 rounded-full hover:bg-[#d9c5c3] transition-all font-medium shadow-md hover:shadow-lg transform hover:scale-105"
                  >
                    📸 Capture Image
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-6">
                  <div className="text-8xl mb-4">📹</div>
                  <button
                    onClick={startCamera}
                    className="px-10 py-4 bg-[#E8D5D3] text-stone-800 rounded-full hover:bg-[#d9c5c3] transition-all text-lg font-semibold shadow-lg hover:shadow-xl transform hover:scale-105"
                  >
                    Start Camera
                  </button>
                  <p className="text-stone-500 text-sm text-center max-w-xs">
                    We'll need access to your camera to help you practice
                  </p>
                </div>
              )}
            </motion.div>

            {/* Right Panel - Translation Results */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-3xl shadow-xl p-8 flex items-center justify-center min-h-[500px]"
            >
              <h2 className="absolute top-8 left-8 text-2xl font-semibold text-stone-800">
                Translation
              </h2>

              {prediction?.processing ? (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center gap-6"
                >
                  <div className="relative">
                    <div className="animate-spin rounded-full h-16 w-16 border-4 border-[#E8D5D3] border-t-transparent"></div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-2xl">🤖</div>
                    </div>
                  </div>
                  <p className="text-xl text-stone-600 font-light">
                    Analyzing your sign...
                  </p>
                </motion.div>
              ) : prediction?.error ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex flex-col items-center gap-4 text-center"
                >
                  <div className="text-6xl mb-2">⚠️</div>
                  <p className="text-xl text-red-500 font-light max-w-sm">
                    {prediction.message}
                  </p>
                </motion.div>
              ) : prediction ? (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex flex-col items-center gap-8 w-full px-4"
                >
                  {/* Main Prediction */}
                  <div className="flex flex-col items-center gap-3">
                    <p className="text-sm text-stone-500 uppercase tracking-widest font-medium">
                      Detected Sign
                    </p>
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 200, damping: 15 }}
                      className="text-9xl font-bold text-stone-800 bg-gradient-to-br from-stone-800 to-stone-600 bg-clip-text text-transparent"
                    >
                      {prediction.predicted_letter}
                    </motion.div>
                  </div>
                  
                  {/* Confidence */}
                  <div className="flex flex-col items-center gap-3 w-full max-w-sm">
                    <p className="text-sm text-stone-500 uppercase tracking-widest font-medium">
                      Confidence
                    </p>
                    <div className="w-full bg-stone-200 rounded-full h-4 overflow-hidden shadow-inner">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${(prediction.confidence * 100).toFixed(0)}%` }}
                        transition={{ duration: 1, ease: "easeOut" }}
                        className="bg-gradient-to-r from-[#E8D5D3] to-[#d9c5c3] h-full rounded-full shadow-sm"
                      ></motion.div>
                    </div>
                    <motion.p 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 }}
                      className="text-3xl font-bold text-stone-700"
                    >
                      {(prediction.confidence * 100).toFixed(1)}%
                    </motion.p>
                  </div>

                  {/* Encouraging Message */}
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                    className="text-center"
                  >
                    {prediction.confidence > 0.7 ? (
                      <p className="text-lg text-green-600 font-medium">
                        🎉 Great job! Your sign is clear!
                      </p>
                    ) : prediction.confidence > 0.4 ? (
                      <p className="text-lg text-amber-600 font-medium">
                        👍 Good effort! Try refining your hand position.
                      </p>
                    ) : (
                      <p className="text-lg text-stone-600 font-medium">
                        💪 Keep practicing! Check the Learn page for guidance.
                      </p>
                    )}
                  </motion.div>
                </motion.div>
              ) : (
                <div className="flex flex-col items-center gap-4 text-center">
                  <div className="text-7xl mb-2">👋</div>
                  <p className="text-xl text-stone-600 font-light max-w-sm">
                    Your translation will appear here...
                  </p>
                  <p className="text-sm text-stone-400">
                    Capture a sign to get started
                  </p>
                </div>
              )}
            </motion.div>
          </div>

          {/* Tips Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-12 bg-white/60 backdrop-blur-sm rounded-2xl p-8 max-w-4xl mx-auto"
          >
            <h3 className="text-xl font-semibold text-stone-800 mb-4 text-center">
              💡 Tips for Best Results
            </h3>
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div>
                <div className="text-3xl mb-2">💡</div>
                <p className="text-stone-600 text-sm">
                  Ensure good lighting for clear hand visibility
                </p>
              </div>
              <div>
                <div className="text-3xl mb-2">🖐️</div>
                <p className="text-stone-600 text-sm">
                  Position your hand in the center of the frame
                </p>
              </div>
              <div>
                <div className="text-3xl mb-2">📐</div>
                <p className="text-stone-600 text-sm">
                  Make sure only your hand is visible, avoid distractions
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}

export default PracticePage
