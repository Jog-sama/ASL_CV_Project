import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import LearnPage from './pages/LearnPage'
import PracticePage from './pages/PracticePage'
import NotFound from './pages/NotFound'
import './App.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LearnPage />} />
        <Route path="/practice" element={<PracticePage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  )
}

export default App
