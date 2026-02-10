const BackgroundArt = () => {
  return (
    <svg
      className="fixed inset-0 w-full h-full opacity-50 pointer-events-none" // Changed from opacity-5 to opacity-15
      viewBox="0 0 1000 1000"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Hand gesture flowing into text */}
      <g className="animate-pulse" style={{ animationDuration: '4s' }}>
        {/* Hand outline - made thicker and bigger */}
        <path
          d="M 200 400 Q 220 380 240 400 L 250 450 Q 255 470 240 480 Q 225 490 210 480 L 200 450 Z"
          fill="none"
          stroke="#E8D5D3"
          strokeWidth="4" // Changed from 2 to 4
        />
        <path
          d="M 240 400 Q 250 370 260 400 L 265 460 Q 268 480 255 490 Q 242 500 235 485 L 240 450 Z"
          fill="none"
          stroke="#E8D5D3"
          strokeWidth="4"
        />
        <path
          d="M 260 400 Q 270 365 280 395 L 285 465 Q 288 485 275 495 Q 262 505 255 490 L 260 455 Z"
          fill="none"
          stroke="#E8D5D3"
          strokeWidth="4"
        />
        
        {/* Flowing particles from hand to text - bigger */}
        <circle cx="300" cy="420" r="6" fill="#E8D5D3" opacity="0.8" />
        <circle cx="350" cy="415" r="5" fill="#E8D5D3" opacity="0.7" />
        <circle cx="400" cy="410" r="4" fill="#E8D5D3" opacity="0.6" />
        <circle cx="450" cy="408" r="4" fill="#E8D5D3" opacity="0.5" />
        <circle cx="500" cy="405" r="3" fill="#E8D5D3" opacity="0.4" />
        
        {/* Text letters forming - bigger and more visible */}
        <text x="550" y="420" fontFamily="sans-serif" fontSize="60" fill="#E8D5D3" opacity="0.5">
          A
        </text>
        <text x="610" y="420" fontFamily="sans-serif" fontSize="60" fill="#E8D5D3" opacity="0.4">
          B
        </text>
        <text x="670" y="420" fontFamily="sans-serif" fontSize="60" fill="#E8D5D3" opacity="0.3">
          C
        </text>
      </g>
      
      {/* Additional decorative hand gestures scattered */}
      <g opacity="0.2"> // Changed from 0.08 to 0.2
        <path
          d="M 700 200 Q 710 190 720 200 L 725 230 Q 728 240 720 245 Q 712 250 705 245 L 700 230 Z"
          fill="none"
          stroke="#E8D5D3"
          strokeWidth="3"
        />
        <path
          d="M 150 700 Q 160 690 170 700 L 175 730 Q 178 740 170 745 Q 162 750 155 745 L 150 730 Z"
          fill="none"
          stroke="#E8D5D3"
          strokeWidth="3"
        />
      </g>
    </svg>
  )
}

export default BackgroundArt