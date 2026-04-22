import React, { useRef, useEffect, useState } from 'react';

interface ECGWaveformProps {
  data: number[];
  leadName: string;
  color: string;
  width?: number;
  height?: number;
  animate?: boolean;
  speed?: number;
  amplitude?: number;
  highlighted?: boolean;
}

const ECGWaveform: React.FC<ECGWaveformProps> = ({
  data,
  leadName,
  color,
  width = 600,
  height = 120,
  animate = true,
  speed = 1,
  amplitude = 1,
  highlighted = false
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const scrollOffsetRef = useRef<number>(0);
  const displayDataRef = useRef<number[]>([]);
  const [redrawTrigger, setRedrawTrigger] = useState(0);
  const [isAnimatingPrev, setIsAnimatingPrev] = useState(animate);

  // Trigger redraw when animation state changes
  useEffect(() => {
    if (animate !== isAnimatingPrev) {
      setIsAnimatingPrev(animate);
      setRedrawTrigger(prev => prev + 1);
    }
  }, [animate, isAnimatingPrev]);

  // Reset display buffer when data changes (new patient selected)
  useEffect(() => {
    if (data.length > 0) {
      const bufferSize = width * 2;
      displayDataRef.current = new Array(bufferSize);
      // Fill buffer with the beginning of the data
      for (let i = 0; i < bufferSize; i++) {
        displayDataRef.current[i] = data[i % data.length] || 0;
      }
      scrollOffsetRef.current = 0;
      
      // Debug logging
      const sampleValues = displayDataRef.current.slice(0, 10);
      console.log(`${leadName}: Buffer filled with ${bufferSize} samples from ${data.length} source samples`);
      console.log(`${leadName}: Sample values:`, sampleValues);
      console.log(`${leadName}: Data range: ${Math.min(...data)} to ${Math.max(...data)}`);
      
      // Trigger a redraw
      setRedrawTrigger(prev => prev + 1);
    }
  }, [data, width, leadName]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution
    canvas.width = width;
    canvas.height = height;

    const draw = () => {
      // Clear canvas with pink ECG paper background
      ctx.fillStyle = '#ffe6e6';
      ctx.fillRect(0, 0, width, height);

      // Draw scrolling grid (medical ECG paper style)
      ctx.strokeStyle = '#ffb3b3';
      ctx.lineWidth = 0.5;

      const gridOffset = scrollOffsetRef.current % 5;

      // Small squares (1mm) - scrolling
      for (let x = -gridOffset; x < width; x += 5) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 5) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Large squares (5mm) - scrolling
      ctx.strokeStyle = '#ff8080';
      ctx.lineWidth = 1;
      const largeGridOffset = scrollOffsetRef.current % 25;
      for (let x = -largeGridOffset; x < width; x += 25) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 25) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Add new data points from CSV data (scrolling right to left)
      if (data.length > 0 && animate) {
        const dataIndex = Math.floor(scrollOffsetRef.current / 2) % data.length;
        displayDataRef.current.shift(); // Remove oldest point
        displayDataRef.current.push(data[dataIndex]); // Add new point at the end
      }

      const centerY = height / 2;
      // Adaptive scaling based on actual data range and amplitude control
      const dataValues = displayDataRef.current.filter(v => v !== undefined && isFinite(v));
      const maxAbs = dataValues.length > 0 ? Math.max(...dataValues.map(v => Math.abs(v))) : 0;
      const baseScale = maxAbs > 0 ? (height * 0.4) / maxAbs : height * 0.35;
      const scale = baseScale * amplitude; // Apply amplitude multiplier

      // Draw ECG waveform
      if (displayDataRef.current.length > 0) {
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1.5;
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;

        ctx.beginPath();
        let hasDrawn = false;

        for (let x = 0; x < width; x++) {
          const dataIndex = Math.floor(x * 2); // 2 samples per pixel
          if (dataIndex < displayDataRef.current.length) {
            const value = displayDataRef.current[dataIndex] || 0;
            const y = centerY - value * scale;

            if (!hasDrawn) {
              ctx.moveTo(x, y);
              hasDrawn = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }

        if (hasDrawn) {
          ctx.stroke();
        }
        ctx.shadowBlur = 0;

        // Draw scanning line (like real ECG machine) only when animating
        if (animate) {
          const scanLineX = width - 2;
          ctx.strokeStyle = '#cc0000';
          ctx.lineWidth = 2;
          ctx.globalAlpha = 0.4;
          ctx.beginPath();
          ctx.moveTo(scanLineX, 0);
          ctx.lineTo(scanLineX, height);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }

      // Draw baseline
      ctx.strokeStyle = '#ff6666';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Animate scrolling - restart if needed
      if (animate) {
        scrollOffsetRef.current += 1.5 * speed; // Apply speed multiplier
        animationRef.current = requestAnimationFrame(draw);
      }
    };

    // Start drawing immediately
    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = undefined;
      }
    };
  }, [data, color, width, height, animate, speed, amplitude, redrawTrigger]);

  return (
    <div className={`relative ${highlighted ? 'ring-4 ring-yellow-400 ring-opacity-75 shadow-xl' : ''}`}>
      <canvas
        ref={canvasRef}
        className="rounded-lg"
        style={{ width: '100%', height: 'auto' }}
      />
      <div className={`absolute top-2 left-2 lead-label px-2 py-1 rounded border ${
        highlighted 
          ? 'bg-yellow-200 border-yellow-400 font-bold text-yellow-900' 
          : 'bg-white/90 border-gray-300'
      }`}>
        <span className={highlighted ? 'text-yellow-900 font-bold' : 'text-black font-semibold'}>
          {leadName}
          {highlighted && <span className="ml-2 text-xs">★ KEY LEAD</span>}
        </span>
      </div>
    </div>
  );
};

export default ECGWaveform;
