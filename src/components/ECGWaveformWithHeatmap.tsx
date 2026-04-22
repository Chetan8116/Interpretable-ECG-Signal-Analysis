import React, { useRef, useEffect } from 'react';

interface HeatmapSegment {
  start: number;
  end: number;
  color: string;
  intensity: number;
  label?: string;
}

interface ECGWaveformWithHeatmapProps {
  data: number[];
  leadName: string;
  color: string;
  width?: number;
  height?: number;
  animate?: boolean;
  speed?: number;
  amplitude?: number;
  highlighted?: boolean;
  heatmapSegments?: HeatmapSegment[];
}

const ECGWaveformWithHeatmap: React.FC<ECGWaveformWithHeatmapProps> = ({
  data,
  leadName,
  color,
  width = 600,
  height = 120,
  animate = true,
  speed = 1,
  amplitude = 1,
  highlighted = false,
  heatmapSegments = []
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const scrollOffsetRef = useRef<number>(0);
  const displayDataRef = useRef<number[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution
    canvas.width = width;
    canvas.height = height;

    // Initialize display buffer with actual data
    if (data.length > 0 && displayDataRef.current.length === 0) {
      const bufferSize = width * 2;
      displayDataRef.current = new Array(bufferSize).fill(0);
      for (let i = 0; i < Math.min(bufferSize, data.length); i++) {
        displayDataRef.current[bufferSize - data.length + i] = data[i];
      }
    }

    const draw = () => {
      // Clear canvas with pink ECG paper background
      ctx.fillStyle = '#ffe6e6';
      ctx.fillRect(0, 0, width, height);

      // Draw scrolling grid
      ctx.strokeStyle = '#ffb3b3';
      ctx.lineWidth = 0.5;
      const gridOffset = scrollOffsetRef.current % 5;
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

      // Large squares
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

      // Add new data points
      if (data.length > 0 && animate) {
        const dataIndex = Math.floor(scrollOffsetRef.current / 2) % data.length;
        displayDataRef.current.shift();
        displayDataRef.current.push(data[dataIndex]);
      }

      const centerY = height / 2;
      const dataValues = displayDataRef.current.filter(v => v !== undefined);
      const maxAbs = Math.max(...dataValues.map(v => Math.abs(v)));
      const baseScale = maxAbs > 0 ? (height * 0.4) / maxAbs : height * 0.35;
      const scale = baseScale * amplitude;

      if (displayDataRef.current.length > 0 && displayDataRef.current.some(v => v !== undefined && v !== 0)) {

        // 1. Draw heatmap — background shade + crisp bar — BEHIND the waveform
        if (heatmapSegments.length > 0 && !animate) {
          heatmapSegments.forEach(segment => {
            const startX = Math.floor((segment.start / data.length) * width);
            const endX   = Math.ceil((segment.end   / data.length) * width);
            const barW   = Math.max(2, endX - startX);

            if (!segment.label) {
              // Soft background shading (whole cycle)
              ctx.fillStyle = segment.color;
              ctx.fillRect(startX, 0, barW, height);
            } else {
              // Crisp feature bar
              ctx.fillStyle = segment.color.replace(/[\d.]+\)$/, '0.80)');
              ctx.fillRect(startX, 0, barW, height);
              // Accent cap
              ctx.fillStyle = segment.color.replace(/[\d.]+\)$/, '1)');
              ctx.fillRect(startX, 0, barW, 3);
              ctx.fillRect(startX, height - 3, barW, 3);
            }
          });
        }

        // 2. Draw ECG waveform ON TOP in black
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1.5;
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;

        ctx.beginPath();
        for (let x = 0; x < width; x++) {
          const dataIndex = Math.floor(x * 2);
          const value = displayDataRef.current[dataIndex] || 0;
          const y = centerY - value * scale;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Scanning line (animated mode only)
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

      // Animate scrolling
      if (animate) {
        scrollOffsetRef.current += 1.5 * speed;
        animationRef.current = requestAnimationFrame(draw);
      }
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [data, color, width, height, animate, speed, amplitude, heatmapSegments]);

  // Reset display buffer when data changes
  useEffect(() => {
    if (data.length > 0) {
      const bufferSize = width * 2;
      displayDataRef.current = new Array(bufferSize);
      for (let i = 0; i < bufferSize; i++) {
        displayDataRef.current[i] = data[i % data.length] || 0;
      }
      scrollOffsetRef.current = 0;
    }
  }, [data, width]);

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
      
      {/* Feature marker badges (bottom-right) */}
      {heatmapSegments.some(s => s.label) && (
        <div className="absolute bottom-2 right-2 flex flex-wrap justify-end gap-1 max-w-[75%]">
          {Array.from(
            heatmapSegments
              .filter(segment => Boolean(segment.label))
              .reduce((acc, segment) => {
                if (segment.label && !acc.has(segment.label)) {
                  acc.set(segment.label, segment.color);
                }
                return acc;
              }, new Map<string, string>())
          ).slice(0, 4).map(([label, markerColor]) => (
            <div
              key={label}
              className="px-2 py-0.5 rounded text-[10px] font-bold text-white"
              style={{ background: markerColor.replace(/[\d.]+\)$/, '0.9)') }}
            >
              {label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ECGWaveformWithHeatmap;
