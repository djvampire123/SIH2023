import React, { useState, useEffect, useRef } from 'react';
import './YourStyle.css'; // Make sure to import the CSS file appropriately

const ImageCanvasComponent = () => {
    const [isDrawing, setIsDrawing] = useState(false);
    const [roi, setRoi] = useState({ x: 0, y: 0, width: 0, height: 0 });
    const canvasRef = useRef(null);
    const imageRef = useRef(null);

    // Function to handle the loading of the image and updating the canvas
    const handleImageLoad = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = imageRef.current;

        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        // Redraw the rectangle or other elements if needed
    };

    // Function to handle the drawing start
    const startDrawing = (e) => {
        // Set the initial ROI position and setIsDrawing to true
        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setRoi({ ...roi, x, y });
        setIsDrawing(true);
    };

    // Function to handle the drawing move
    const draw = (e) => {
        if (!isDrawing) return;
        // Update the ROI width and height as the mouse moves
        const rect = canvasRef.current.getBoundingClientRect();
        const width = e.clientX - rect.left - roi.x;
        const height = e.clientY - rect.top - roi.y;
        setRoi({ ...roi, width, height });
    };

    // Function to handle the end of drawing
    const stopDrawing = () => {
        setIsDrawing(false);
        // Finalize the drawing and handle any logic you need here
    };

    // Draw the ROI rectangle on the canvas
    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Draw the rectangle
        ctx.beginPath();
        ctx.rect(roi.x, roi.y, roi.width, roi.height);
        ctx.strokeStyle = 'red'; // Example color
        ctx.stroke();
    }, [roi]);

    // Load the image when the component mounts
    useEffect(() => {
        if (imageRef.current) {
            imageRef.current.addEventListener('load', handleImageLoad);
        }

        return () => {
            if (imageRef.current) {
                imageRef.current.removeEventListener('load', handleImageLoad);
            }
        };
    }, []);

    return (
        <div id="image-container" style={{ position: 'relative', display: 'inline-block' }}>
            <canvas
                id="roi-canvas"
                ref={canvasRef}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseOut={stopDrawing}
                style={{ position: 'absolute', top: 0, left: 0 }}
            />
            <img
                ref={imageRef}
                id="image"
                src="S:\Files\Work\Hackathon\SIH\Changing\Bounding_box_web\static\output (1).jpg" // Replace with your image path
                alt=""
                style={{ width: '500px', height: 'auto', display: 'block' }} // Adjust the width and height as needed
            />
        </div>
    );
};

export default ImageCanvasComponent;