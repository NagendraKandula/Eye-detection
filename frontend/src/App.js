import React, { useRef, useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const fileInputRef = useRef();
  const [results, setResults] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [loading, setLoading] = useState(false);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) {
      alert('Please select an image first');
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('https://eye-detection.onrender.com/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        alert(data.error || 'Error during prediction');
        setResults([]);
        setLoading(false);
        return;
      }

      setResults(data.result);
    } catch (error) {
      console.error('Error during image submission:', error);
      alert('Network error or server not reachable');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setResults([]);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setImage(file);
      setResults([]);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className={`App ${darkMode ? 'dark-mode' : ''}`}>
      <h1>Eye Disease Detection and Remedy</h1>
      <button onClick={toggleDarkMode} className="dark-mode-toggle">
        {darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      </button>

      <form className="image-form" onSubmit={handleSubmit}>
        <div
          className="image-viewer"
          onClick={handleClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {previewImage ? (
            <img src={previewImage} alt="Preview" />
          ) : (
            <p className="drag-drop-text">
              Drag & Drop or Click to Select Image
            </p>
          )}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImageChange}
            accept="image/*"
            style={{ display: 'none' }}
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Submit'}
        </button>
      </form>

      <div className="results-section">
        {results.length > 0 &&
          results.map((result, index) => (
            <div className="result-card" key={index}>
              <h2>{result.model}</h2>
              <h3>{result.name}</h3>
              <p>
                <strong>Class ID:</strong> {result.predicted_class}
              </p>
              <p>
                Result Accuracy: <strong>{result.accuracy}</strong>
              </p>
              <p>{result.remedy}</p>
            </div>
          ))}
      </div>
    </div>
  );
}

export default App;
