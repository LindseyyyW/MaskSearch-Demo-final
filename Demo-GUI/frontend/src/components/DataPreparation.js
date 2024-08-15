import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import CheckPopup from './CheckPopup';
import './DataPreparation.css';

const DataPreparation = () => {
    const [misclassifiedCells, setMisclassifiedCells] = useState({});
    const [selectedLines, setSelectedLines] = useState({});
    const [selectedImages, setSelectedImages] = useState([]);
    const [isPopupOpen, setIsPopupOpen] = useState(false);
    const [animalNames, setNames] = useState({});

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://localhost:9000/api/scenario1/topk_search/pairs', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                });
                const result = await response.json();
                const data = result.dict;
                const names = result.names;
                setNames(names);
                const entries = Object.entries(data);
                // Sort the array based on the length of the value
                entries.sort(([, a], [, b]) => b.length - a.length);
                // Convert the sorted array back into an object
                const sortedMisclassifiedCells = Object.fromEntries(entries);
                setMisclassifiedCells(sortedMisclassifiedCells);
                //setMisclassifiedCells(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        fetchData();
    }, []);
    console.log(misclassifiedCells);
    const handleToggleSelect = (index) => {
        setSelectedLines(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    };

    const handleOpenPopup = (imageIds) => {
        setSelectedImages(imageIds);
        setIsPopupOpen(true);
    };

    const handleClosePopup = () => {
        setIsPopupOpen(false);
    };

    return (
        <div>
            <h1>Top-100 misclassified cells:</h1>
            <div className="misclassified-list">
                {Object.entries(misclassifiedCells).map(([key, imageIds], index) => {
                    const match = key.match(/\((\d+),(\d+)\)/);
                    const x = parseInt(match[1], 10);
                    const y = parseInt(match[2], 10);
                    const img = imageIds[0]
                    let X = animalNames[x];
                    let Y = animalNames[y];
                    // console.log(x);
                    // console.log(y);
                    // console.log(key);

                    return (
                        <div key={index} className={`misclassified-line ${selectedLines[index] ? 'selected' : ''}`}>
                            <div className="cell-info">
                                <span style={{ color: 'blue' }}>{X}</span> predicted as <span style={{ color: 'blue' }}>{Y}</span>
                            </div>
                            <img src={`http://localhost:9000/orig_image/${x}.png`} alt={`Image ${x}`} className="larger-img" />
                            <img src={`http://localhost:9000/topk_results/${img}.png`} alt={`Image ${img}`} className="larger-img" />
                            <img src={`http://localhost:9000/orig_image/${y}.png`} alt={`Image ${y}`} className="larger-img" />
                            <div className="actions">
                                <button className="custom-btn" onClick={() => handleOpenPopup(imageIds)}>Examine More Examples</button>
                                {/* <button className="custom-btn" onClick={() => handleToggleSelect(index)}>
                                    {selectedLines[index] ? 'Cancel Select' : 'Select'}
                                </button> */}
                            </div>
                        </div>
                    );
                })}
            </div>
            <CheckPopup images={selectedImages} isOpen={isPopupOpen} onClose={handleClosePopup} />
            <p>Click <Link to="/input">here</Link> to start query!</p>
        </div>
    );
};

export default DataPreparation;
