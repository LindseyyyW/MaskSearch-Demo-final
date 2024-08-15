// // src/components/ImageSelection.js
// import React from 'react';
// import Modal from 'react-modal';

// Modal.setAppElement('#root');

// function ImageSelection({ isOpen, imageId, onRequestClose, mode }) {
//     let imagePath;
//     if (mode === 'Top-K') {
//         imagePath = 'topk_results';
//     } else if (mode === 'Filter') {
//         imagePath = 'topk_results';
//     } else if (mode === 'Aggregation') {
//         imagePath = 'aggregation_results';
//     }
//     const imageUrl = `http://localhost:9000/${imagePath}/${imageId}.png`;

//     const handleImageLoadError = () => {
//         console.error('Failed to load image with ID:', imageId);
//     };

//     return (
//         <Modal
//             isOpen={isOpen}
//             onRequestClose={onRequestClose}
//             style={{
//                 overlay: {
//                     backgroundColor: 'rgba(0, 0, 0, 0.8)' // Optional: dark overlay
//                 },
//                 content: {
//                     position: 'fixed',
//                     top: '50%',
//                     left: '50%',
//                     right: 'auto',
//                     bottom: 'auto',
//                     transform: 'translate(-50%, -50%)',
//                     maxWidth: '90%', // Limiting image size
//                     maxHeight: '90%', // Limiting image size
//                     overflow: 'auto' // Ensures content can be scrolled if larger than the modal
//                 }
//             }}
//         >
//             <div style={{ textAlign: 'center' }}>
//                 <img
//                     src={imageUrl}
//                     alt={`Selected Image ${imageId}`}
//                     onError={handleImageLoadError}
//                     style={{ maxWidth: '100%', maxHeight: '100vh' }} // Resizes image to not be too large
//                 />
//                 <div>
//                     <button onClick={onRequestClose} style={{ marginTop: '20px' }}>Close</button>
//                 </div>
//             </div>
//         </Modal>
//     );
// }

// export default ImageSelection;


import React, { useState, useEffect } from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root');

function ImageSelection({ scenario, isOpen, imageId, onRequestClose, mode }) {
    const [stat, setStat] = useState('');
    const [imageUrl, setImageUrl] = useState('');

    useEffect(() => {
        let imagePath;
        let statUrl;

        if (scenario === 'scenario1') {
            if (mode === 'Top-K' || mode === 'Filter' || mode === 'Aggregation') {
                imagePath = 'topk_results';
            }
            setImageUrl(`http://localhost:9000/${imagePath}/${imageId}.jpg`);
        } else if (scenario === 'scenario2') {
            imagePath = 'topk_images';
            statUrl = `http://localhost:9000/topk_labels/${imageId}`;
            setImageUrl(`http://localhost:9000/${imagePath}/${imageId}.JPEG`);

            const fetchImageLabels = async () => {
                const response = await fetch(statUrl, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                });
                const labels = await response.json();
                const correctness = labels.correctness ? 'Correct' : 'Incorrect';
                const attack = labels.attack ? 'Yes' : 'No';
                setStat(`Prediction: ${correctness};  Attack: ${attack}`);
            };

            fetchImageLabels();
        } else if (scenario === 'scenario3') {
            if (mode === 'Top-K') {
                imagePath = 'topk_results';
            } else if (mode === 'Filter') {
                imagePath = 'filter_results';
            } else if (mode === 'Aggregation') {
                imagePath = 'aggregation_results';
            }
            setImageUrl(`http://localhost:9000/${imagePath}/${imageId}.jpg`);
        }
    }, [imageId, mode, scenario]);

    const handleImageLoadError = () => {
        console.error('Failed to load image with ID:', imageId);
    };

    return (
        <Modal
            isOpen={isOpen}
            onRequestClose={onRequestClose}
            style={{
                overlay: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)'
                },
                content: {
                    position: 'fixed',
                    top: '50%',
                    left: '50%',
                    right: 'auto',
                    bottom: 'auto',
                    transform: 'translate(-50%, -50%)',
                    maxWidth: '90%',
                    maxHeight: '90%',
                    overflow: 'auto'
                }
            }}
        >
            <div style={{ textAlign: 'center' }}>
                <img
                    src={imageUrl}
                    alt={`Selected Image ${imageId}`}
                    onError={handleImageLoadError}
                    style={{ maxWidth: '100%', maxHeight: '80vh' }}
                />
                {scenario === 'scenario2' && (
                    <div style={{ marginTop: '5px' }}>
                        {stat}
                    </div>
                )}
                <div>
                    <button onClick={onRequestClose} style={{ marginTop: '20px' }}>Close</button>
                </div>
            </div>
        </Modal>
    );
}

export default ImageSelection;