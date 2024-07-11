// import React from 'react';
// import './CheckPopup.css';

// const CheckPopup = ({ isOpen, onClose }) => {
//     if (!isOpen) return null;

//     const [images, setImages] = useState([]);

//     useEffect(() => {
//         // Fetch the pairs from the backend
//         const fetchIdx = async () => {
//             const response = await fetch('http://localhost:8000/api/get_pairs');
//             const data = await response.json();
//             const idx = data.index
//             setImages(idx);
//         };

//         fetchIdx();
//     }, []);
//     // Dummy data showing 25 images
//     //const images = Array(25).fill('1.jpg');

//     return (
//         <div className="check-popup-overlay">
//             <div className="check-popup">
//                 <button className="close-btn" onClick={onClose}>x</button>
//                 <div className="images-grid">
//                     {images.map((img, index) => (
//                         <img key={index} src={`http://localhost:8000/topk_results/${img}`} alt={`Image ${index}`} className="popup-img" />
//                     ))}
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default CheckPopup;
import React from 'react';
import './CheckPopup.css';

const CheckPopup = ({ images, isOpen, onClose }) => {
    if (!isOpen) return null;

    return (
        <div className="check-popup-overlay">
            <div className="check-popup">
                <button className="close-btn" onClick={onClose}>x</button>
                <div className="images-grid">
                    {images.map((id, index) => (
                        <img key={index} src={`http://localhost:9000/topk_results/${id}.png`} alt={`Image ${id}`} className="popup-img" />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default CheckPopup;
