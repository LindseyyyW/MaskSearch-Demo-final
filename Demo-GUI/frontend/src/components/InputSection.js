// src/components/InputSection.js
import React, { useState } from 'react';
import QueryCommand from './QueryCommand';
import './InputSection.css'; // Make sure to create this CSS file

function InputSection({scenario, onSearchResults, onModeChange, setExecutionTime, isAug, ms}) {
    const [mode, setMode] = useState('Top-K');
    const [aug, setAug] = useState(false)
    const [k, setK] = useState('5');
    const [roi, setRoi] = useState('object bounding box');
    const [pixelUpperBound, setPixelUpperBound] = useState('1');
    const [pixelLowerBound, setPixelLowerBound] = useState('0.5');
    const [order, setOrder] = useState('ASC');
    const [threshold, setThreshold] = useState('0.2');
    const [thresholdDirection, setThresholdDirection] = useState('>');
    const [queryCommand, setQueryCommand] = useState('');
    const [isQueryActive, setIsQueryActive] = useState(false);
    const [isPathActive, setIsPathActive] = useState(false);
    const [latestImageIds, setLatestImageIds] = useState([]);


    const fetchQueryCommand = async (path, body) => {
        const response = await fetch(`http://localhost:8000${path}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });
        const data = await response.json();
        setQueryCommand(data.query_command);
        onSearchResults(data.image_ids);
        setExecutionTime(data.execution_time);  // Set execution time received from the backend
        //setIsQueryActive(true); // Enable the Augment button
        return data;
    };

    

    const handleStartQuery = async () => {
        handleAugment(false);
        let path, body;
        if (mode === 'Top-K') {
            path = `/api/${scenario}/topk_search`;
            body = { k, roi, pixelUpperBound, pixelLowerBound, order, ms};
        } else {
            path = `/api/${scenario}/filter_search`;
            body = { threshold, thresholdDirection, roi, pixelUpperBound, pixelLowerBound, ms};
        }

        const data = await fetchQueryCommand(path, body);
        setQueryCommand(data.query_command);
        onSearchResults(data.image_ids);
        setLatestImageIds(data.image_ids);
        //imgs = data.image_ids;
        setIsQueryActive(true); // Enable the Augment button
        await fetchQueryCommand(path, body);
    };

    const handleStartAugment = async () => {
        handleAugment(true);
        const response = await fetch('http://localhost:8000/api/scenario1/augment', {method: 'POST', headers: {
            'Content-Type': 'application/json',
        }, body: JSON.stringify({ image_ids: latestImageIds }),});
        const data = await response.json();
        onSearchResults(data.image_ids);
        setIsPathActive(true);
    };


    const handleModeChange = (newMode) => {
        setMode(newMode); // Update local state
        setIsQueryActive(false); // Reset the active state when changing modes
        onModeChange(newMode); // Inform the parent component about the mode change
    };

    const handleAugment = (newaug) => {
        setAug(newaug);
        isAug(newaug);
    }

    return (
        <div className="input-section">
            <div className="header">
                <h2 className="title">{mode} Query</h2>
                <div className="mode-switch">
                    <button className={mode === 'Top-K' ? 'active' : ''} onClick={() => handleModeChange('Top-K')}>Top-K</button>
                    <button className={mode === 'Filter' ? 'active' : ''} onClick={() => handleModeChange('Filter')}>Filter</button>
                    <button className={mode === ''}>Aggregation</button>
                </div>
            </div>
            {mode === 'Top-K' ? (
                <>
                    {/* Top-K specific fields */}
                    <div className="input-container">
                        <label htmlFor="K" className="input-label">K:</label>
                        <select id="k" value={k} onChange={(e) => setK(e.target.value)} className="input-field">
                            {[5, 10, 15, 20, 25].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-container">
                        <label htmlFor="ROI" className="input-label">ROI:</label>
                        <select id="roi" value={roi} onChange={(e) => setRoi(e.target.value)} className="input-field">
                            {['object bounding box', 'customizing box'].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelUpperBound" className="input-label">Pixel Value Upper Bound:</label>
                        <input
                            id="pixelUpperBound"
                            type="text"
                            value={pixelUpperBound}
                            onChange={(e) => setPixelUpperBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelLowerBound" className="input-label">Pixel Value Lower Bound:</label>
                        <input
                            id="pixelLowerBound"
                            type="text"
                            value={pixelLowerBound}
                            onChange={(e) => setPixelLowerBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                    <div className="input-container">
                        <label htmlFor="Order" className="input-label">Order:</label>
                        <select id="order" value={order} onChange={(e) => setOrder(e.target.value)} className="input-field">
                            {['ASC', 'DESC'].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                </>
            ) : (
                <>
                    {/* Filter specific fields */}
                    <div className="input-container">
                        <label htmlFor="threshold" className="input-label">Threshold:</label>
                        <div className="threshold-container">
                            <select
                                id="thresholdDirection"
                                value={thresholdDirection}
                                onChange={(e) => setThresholdDirection(e.target.value)}
                                className="threshold-field"
                            >
                                <option value=">">&gt;</option>
                                <option value="<">&lt;</option>
                            </select>
                            <input
                                id="thresholdValue"
                                type="text"
                                value={threshold}
                                onChange={(e) => setThreshold(e.target.value)}
                                className="threshold-field"
                            />
                        </div>
                    </div>
                    <div className="input-container">
                        <label htmlFor="ROI" className="input-label">ROI:</label>
                        <select id="roi" value={roi} onChange={(e) => setRoi(e.target.value)} className="input-field">
                            {['object bounding box', 'customizing box'].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelUpperBound" className="input-label">Pixel Value Upper Bound:</label>
                        <input
                            id="pixelUpperBound"
                            type="text"
                            value={pixelUpperBound}
                            onChange={(e) => setPixelUpperBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelLowerBound" className="input-label">Pixel Value Lower Bound:</label>
                        <input
                            id="pixelLowerBound"
                            type="text"
                            value={pixelLowerBound}
                            onChange={(e) => setPixelLowerBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                </>
            )}
            <div className="halfsize-buttons">
                <button className="start-halfsize-btn" onClick={handleStartQuery}>
                    Start Query
                </button>
                <button className="start-halfsize-btn" onClick={handleStartAugment} disabled={!isQueryActive}>
                    Augment
                </button>
                {/* <button className="appending-halfsize-btn" onClick={handleShowExecution} disabled={!isQueryActive}>
                    Execution Detail
                </button> */}
            </div>
            {/* <div className="halfsize-buttons">
                <button className="start-halfsize-btn" onClick={handleStartAugment} disabled={!isQueryActive}>
                    Augment
                </button>
                <button className="appending-halfsize-btn" onClick={handleShowPath} disabled={!isPathActive}>
                    Show Path
                </button>
            </div> */}
            {/* <button onClick={() => {}} disabled={!isQueryActive}>Start Augment</button> */}
            <QueryCommand command={queryCommand} />
        </div>
    );
}

export default InputSection;

// src/components/InputSection.js
// import React, { useState } from 'react';
// import QueryCommand from './QueryCommand';
// import './InputSection.css';

// // TODO: Add the field for augment
// function InputSection({ onSearchResults, onModeChange, setExecutionTime }) {
//     const [mode, setMode] = useState('Top-K');
//     const [k, setK] = useState('5');
//     const [roi, setRoi] = useState('object bounding box');
//     const [pixelUpperBound, setPixelUpperBound] = useState('1');
//     const [pixelLowerBound, setPixelLowerBound] = useState('0.5');
//     const [order, setOrder] = useState('ASC');
//     const [threshold, setThreshold] = useState('0.2');
//     const [thresholdDirection, setThresholdDirection] = useState('>');
//     const [queryCommand, setQueryCommand] = useState('');
//     const [isQueryActive, setIsQueryActive] = useState(false);
//     const [isPathActive, setIsPathActive] = useState(false);


//     const fetchQueryCommand = async (path, body) => {
//         const response = await fetch(`http://localhost:8000${path}`, {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify(body),
//         });
//         const data = await response.json();
//         setQueryCommand(data.query_command);
//         onSearchResults(data.image_ids);
//         setExecutionTime(data.execution_time);  // Set execution time received from the backend
//         setIsQueryActive(true); // Enable the Augment button
//         return data;
//     };

//     const handleStartQuery = async () => {
//         let path, body;
//         if (mode === 'Top-K') {
//             path = '/api/topk_search';
//             body = { k, roi, pixelUpperBound, pixelLowerBound, order };
//         } 
//         if (mode === 'Filter') {
//             path = '/api/filter_search';
//             body = { threshold, thresholdDirection, roi, pixelUpperBound, pixelLowerBound };
//         }
//         if (mode === 'Aggregation') {
//             path = '/api/filter_search';
//             body = { threshold, thresholdDirection, roi, pixelUpperBound, pixelLowerBound };
//         }

//         await fetchQueryCommand(path, body);
//     };

//     const handleShowExecution = () => {
//         // Implement later, make a window to show path
//     };

//     const handleStartAugment = async () => {
//         // const response = await fetch('http://localhost:5000/api/augment');
//         // const data = await response.json();
//         // console.log('Augment result:', data.result);
//         setIsPathActive(true);
//         // TODO: Set the status of augment to true, other necessary implementation for augment
//     };

//     const handleShowPath = () => {
//         // Implement later, make a window to show path
//     };

//     const handleModeChange = (newMode) => {
//         setMode(newMode); // Update local state
//         setIsQueryActive(false); // Reset the active state when changing modes
//         onModeChange(newMode); // Inform the parent component about the mode change
//         // TODO: Reset the status of augment to false when changing modes
//     };

//     return (
//         <div className="input-section">
//             <div className="header">
//                 <h2 className="title">{mode}</h2>
//                 <div className="mode-switch">
//                     <button className={mode === 'Top-K' ? 'active' : ''} onClick={() => handleModeChange('Top-K')}>Top-K</button>
//                     <button className={mode === 'Filter' ? 'active' : ''} onClick={() => handleModeChange('Filter')}>Filter</button>
//                     <button className={mode === 'Aggregation' ? 'active' : ''} onClick={() => handleModeChange('Aggregation')}>Aggregation</button>
//                 </div>
//             </div>
//             {mode === 'Top-K' ? (
//                 <>
//                     {/* Top-K specific fields */}
//                     <div className="input-container">
//                         <label htmlFor="K" className="input-label">K:</label>
//                         <select id="k" value={k} onChange={(e) => setK(e.target.value)} className="input-field">
//                             {[5, 10, 15, 20, 25].map(option => (
//                                 <option key={option} value={option}>{option}</option>
//                             ))}
//                         </select>
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="ROI" className="input-label">ROI:</label>
//                         <select id="roi" value={roi} onChange={(e) => setRoi(e.target.value)} className="input-field">
//                             {['object bounding box', 'customizing box'].map(option => (
//                                 <option key={option} value={option}>{option}</option>
//                             ))}
//                         </select>
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="pixelUpperBound" className="input-label">Pixel Value Upper Bound:</label>
//                         <input
//                             id="pixelUpperBound"
//                             type="text"
//                             value={pixelUpperBound}
//                             onChange={(e) => setPixelUpperBound(e.target.value)}
//                             className="input-field"
//                         />
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="pixelLowerBound" className="input-label">Pixel Value Lower Bound:</label>
//                         <input
//                             id="pixelLowerBound"
//                             type="text"
//                             value={pixelLowerBound}
//                             onChange={(e) => setPixelLowerBound(e.target.value)}
//                             className="input-field"
//                         />
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="Order" className="input-label">Order:</label>
//                         <select id="order" value={order} onChange={(e) => setOrder(e.target.value)} className="input-field">
//                             {['ASC', 'DESC'].map(option => (
//                                 <option key={option} value={option}>{option}</option>
//                             ))}
//                         </select>
//                     </div>
//                 </>
//             ) : mode === 'Filter' ? (
//                 <>
//                     {/* Filter specific fields */}
//                     <div className="input-container">
//                         <label htmlFor="threshold" className="input-label">Threshold:</label>
//                         <div className="threshold-container">
//                             <select
//                                 id="thresholdDirection"
//                                 value={thresholdDirection}
//                                 onChange={(e) => setThresholdDirection(e.target.value)}
//                                 className="threshold-field"
//                             >
//                                 <option value=">">&gt;</option>
//                                 <option value="<">&lt;</option>
//                             </select>
//                             <input
//                                 id="thresholdValue"
//                                 type="text"
//                                 value={threshold}
//                                 onChange={(e) => setThreshold(e.target.value)}
//                                 className="threshold-field"
//                             />
//                         </div>
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="ROI" className="input-label">ROI:</label>
//                         <select id="roi" value={roi} onChange={(e) => setRoi(e.target.value)} className="input-field">
//                             {['object bounding box', 'customizing box'].map(option => (
//                                 <option key={option} value={option}>{option}</option>
//                             ))}
//                         </select>
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="pixelUpperBound" className="input-label">Pixel Value Upper Bound:</label>
//                         <input
//                             id="pixelUpperBound"
//                             type="text"
//                             value={pixelUpperBound}
//                             onChange={(e) => setPixelUpperBound(e.target.value)}
//                             className="input-field"
//                         />
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="pixelLowerBound" className="input-label">Pixel Value Lower Bound:</label>
//                         <input
//                             id="pixelLowerBound"
//                             type="text"
//                             value={pixelLowerBound}
//                             onChange={(e) => setPixelLowerBound(e.target.value)}
//                             className="input-field"
//                         />
//                     </div>
//                 </>
//             ) : (
//                 <>
//                     {/* Filter specific fields */}
//                     <div className="input-container">
//                         <label htmlFor="threshold" className="input-label">Threshold:</label>
//                         <div className="threshold-container">
//                             <select
//                                 id="thresholdDirection"
//                                 value={thresholdDirection}
//                                 onChange={(e) => setThresholdDirection(e.target.value)}
//                                 className="threshold-field"
//                             >
//                                 <option value=">">&gt;</option>
//                                 <option value="<">&lt;</option>
//                             </select>
//                             <input
//                                 id="thresholdValue"
//                                 type="text"
//                                 value={threshold}
//                                 onChange={(e) => setThreshold(e.target.value)}
//                                 className="threshold-field"
//                             />
//                         </div>
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="ROI" className="input-label">ROI:</label>
//                         <select id="roi" value={roi} onChange={(e) => setRoi(e.target.value)} className="input-field">
//                             {['object bounding box', 'customizing box'].map(option => (
//                                 <option key={option} value={option}>{option}</option>
//                             ))}
//                         </select>
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="pixelUpperBound" className="input-label">Pixel Value Upper Bound:</label>
//                         <input
//                             id="pixelUpperBound"
//                             type="text"
//                             value={pixelUpperBound}
//                             onChange={(e) => setPixelUpperBound(e.target.value)}
//                             className="input-field"
//                         />
//                     </div>
//                     <div className="input-container">
//                         <label htmlFor="pixelLowerBound" className="input-label">Pixel Value Lower Bound:</label>
//                         <input
//                             id="pixelLowerBound"
//                             type="text"
//                             value={pixelLowerBound}
//                             onChange={(e) => setPixelLowerBound(e.target.value)}
//                             className="input-field"
//                         />
//                     </div>
//                 </>
//             ) }
//             <div className="halfsize-buttons">
//                 <button className="start-halfsize-btn" onClick={handleStartQuery}>
//                     Start Query
//                 </button>
//                 <button className="appending-halfsize-btn" onClick={handleShowExecution} disabled={!isQueryActive}>
//                     Execution Detail
//                 </button>
//             </div>
//             <div className="halfsize-buttons">
//                 <button className="start-halfsize-btn" onClick={handleStartAugment}>
//                     Start Augment
//                 </button>
//                 <button className="appending-halfsize-btn" onClick={handleShowPath} disabled={!isPathActive}>
//                     Show Path
//                 </button>
//             </div>
//             {/* <button onClick={() => {}} disabled={!isQueryActive}>Start Augment</button> */}
//             <QueryCommand command={queryCommand} />
//         </div>
//     );
// }

// export default InputSection;