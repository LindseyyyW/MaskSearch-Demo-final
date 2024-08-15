import React from 'react';
import './ResultsSection.css';

function ResultsSection({ scenario, imageIds, mode, onSelectImage, executionTime, imagesCount, skippedImages, aug }) {
    console.log('ResultsSection mode:', mode);

    let imagePath;
    if (scenario === 'scenario1') {
        if (mode === 'Top-K') {
            imagePath = 'topk_results';
        } else if (mode === 'Filter') {
            imagePath = 'topk_results';
        } else if (mode === 'Aggregation') {
            imagePath = 'aggregation_results';
        }
        if (aug === true) {
            imagePath = 'augment_results';
        }
    } else if (scenario === 'scenario2') {
        if (mode === 'Top-K') {
            imagePath = 'topk_cams';
        }
    } else if (scenario === 'scenario3') {
        imagePath = {
            saliency: 'saliency_images',
            attention: 'human_att_images',
            intersect: 'intersect_visualization',
            union: 'union_visualization'
        };
    }

    const renderScenario1 = () => {
        let len = imageIds.length;
        if (imageIds.length > 50) {
            imageIds = imageIds.slice(0, 50);
        }
        return (
            <>
                <div className="execution-info">
                    <span className="time-label">Execution Time:</span>
                    <span className="time-value">{executionTime.toFixed(3)} seconds</span>
                </div>
                <div className="image-count-info">
                    <span className="image-count-label">Returned Examples:</span>
                    <span className="image-count-value">{len}</span>
                </div>
                <div className="image-container">
                    {imageIds.map((id) => (
                        <img
                            key={id}
                            src={`http://localhost:9000/${imagePath}/${id}.png`}
                            alt={`Image ${id}`}
                            onClick={() => onSelectImage(id)}
                        />
                    ))}
                </div>
            </>
        );
    };

    const renderScenario2 = () => (
        <>
            <div className="execution-info">
                <span className="time-label">Execution Time:</span>
                <span className="time-value">{executionTime.toFixed(3)} seconds</span>
            </div>
            <div className="image-count-info">
                <span className="image-count-label">Skipped Images Count:</span>
                <span className="image-count-value">{skippedImages}</span>
            </div>
            <div className="image-container">
                {imageIds.map((id) => (
                    <img
                        key={id}
                        src={`http://localhost:9000/${imagePath}/${id}.png`}
                        alt={`Image ${id}`}
                        onClick={() => onSelectImage(id)}
                    />
                ))}
            </div>
        </>
    );

    const renderScenario3 = () => {
        let len = imageIds.length;
        if (imageIds.length > 50) {
            imageIds = imageIds.slice(0, 50);
        }
        return (
            <>
                <div className="execution-info">
                    <span className="time-label">Execution Time:</span>
                    <span className="time-value">{executionTime} seconds</span>
                </div>
                <div className="image-count-info">
                        <span className="image-count-label">Returned Examples:</span>
                        <span className="image-count-value">{len}</span>
                </div>
                <div className="results-section scenario3">
                    {imageIds.map((id) => (
                        <div key={id} className="image-row">
                            <div className="image-wrapper">
                                <img
                                    className="result-image"
                                    src={`http://localhost:9000/${imagePath.saliency}/${id}_saliency.jpg`}
                                    alt={`Saliency Image ${id}`}
                                    onClick={() => onSelectImage(id)}
                                />
                                <p>Saliency Mask {id}</p>
                            </div>
                            <div className="image-wrapper">
                                <img
                                    className="result-image"
                                    src={`http://localhost:9000/${imagePath.attention}/${id}.jpg`}
                                    alt={`Human Attention Image ${id}`}
                                    onClick={() => onSelectImage(id)}
                                />
                                <p>Human Attention Mask {id}</p>
                            </div>
                            <div className="image-wrapper">
                                <img
                                    className="result-image"
                                    src={`http://localhost:9000/${imagePath.intersect}/intersect_result_${id}.png`}
                                    alt={`Intersect Image ${id}`}
                                    onClick={() => onSelectImage(id)}
                                />
                                <p>Intersection Mask {id}</p>
                            </div>
                            <div className="image-wrapper">
                                <img
                                    className="result-image"
                                    src={`http://localhost:9000/${imagePath.union}/union_result_${id}.png`}
                                    alt={`Union Image ${id}`}
                                    onClick={() => onSelectImage(id)}
                                />
                                <p>Union Mask {id}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </>
        );
    };
    
    
    return (
        <div className={`results-section ${scenario}`}>
            <div className="info-box">
                {scenario === 'scenario1' && renderScenario1()}
                {scenario === 'scenario2' && renderScenario2()}
                {scenario === 'scenario3' && renderScenario3()}
            </div>
        </div>
    );
}

export default ResultsSection;

