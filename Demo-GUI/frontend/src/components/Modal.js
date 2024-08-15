import React, { useState, useEffect } from 'react';
import Chart from 'chart.js/auto'; 
import './Modal.css'

const Modal = ({ isModalOpen, loadedCount, totalCount, onClose}) => {

  useEffect(() => {
    if (isModalOpen) {
        console.log("here")
      const ctx = document.getElementById('myChart');
      const data = {
        labels: ['Number of masks loaded from disk', 'Number of masks skipped'],
        datasets: [{
          label: 'Comparison of Loaded Masks vs. Skipped Masks',
          data: [parseFloat(loadedCount), parseFloat(totalCount-loadedCount)],
          backgroundColor: ['rgb(255, 99, 132)', 'rgb(54, 162, 235)'],
          hoverOffset: 4
        }]
      };
      const config = {
        type: 'pie',
        data: data,
        options: {
          responsive: true,
          maintainAspectRatio: false,
        }
      };
      //Destroy existing chart instance before creating a new one
      if (ctx.chart) {
        ctx.chart.destroy();
      }
      ctx.chart = new Chart(ctx, config);
    }
  }, [isModalOpen, loadedCount, totalCount]);

  if (!isModalOpen) {
    console.log("no")
    return null;
  }
  return (
    <div className="modal-container">
      <button className="close-button" onClick={onClose}>x</button>
      <div className="chart-container">
        <canvas id="myChart"></canvas>
      </div>
    </div>
  );
};

export default Modal;