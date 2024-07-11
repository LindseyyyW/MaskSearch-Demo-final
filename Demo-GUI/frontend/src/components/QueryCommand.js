// src/components/QueryCommand.js
import React from 'react';
import './QueryCommand.css';

const QueryCommand = ({ command }) => {
    // Function to remove leading whitespace from each line
    const formatCommand = (cmd) => {
        return cmd
            .split('\n') // Split the command into lines
            .map(line => line.trimStart()) // Remove leading whitespace from each line
            .join('\n'); // Rejoin the lines into a single string
    };

    return (
        <div className="query-command-container">
            <h3>Query Command</h3>
            <textarea readOnly value={formatCommand(command)} />
        </div>
    );
};

export default QueryCommand;
