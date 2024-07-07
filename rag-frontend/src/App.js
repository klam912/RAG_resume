import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
    const [file, setFile] = useState(null);
    const [query, setQuery] = useState('');
    const [message, setMessage] = useState('');

    const onChangeHandler = (event) => {
        setFile(event.target.files[0]);
    };

    const onSubmitHandler = async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('file', file);
        formData.append('query', query);

        try {
            const response = await axios.post('http://localhost:5000/upload-and-query', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setMessage(response.data.message);
        } catch (error) {
            console.error('Error uploading file or querying:', error);
            setMessage('Error occurred during upload or query.');
        }
    };

    return (
        <div>
            <h1>RAG Application</h1>
            <form onSubmit={onSubmitHandler}>
                <input type="file" onChange={onChangeHandler} />
                <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Enter your query" />
                <button type="submit">Submit</button>
            </form>
            {message && <p>{message}</p>}
        </div>
    );
};

export default App;
