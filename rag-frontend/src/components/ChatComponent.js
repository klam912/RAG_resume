import React, { useState } from 'react';
import axios from 'axios';

const ChatComponent = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');

    const handleSend = async () => {
        if (input.trim() === '') return;

        const newMessage = { sender: 'user', text: input };
        setMessages([...messages, newMessage]);

        try {
            const response = await axios.post('http://localhost:5000/chat', { query: input });
            const botMessage = { sender: 'bot', text: response.data.response };
            setMessages([...messages, newMessage, botMessage]);
        } catch (error) {
            const errorMessage = { sender: 'bot', text: 'Error getting response' };
            setMessages([...messages, newMessage, errorMessage]);
        }

        setInput('');
    };

    return (
        <div style={{ padding: '20px', maxWidth: '600px', margin: 'auto' }}>
            <div style={{ border: '1px solid #ccc', padding: '10px', height: '400px', overflowY: 'scroll' }}>
                {messages.map((msg, index) => (
                    <div key={index} style={{ textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
                        <strong>{msg.sender}:</strong> {msg.text}
                    </div>
                ))}
            </div>
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                style={{ width: '80%', padding: '10px' }}
            />
            <button onClick={handleSend} style={{ width: '20%', padding: '10px' }}>
                Send
            </button>
        </div>
    );
};

export default ChatComponent;
