document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');

    let chatMessages = [];
    

    function updateChatWindow() {
        chatContainer.innerHTML = '';
        chatMessages.forEach(message => {
            const messageElement = document.createElement('div');
            if (message.source == 'AI' && message.steps_taken) {
                // Create a separate element for the steps_taken with the specified class
                const stepsTakenContainer = document.createElement('div');
                const stepsTakenElement = document.createElement('span')
                stepsTakenElement.textContent = message.steps_taken;
                stepsTakenElement.classList.add('steps-taken-message');
                
                // Append the stepsTakenElement to the stepsTakenContainer
                stepsTakenContainer.appendChild(stepsTakenElement);
                stepsTakenContainer.classList.add('steps-taken-container');

                // Append the stepsTakenContainer to the message container
                messageElement.appendChild(stepsTakenContainer);
                const responseElement = document.createElement('span');
                responseElement.textContent = '[AI]:\n' + message.message;
                // You can add additional styling to this element if needed
                messageElement.appendChild(responseElement);
            } 
            else {
                // If no steps_taken, add the entire message text to the container
                messageElement.textContent = '[' + message.source + ']:\n' + message.message;
            }
            messageElement.classList.add('message')
            chatContainer.appendChild(messageElement)   
        });
        // Scroll to the bottom to show the latest message
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    messageInput.addEventListener('input', function() {
        const lineHeight = 20; // Adjust this value based on your styling
    
        // Calculate the remaining space between #chat-container and #chat-form
        const remainingSpace = window.innerHeight - chatContainer.offsetHeight - 10;
    
        // Calculate the desired height for #chat-form based on content and remaining space
        const desiredFormHeight = Math.min(remainingSpace, messageInput.scrollHeight);
    
        // Calculate the number of lines in the input
        const numLines = Math.floor(desiredFormHeight / lineHeight) + 1;

        // Adjust the height of #chat-form and #chat-container
        if (numLines > 2) {
            chatForm.style.height = `${Math.max(((10 * window.innerHeight) / 100), numLines * lineHeight)}px`;
            chatContainer.style.height = `${((90 * window.innerHeight) / 100) - numLines * lineHeight}px`;
        }
        else {
            chatForm.style.height = '10vh'
            chatContainer.style.height = '85vh'
        }
    });

    messageInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && event.shiftKey) {
            // Insert a newline character without triggering form submission
            event.preventDefault();
            messageInput.value += '\n';
        } 
    });

    messageInput.addEventListener('keyup', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            // Trigger the form submission
            chatForm.dispatchEvent(new Event('submit'))
        }
    })

    chatForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const userMessage = messageInput.value.trim();
        if (userMessage === '') return;

        chatMessages.push({
            source: 'You',
            message: userMessage
        });
        chatMessages.push({
            source: 'AI',
            message: 'Thinking...'
        })
        updateChatWindow()
        chatMessages.pop()
        const requestBody = {
            messages: chatMessages
        };

        // Get the agents response
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        })
        .then(response => response.json())
        .then(data => {
            // Add response to message list and update the chat window
            const aiMessage = data;
            chatMessages.push(aiMessage);
            updateChatWindow();
        })
        .catch(error => {
            console.error('Error:', error);
            chatMessages.push({source:'AI', message: 'Hmm, I seem to have encountered an error in my line of thought'});
            updateChatWindow();
        });

        // Clear the input field
        messageInput.value = '';
    });

    // Initial chat message (e.g., welcome message)
    chatMessages.push({
        source: 'AI',
        message: 'Welcome to the chat!'
    });

    // Update the chat window on page load
    updateChatWindow();
});

window.onload = function() {
    // Hide the chat interface and show loading spinner initially
    document.getElementById('chat-container').style.display = 'none';
    document.getElementById('chat-form').style.display = 'none';
    document.getElementsByClassName('loader-container')[0].style.display= 'flex';

    // Function to extract query parameter from URL
    const urlParams = new URLSearchParams(window.location.search);
    let code = urlParams.get('code');

    if (code == null) {
        window.location.href = '/'
    }

    // Send a POST request to '/api/auth' with the code parameter
    fetch('/api/auth', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 'code': code })
    })
    .then(response => {
        if (response.status === 200) {
            // If response status is 200, hide loading spinner and show chat interface
            document.getElementById('chat-container').style.display = 'flex';
            document.getElementById('chat-form').style.display = 'flex';
            document.getElementsByClassName('loader-container')[0].style.display='none';
            console.log('Authentication successful');
        } else {
            // If response status is not 200, redirect the user to the login page
            alert('Authentication failed, click ok to return to the login page')
            console.log('Authentication failed, redirecting to login page');
            window.location.href = '/';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        // Redirect the user to the login page on error
        window.location.href = '/';
    });
};