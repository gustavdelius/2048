document.addEventListener("DOMContentLoaded", () => {
    const gridContainer = document.getElementById("grid");
    const tileContainer = document.getElementById("tile-container");
    const scoreContainer = document.querySelector(".score-container");
    const messageContainer = document.getElementById("game-message");
    const messageText = messageContainer.querySelector("p");
    
    const startAIBtn = document.getElementById("start-ai-btn");
    const stopAIBtn = document.getElementById("stop-ai-btn");
    const aiSpeedSlider = document.getElementById("ai-speed");
    const aiSpeedVal = document.getElementById("speed-val");

    
    let board = null;
    let score = 0;
    let lockOperations = false;
    let prevBoard = null;
    let aiInterval = null;
    let aiSpeed = 500;


    // Constants based on CSS
    const gridSize = 3;
    const gapSize = 15;
    const gameSize = 400;
    const tileSize = (gameSize - gapSize * (gridSize + 1)) / gridSize;

    function getPosition(row, col) {
        return {
            x: gapSize + col * (tileSize + gapSize),
            y: gapSize + row * (tileSize + gapSize)
        };
    }

    // Initialize game
    function init() {
        document.getElementById("restart-btn").addEventListener("click", resetGame);
        document.getElementById("retry-btn").addEventListener("click", resetGame);
        setupInput();
        setupAI();
        
        // Fetch initial state
        fetchState();
    }

    function setupAI() {
        if (aiSpeedSlider) aiSpeed = parseInt(aiSpeedSlider.value, 10);
        
        startAIBtn.addEventListener("click", startAI);
        stopAIBtn.addEventListener("click", stopAI);
        
        aiSpeedSlider.addEventListener("input", (e) => {
            aiSpeed = parseInt(e.target.value, 10);
            aiSpeedVal.textContent = aiSpeed;
            if (aiInterval) {
                // Restart interval with new speed
                stopAI();
                startAI();
            }
        });
    }

    function startAI() {
        if (aiInterval) return;
        startAIBtn.disabled = true;
        stopAIBtn.disabled = false;
        startAIBtn.style.cursor = 'not-allowed';
        startAIBtn.style.background = '#ccc';
        stopAIBtn.style.cursor = 'pointer';
        stopAIBtn.style.background = '#8f7a66';
        
        aiInterval = setInterval(makeAIMove, aiSpeed);
    }
    
    function stopAI() {
        if (aiInterval) {
            clearInterval(aiInterval);
            aiInterval = null;
        }
        startAIBtn.disabled = false;
        stopAIBtn.disabled = true;
        startAIBtn.style.cursor = 'pointer';
        startAIBtn.style.background = '#8f7a66';
        stopAIBtn.style.cursor = 'not-allowed';
        stopAIBtn.style.background = '#ccc';
    }

    async function makeAIMove() {
        if (lockOperations && aiInterval) {
            return; 
        }
        
        lockOperations = true;
        try {
            prevBoard = JSON.parse(JSON.stringify(board));
            const res = await fetch("/api/ai_move", {
                method: "POST",
                headers: { "Content-Type": "application/json" }
            });
            const data = await res.json();
            
            if (res.ok) {
                score += Math.floor(data.reward * 100);
                updateState(data);
                
                if (data.game_over) {
                    stopAI();
                }
            } else {
                console.error("AI Move error:", data.error);
                stopAI();
                lockOperations = false;
            }
        } catch (e) {
            console.error("Error making AI move:", e);
            stopAI();
            lockOperations = false;
        }
    }


    function setupInput() {
        document.addEventListener("keydown", event => {
            if (lockOperations) return;
            
            const key = event.key;
            let action = -1;
            
            if (key === "ArrowUp" || key === "w" || key === "W") action = 0;
            else if (key === "ArrowDown" || key === "s" || key === "S") action = 1;
            else if (key === "ArrowLeft" || key === "a" || key === "A") action = 2;
            else if (key === "ArrowRight" || key === "d" || key === "D") action = 3;
            
            if (action !== -1) {
                event.preventDefault();
                makeMove(action);
            }
        });
    }

    async function fetchState() {
        try {
            const res = await fetch("/api/state");
            const data = await res.json();
            updateState(data);
        } catch (e) {
            console.error("Error fetching state:", e);
        }
    }

    async function makeMove(action) {
        lockOperations = true;
        try {
            prevBoard = JSON.parse(JSON.stringify(board));
            const res = await fetch("/api/move", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action })
            });
            const data = await res.json();
            
            if (data.valid_move) {
                // If it was a valid move, increment score
                score += Math.floor(data.reward * 100); 
                updateState(data);
            } else {
                // Invalid move, apply the negative reward to score
                score += Math.floor(data.reward * 100);
                scoreContainer.textContent = score;
                // Unlock immediately
                lockOperations = false;
            }
        } catch (e) {
            console.error("Error making move:", e);
            lockOperations = false;
        }
    }

    async function resetGame() {
        messageContainer.classList.remove("game-over");
        // Don't stop AI automatically on manual reset unless desired, but it's safer
        stopAI();
        lockOperations = true;
        score = 0;
        prevBoard = null;
        try {
            const res = await fetch("/api/reset", {
                method: "POST"
            });
            const data = await res.json();
            updateState(data);
        } catch (e) {
            console.error("Error resetting game:", e);
            lockOperations = false;
        }
    }

    function updateState(data) {
        board = data.board;
        scoreContainer.textContent = score;
        
        renderTickets();
        
        if (data.game_over) {
            messageText.textContent = "Game over!";
            messageContainer.classList.add("game-over");
        }
        
        // Unlock inputs for next move after transitions happen
        setTimeout(() => {
            lockOperations = false;
        }, 150);
    }
    
    // Create UI representations from the 2D array
    function renderTickets() {
        tileContainer.innerHTML = '';
        
        // In a more complex React-like app, we'd preserve DOM elements and animate movements 
        // to show slides and merges. For simplicity here, we re-render entirely.
        // The actual sliding animations can be complex to calculate from just raw new boards.
        // We'll just fade in/pop the actual final board state to start.
        for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
                const val = board[r][c];
                if (val !== 0) {
                    // Check if it's new (in prevBoard it was 0)
                    let isNew = false;
                    let isMerged = false;
                    
                    if (prevBoard) {
                        // Very rough heuristic for animations:
                        // If cell was empty before, it's either new tile or slid tile. 
                        // If cell was a smaller tile, it was merged.
                        if (prevBoard[r][c] !== 0 && prevBoard[r][c] < val) {
                            isMerged = true;
                        } else if (prevBoard[r][c] === 0) {
                            isNew = true; // Might just be slid, but pop effect is ok
                        }
                    } else {
                        isNew = true;
                    }

                    createTile(r, c, val, isNew, isMerged);
                }
            }
        }
    }

    function createTile(r, c, value, isNew, isMerged) {
        const d_wrapper = document.createElement("div");
        d_wrapper.className = `tile tile-${value > 2048 ? 'super' : value}`;
        
        const pos = getPosition(r, c);
        d_wrapper.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
        
        const d_inner = document.createElement("div");
        d_inner.className = "tile-inner";
        d_inner.textContent = value;
        
        if (isMerged) {
            d_wrapper.classList.add("tile-merged");
        }
        
        d_wrapper.appendChild(d_inner);
        tileContainer.appendChild(d_wrapper);
    }

    init();
});
