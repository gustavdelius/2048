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

    let score = 0;
    let lockOperations = false;
    let aiInterval = null;
    let aiSpeed = 500;
    
    // NEW TILE MANAGER
    let tiles = [];
    let tileIdCounter = 0;

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

    class Tile {
        constructor(r, c, val, isInitial = false) {
            this.id = tileIdCounter++;
            this.r = r;
            this.c = c;
            this.val = val;
            this.mergedInto = null;
            this.toRemove = false;
            
            this.el = document.createElement("div");
            this.el.className = `tile tile-${this.val > 2048 ? 'super' : this.val}`;
            
            this.inner = document.createElement("div");
            this.inner.className = isInitial ? "tile-inner" : "tile-inner tile-new";
            this.inner.textContent = this.val;
            
            this.el.appendChild(this.inner);
            tileContainer.appendChild(this.el);
            this.updatePosition();
        }
        
        updatePosition() {
            const pos = getPosition(this.r, this.c);
            this.el.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
        }
        
        updateValue(newVal) {
            this.val = newVal;
            this.el.className = `tile tile-${this.val > 2048 ? 'super' : this.val}`;
            this.inner.textContent = this.val;
            
            // Retrigger merge animation
            this.inner.className = "tile-inner tile-merged";
        }
        
        remove() {
            if (this.el.parentNode) {
                this.el.parentNode.removeChild(this.el);
            }
        }
    }

    function init() {
        document.getElementById("restart-btn").addEventListener("click", resetGame);
        document.getElementById("retry-btn").addEventListener("click", resetGame);
        setupInput();
        setupAI();
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
            
            // Clear existing tiles completely on load
            tiles.forEach(t => t.remove());
            tiles = [];
            
            for (let r = 0; r < gridSize; r++) {
                for (let c = 0; c < gridSize; c++) {
                    if (data.board[r][c] > 0) {
                        tiles.push(new Tile(r, c, data.board[r][c], true));
                    }
                }
            }
            
            updateGameState(data);
        } catch (e) {
            console.error("Error fetching state:", e);
        }
    }

    async function makeMove(action) {
        lockOperations = true;
        try {
            const res = await fetch("/api/move", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: action })
            });
            const data = await res.json();
            
            handleMoveResponse(data, action);
        } catch (e) {
            console.error("Error making move:", e);
            lockOperations = false;
        }
    }

    async function makeAIMove() {
        if (lockOperations && aiInterval) return; 
        
        lockOperations = true;
        try {
            const res = await fetch("/api/ai_move", {
                method: "POST",
                headers: { "Content-Type": "application/json" }
            });
            const data = await res.json();
            
            if (res.ok) {
                handleMoveResponse(data, data.action);
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

    function handleMoveResponse(data, action) {
        score += Math.floor(data.reward * 100);
        scoreContainer.textContent = score;

        if (data.valid_move && action !== undefined && action !== null) {
            // 1. Simulate the slide locally to animate
            slideTilesLocally(action);
            
            // 2. Wait for CSS transition (100ms) to finish, then reconcile state
            setTimeout(() => {
                reconcileBoard(data.board);
                updateGameState(data);
            }, 100);
        } else {
            // Invalid move or no action, just reconcile immediately
            reconcileBoard(data.board);
            updateGameState(data);
            
            if (!data.valid_move && data.game_over) {
                stopAI();
            }
        }
    }

    async function resetGame() {
        messageContainer.classList.remove("game-over");
        stopAI();
        lockOperations = true;
        score = 0;
        try {
            const res = await fetch("/api/reset", { method: "POST" });
            const data = await res.json();
            
            tiles.forEach(t => t.remove());
            tiles = [];
            
            for (let r = 0; r < gridSize; r++) {
                for (let c = 0; c < gridSize; c++) {
                    if (data.board[r][c] > 0) {
                        tiles.push(new Tile(r, c, data.board[r][c], true));
                    }
                }
            }
            
            updateGameState(data);
        } catch (e) {
            console.error("Error resetting game:", e);
            lockOperations = false;
        }
    }

    function updateGameState(data) {
        if (data.game_over) {
            messageText.textContent = "Game over!";
            messageContainer.classList.add("game-over");
            stopAI();
        }
        
        setTimeout(() => {
            lockOperations = false;
        }, 50);
    }

    function getLocalGrid() {
        let grid = [[null,null,null], [null,null,null], [null,null,null]];
        for (let t of tiles) {
            if (!t.toRemove) grid[t.r][t.c] = t;
        }
        return grid;
    }

    function slideTilesLocally(action) {
        let grid = getLocalGrid();
        
        let isVertical = (action === 0 || action === 1);
        let isForward = (action === 1 || action === 3); 
        
        for (let i = 0; i < gridSize; i++) {
            let line = []; 
            for (let j = 0; j < gridSize; j++) {
                let r = isVertical ? j : i;
                let c = isVertical ? i : j;
                if (grid[r][c]) line.push(grid[r][c]);
            }
            
            if (isForward) line.reverse();
            
            let newLine = [];
            let skip = false;
            for (let k = 0; k < line.length; k++) {
                if (skip) {
                    skip = false;
                    continue;
                }
                if (k + 1 < line.length && line[k].val === line[k+1].val) {
                    let keeper = line[k];
                    let removed = line[k+1];
                    removed.toRemove = true;
                    removed.mergedInto = keeper;
                    newLine.push(keeper);
                    skip = true;
                } else {
                    newLine.push(line[k]);
                }
            }
            
            for (let k = 0; k < newLine.length; k++) {
                let targetIdx = isForward ? (gridSize - 1 - k) : k;
                let targetR = isVertical ? targetIdx : i;
                let targetC = isVertical ? i : targetIdx;
                
                let t = newLine[k];
                if (t.r !== targetR || t.c !== targetC) {
                    t.r = targetR;
                    t.c = targetC;
                    t.updatePosition();
                }
            }
            
            for (let t of line) {
                if (t.toRemove && t.mergedInto) {
                    if (t.r !== t.mergedInto.r || t.c !== t.mergedInto.c) {
                        t.r = t.mergedInto.r;
                        t.c = t.mergedInto.c;
                        t.updatePosition();
                    }
                }
            }
        }
    }

    function reconcileBoard(serverBoard) {
        let activeTiles = [];
        
        // Remove destroyed tiles, double values of keepers
        for (let t of tiles) {
            if (t.toRemove) {
                t.remove();
            } else {
                let mergedTiles = tiles.filter(removed => removed.toRemove && removed.mergedInto === t);
                if (mergedTiles.length > 0) {
                    t.updateValue(t.val * 2);
                } else {
                    // strip animation classes so it doesn't replay
                    t.inner.className = "tile-inner"; 
                }
                activeTiles.push(t);
            }
        }
        
        tiles = activeTiles;
        
        // Spawn any new tiles
        for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
                let serverVal = serverBoard[r][c];
                let localTile = tiles.find(t => t.r === r && t.c === c);
                
                if (serverVal > 0 && !localTile) {
                    tiles.push(new Tile(r, c, serverVal));
                } else if (serverVal > 0 && localTile && localTile.val !== serverVal) {
                    // Failsafe 
                    localTile.updateValue(serverVal);
                } else if (serverVal === 0 && localTile) {
                    // Failsafe
                    localTile.remove();
                    tiles = tiles.filter(t => t !== localTile);
                }
            }
        }
    }

    init();
});
