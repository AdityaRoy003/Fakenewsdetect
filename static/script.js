document.addEventListener("DOMContentLoaded", () => {
    // --- Theme Toggle ---
    const themeBtn = document.getElementById("theme-toggle");
    const htmlTag = document.documentElement;
    let isLight = false;
    
    themeBtn.addEventListener("click", () => {
        isLight = !isLight;
        if (isLight) {
            htmlTag.setAttribute("data-theme", "light");
            themeBtn.textContent = "🌙 Dark Mode";
        } else {
            htmlTag.setAttribute("data-theme", "dark");
            themeBtn.textContent = "☀️ Light Mode";
        }
    });

    // --- Drag and Drop File & Text Logic ---
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("image-upload");
    const fileNameDisplay = document.getElementById("file-name");
    const textInput = document.getElementById("article-text");

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            fileNameDisplay.textContent = fileInput.files[0].name;
        } else {
            const text = e.dataTransfer.getData("text/plain");
            if (text) {
                textInput.value = text;
            }
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) fileNameDisplay.textContent = e.target.files[0].name;
        else fileNameDisplay.textContent = "No file chosen";
    });

    // --- History Panel System ---
    const historyToggle = document.getElementById("history-toggle");
    const historyPanel = document.getElementById("history-panel");
    const closeHistory = document.getElementById("close-history");
    const historyList = document.getElementById("history-list");

    historyToggle.addEventListener("click", () => historyPanel.classList.add("open"));
    closeHistory.addEventListener("click", () => historyPanel.classList.remove("open"));

    function saveToHistory(verdict, textPreview) {
        let history = JSON.parse(localStorage.getItem("fakeNewsHistory")) || [];
        const item = {
            id: Date.now(),
            verdict: verdict, // "REAL" or "FAKE"
            text: textPreview.substring(0, 50) + "...",
            date: new Date().toLocaleString()
        };
        history.unshift(item); // Add to top
        if (history.length > 10) history.pop(); // Keep only last 10
        localStorage.setItem("fakeNewsHistory", JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        let history = JSON.parse(localStorage.getItem("fakeNewsHistory")) || [];
        historyList.innerHTML = "";
        if (history.length === 0) {
            historyList.innerHTML = "<p style='color:var(--text-muted);font-size:0.9rem'>No past scans found.</p>";
            return;
        }
        history.forEach(item => {
            const div = document.createElement("div");
            div.className = `history-item ${item.verdict === "FAKE" ? "fake" : "real"}`;
            div.innerHTML = `
                <div class="history-item-header">
                    <div class="history-date">${item.date}</div>
                    <button class="delete-history-btn" data-id="${item.id}">✖</button>
                </div>
                <div class="history-title"><strong>${item.verdict}</strong>: ${item.text}</div>
            `;
            historyList.appendChild(div);
        });
    }
    
    // Handle deleting history items
    historyList.addEventListener("click", (e) => {
        if (e.target.classList.contains("delete-history-btn")) {
            const idToRemove = parseInt(e.target.getAttribute("data-id"));
            let history = JSON.parse(localStorage.getItem("fakeNewsHistory")) || [];
            history = history.filter(item => item.id !== idToRemove);
            localStorage.setItem("fakeNewsHistory", JSON.stringify(history));
            renderHistory();
        }
    });
    
    renderHistory(); // Initial render

    // --- Flip Card Toggles ---
    const flipCard = document.getElementById("flip-card");
    document.getElementById("btn-detailed").addEventListener("click", () => {
        flipCard.classList.add("is-flipped");
        flipCard.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    document.getElementById("btn-quick").addEventListener("click", () => {
        flipCard.classList.remove("is-flipped");
        flipCard.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    // --- Settings Modal System ---
    const settingsToggle = document.getElementById("settings-toggle");
    const settingsModal = document.getElementById("settings-modal");
    const closeSettings = document.getElementById("close-settings");
    const saveSettings = document.getElementById("save-settings");
    
    // Settings inputs
    const setCred = document.getElementById("set-cred");
    const setCredVal = document.getElementById("set-cred-val");
    const setWeight = document.getElementById("set-weight");
    const setPush = document.getElementById("set-push");
    const setEmail = document.getElementById("set-email");
    const setLang = document.getElementById("set-lang");
    const setDetail = document.getElementById("set-detail");

    setCred.addEventListener("input", (e) => setCredVal.textContent = parseFloat(e.target.value).toFixed(2));

    settingsToggle.addEventListener("click", () => {
        loadSettings();
        settingsModal.classList.add("open");
    });
    
    closeSettings.addEventListener("click", () => settingsModal.classList.remove("open"));

    function loadSettings() {
        const conf = JSON.parse(localStorage.getItem("fakeNewsSettings")) || {
            minCred: 0.50, weight: 0.5, push: false, email: false, lang: 'en', detail: 'quick'
        };
        setCred.value = conf.minCred;
        setCredVal.textContent = parseFloat(conf.minCred).toFixed(2);
        setWeight.value = conf.weight;
        setPush.checked = conf.push;
        setEmail.checked = conf.email;
        setLang.value = conf.lang;
        setDetail.value = conf.detail;
    }

        saveSettings.addEventListener("click", () => {
        const conf = {
            minCred: parseFloat(setCred.value),
            weight: parseFloat(setWeight.value),
            push: setPush.checked,
            email: setEmail.checked,
            lang: setLang.value,
            detail: setDetail.value
        };
        localStorage.setItem("fakeNewsSettings", JSON.stringify(conf));
        settingsModal.classList.remove("open");
        
        if (conf.push && Notification.permission !== "granted") {
            Notification.requestPermission();
        }
        if(conf.email) alert("Email alerts configured! Notifications will be sent to your registered address.");
    });
    
    // --- Intelligence Hub System ---
    const intelToggle = document.getElementById("intel-toggle");
    const intelModal = document.getElementById("intel-modal");
    const closeIntel = document.getElementById("close-intel");

    intelToggle.addEventListener("click", () => intelModal.classList.add("open"));
    closeIntel.addEventListener("click", () => intelModal.classList.remove("open"));
    
    // Load initially to apply preferences
    loadSettings();

    // --- Form Submit Logic ---
    const form = document.getElementById("predict-form");
    const resultContainer = document.getElementById("result-container");
    const loadingSpinner = document.getElementById("loading-spinner");
    const resultContent = document.getElementById("result-content");
    const btnSubmit = document.getElementById("submit-btn");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Reset UI
        flipCard.classList.remove("is-flipped");
        resultContainer.style.display = "block";
        loadingSpinner.style.display = "block";
        resultContent.style.display = "none";
        btnSubmit.disabled = true;
        btnSubmit.textContent = "Analyzing...";

        const formData = new FormData();
        const articleTextVal = textInput.value;
        formData.append("text", articleTextVal);
        
        // Grab values from settings cache
        const conf = JSON.parse(localStorage.getItem("fakeNewsSettings")) || { minCred: 0.50, weight: 0.5, push: false, email: false, lang: 'en', detail: 'quick' };
        
        // Use the weight slider to balance author vs domain credibility
        const metadata = {
            author_credibility: 1.0 - conf.weight,
            domain_reputation: conf.weight
        };
        formData.append("metadata", JSON.stringify(metadata));

        if (fileInput.files.length > 0) {
            formData.append("file", fileInput.files[0]);
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Network response was not ok");
            const data = await response.json();

            // Hide Loading, Show Content
            loadingSpinner.style.display = "none";
            resultContent.style.display = "block";

            // Verdict Banner with Threshold Override & Translations
            const banner = document.getElementById("verdict-banner");
            const verdictText = document.getElementById("verdict-text");
            banner.className = "verdict-banner"; 
            
            // Override the default 0.5 threshold with the user's custom Min Credibility setting
            const isFakeAdjusted = data.fake_probability >= conf.minCred;
            
            // Translations
            let realStr = "✅ REAL";
            let fakeStr = "🚨 FAKE NEWS";
            if (conf.lang === 'es') { realStr += " (Verdadero)"; fakeStr += " (Falso)"; }
            else if (conf.lang === 'fr') { realStr += " (Vrai)"; fakeStr += " (Faux)"; }
            else if (conf.lang === 'de') { realStr += " (Wahr)"; fakeStr += " (Falsch)"; }

            if (isFakeAdjusted) {
                banner.classList.add("fake");
                verdictText.textContent = fakeStr;
                
                // Trigger Push Notification if enabled
                if (conf.push && Notification.permission === "granted") {
                    new Notification("Fake News Detector 🚨", { body: "Suspicious content flagged based on your security thresholds!" });
                }
            } else {
                banner.classList.add("real");
                verdictText.textContent = realStr;
            }
            
            requestAnimationFrame(() => {
                banner.classList.add("show", "pulse");
            });

            // Stats
            const probPct = (data.fake_probability * 100).toFixed(2) + "%";
            document.getElementById("prob-text").textContent = probPct;
            document.getElementById("prob-fill").style.width = "0%"; // reset
            document.getElementById("prob-fill").style.background = isFakeAdjusted ? "var(--danger)" : "var(--success)";
            
            setTimeout(() => {
                document.getElementById("prob-fill").style.width = probPct;
            }, 50);

            // Sources
            document.getElementById("sources-count").textContent = data.sources_found || 0;
            const sourcesList = document.getElementById("sources-list");
            if (data.sources_found > 0) {
                sourcesList.innerHTML = data.search_context.replace(/\n/g, "<br>");
            } else {
                sourcesList.innerHTML = "<i>No reputable sources found online for this claim.</i>";
            }
            
            // Community Notes
            const cNotesContainer = document.getElementById("community-notes-container");
            const cNotesList = document.getElementById("community-notes-list");
            if (data.community_notes && data.community_notes.length > 0) {
                cNotesContainer.style.display = "block";
                cNotesList.innerHTML = "";
                data.community_notes.forEach(note => {
                    const div = document.createElement("div");
                    div.className = "community-note";
                    div.textContent = note;
                    cNotesList.appendChild(div);
                });
            } else {
                cNotesContainer.style.display = "none";
            }

            // Save to History using the custom adjusted verdict
            saveToHistory(isFakeAdjusted ? "FAKE" : "REAL", articleTextVal);

            // --- Populate Explainability (Back of Card) ---
            document.getElementById("seg-text").style.width = data.contribution_text + "%";
            document.getElementById("val-text").textContent = data.contribution_text;

            document.getElementById("seg-img").style.width = data.contribution_image + "%";
            document.getElementById("val-img").textContent = data.contribution_image;

            document.getElementById("seg-meta").style.width = data.contribution_metadata + "%";
            document.getElementById("val-meta").textContent = data.contribution_metadata;

            document.getElementById("ai-highlights").textContent = data.explainability_highlights;
            
            // Bias & Media Analysis
            document.getElementById("bias-indicator").style.left = (data.bias_score * 100) + "%";
            const mediaAuthText = document.getElementById("media-auth-text");
            mediaAuthText.textContent = data.media_authenticity;
            mediaAuthText.style.color = data.media_authenticity.includes("⚠️") ? "var(--warning)" : "var(--success)";
            
            // Detail Mode Auto-Flip
            if (conf.detail === "full") {
                setTimeout(() => {
                    flipCard.classList.add("is-flipped");
                }, 1000);
            }

        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while connecting to the backend.");
            loadingSpinner.style.display = "none";
            resultContainer.style.display = "none";
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.textContent = "Check Authenticity";
            resultContainer.scrollIntoView({ behavior: "smooth" });
        }
    });
});
