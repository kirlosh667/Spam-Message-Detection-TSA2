document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const statAccuracy = document.getElementById('statAccuracy');
    const statSamples = document.getElementById('statSamples');
    const statVocab = document.getElementById('statVocab');
    
    const messageInput = document.getElementById('messageInput');
    const charCount = document.getElementById('charCount');
    const btnClear = document.getElementById('btnClear');
    const btnAnalyze = document.getElementById('btnAnalyze');
    
    const toggleLower = document.getElementById('toggleLower');
    const togglePunct = document.getElementById('togglePunct');
    const toggleStop = document.getElementById('toggleStop');
    
    const examplesGrid = document.getElementById('examplesGrid');
    const tabBtns = document.querySelectorAll('.tab-btn');
    
    const emptyState = document.getElementById('emptyState');
    const resultContent = document.getElementById('resultContent');
    
    const badgeContainer = document.getElementById('badgeContainer');
    const badgeIcon = document.getElementById('badgeIcon');
    const badgeTitle = document.getElementById('badgeTitle');
    const confidenceValue = document.getElementById('confidenceValue');
    
    const spamProbText = document.getElementById('spamProbText');
    const hamProbText = document.getElementById('hamProbText');
    const spamBar = document.getElementById('spamBar');
    const hamBar = document.getElementById('hamBar');
    
    const stepOriginal = document.getElementById('stepOriginal');
    const stepCleaned = document.getElementById('stepCleaned');
    const tokenCount = document.getElementById('tokenCount');
    const stepTokens = document.getElementById('stepTokens');
    const stepKeywords = document.getElementById('stepKeywords');

    let allExamples = { spam: [], ham: [] };
    let currentFilter = 'all';

    // 1. Fetch Model Info
    fetch('/api/info')
        .then(res => res.json())
        .then(data => {
            if (data.accuracy !== undefined) {
                statAccuracy.textContent = `${data.accuracy}%`;
                statSamples.textContent = data.total_samples;
                statVocab.textContent = data.vocabulary_size;
            }
        })
        .catch(err => console.error('Failed to load dataset stats:', err));

    // 2. Fetch Sample Examples
    fetch('/api/examples')
        .then(res => res.json())
        .then(data => {
            allExamples = data;
            renderExamples();
        })
        .catch(err => console.error('Failed to load examples:', err));

    // Render Example Cards based on filter
    function renderExamples() {
        examplesGrid.innerHTML = '';
        let listToRender = [];

        if (currentFilter === 'all') {
            const spamLabeled = (allExamples.spam || []).map(ex => ({ ...ex, type: 'spam' }));
            const hamLabeled = (allExamples.ham || []).map(ex => ({ ...ex, type: 'ham' }));
            listToRender = [...spamLabeled, ...hamLabeled];
        } else if (currentFilter === 'spam') {
            listToRender = (allExamples.spam || []).map(ex => ({ ...ex, type: 'spam' }));
        } else if (currentFilter === 'ham') {
            listToRender = (allExamples.ham || []).map(ex => ({ ...ex, type: 'ham' }));
        }

        listToRender.forEach(ex => {
            const card = document.createElement('div');
            card.className = 'example-card';
            card.innerHTML = `
                <div class="example-top">
                    <span class="example-title">${escapeHtml(ex.title)}</span>
                    <span class="example-tag ${ex.type}">${ex.type === 'spam' ? 'Spam' : 'Legitimate'}</span>
                </div>
                <div class="example-snippet">${escapeHtml(ex.text)}</div>
            `;

            card.addEventListener('click', () => {
                messageInput.value = ex.text;
                updateCharCount();
                analyzeMessage();
            });

            examplesGrid.appendChild(card);
        });
    }

    // Filter Tabs Click
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.tab;
            renderExamples();
        });
    });

    // Character Counter
    function updateCharCount() {
        const len = messageInput.value.length;
        charCount.textContent = `${len} character${len === 1 ? '' : 's'}`;
    }
    messageInput.addEventListener('input', updateCharCount);

    // Clear Text
    btnClear.addEventListener('click', () => {
        messageInput.value = '';
        updateCharCount();
        emptyState.classList.remove('hidden');
        resultContent.classList.add('hidden');
        messageInput.focus();
    });

    // Analyze Message Action
    btnAnalyze.addEventListener('click', analyzeMessage);

    function analyzeMessage() {
        const text = messageInput.value.strip ? messageInput.value.strip() : messageInput.value.trim();
        
        if (!text) {
            alert('Please enter or select a message to analyze.');
            return;
        }

        // Show loading button state
        btnAnalyze.classList.add('loading');
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing NLP...';

        const payload = {
            message: text,
            do_lower: toggleLower.checked,
            remove_punct: togglePunct.checked,
            remove_stop: toggleStop.checked
        };

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) throw new Error('Prediction API error');
            return res.json();
        })
        .then(data => {
            renderResults(data);
        })
        .catch(err => {
            console.error(err);
            alert('An error occurred while analyzing the message.');
        })
        .finally(() => {
            btnAnalyze.classList.remove('loading');
            btnAnalyze.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i> Analyze Message';
        });
    }

    function renderResults(data) {
        emptyState.classList.add('hidden');
        resultContent.classList.remove('hidden');

        const isSpam = data.is_spam;

        // Badge display
        if (isSpam) {
            badgeContainer.className = 'result-badge-container spam-badge';
            badgeIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            badgeTitle.textContent = 'SPAM DETECTED';
        } else {
            badgeContainer.className = 'result-badge-container ham-badge';
            badgeIcon.innerHTML = '<i class="fa-solid fa-shield-check"></i>';
            badgeTitle.textContent = 'LEGITIMATE (NOT SPAM)';
        }

        confidenceValue.textContent = `${data.confidence}%`;

        // Probability bars
        spamProbText.textContent = `${data.spam_probability}%`;
        hamProbText.textContent = `${data.ham_probability}%`;
        
        // Timeout to trigger smooth CSS width transition
        setTimeout(() => {
            spamBar.style.width = `${data.spam_probability}%`;
            hamBar.style.width = `${data.ham_probability}%`;
        }, 50);

        // Preprocessing pipeline details
        stepOriginal.textContent = data.original_text;
        stepCleaned.textContent = data.preprocessed_text || '(Empty after preprocessing)';
        
        tokenCount.textContent = data.tokens.length;
        stepTokens.innerHTML = '';
        if (data.tokens.length === 0) {
            stepTokens.innerHTML = '<span class="token-badge" style="opacity: 0.5;">No tokens remaining</span>';
        } else {
            data.tokens.forEach(tok => {
                const badge = document.createElement('span');
                badge.className = 'token-badge';
                badge.textContent = tok;
                stepTokens.appendChild(badge);
            });
        }

        // Keywords & Weights
        stepKeywords.innerHTML = '';
        if (!data.top_keywords || data.top_keywords.length === 0) {
            stepKeywords.innerHTML = '<span style="font-size:0.85rem; color: var(--text-muted);">No prominent vocabulary features found in training dictionary.</span>';
        } else {
            data.top_keywords.forEach(kw => {
                const kwBadge = document.createElement('span');
                kwBadge.className = `kw-badge ${kw.bias === 'spam' ? 'spam-bias' : 'ham-bias'}`;
                kwBadge.innerHTML = `
                    <i class="${kw.bias === 'spam' ? 'fa-solid fa-circle-exclamation' : 'fa-solid fa-circle-check'}"></i>
                    ${escapeHtml(kw.word)}
                    <span class="kw-weight">(TF-IDF: ${kw.tfidf})</span>
                `;
                stepKeywords.appendChild(kwBadge);
            });
        }
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
