document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const contentBody = document.getElementById('content-body');
    const navItems = document.querySelectorAll('.nav-item');
    const searchInput = document.getElementById('job-search');

    // State
    let currentView = 'dashboard';
    let jobsData = {
        found: [],
        evaluated: []
    };

    // Initialize
    init();

    async function init() {
        // Setup Navigation
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const target = e.currentTarget.getAttribute('data-target');
                if (target === 'settings') return; // Placeholder

                navItems.forEach(nav => nav.classList.remove('active'));
                e.currentTarget.classList.add('active');

                currentView = target;
                renderView();
            });
        });

        // Setup Search
        searchInput.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            renderJobsGrid(getFilteredJobs(term));
        });

        // Load initial data
        await fetchAllData();
        renderView();
    }

    async function fetchAllData() {
        try {
            showLoading();
            const [foundRes, evalRes] = await Promise.all([
                fetch('/api/jobs/found').catch(() => ({ ok: false })),
                fetch('/api/jobs/evaluated').catch(() => ({ ok: false }))
            ]);

            if (foundRes.ok) jobsData.found = await foundRes.json();
            if (evalRes.ok) jobsData.evaluated = await evalRes.json();

        } catch (error) {
            console.error("Error fetching data:", error);
            contentBody.innerHTML = `
                <div class="glass-panel" style="padding: 2rem; text-align: center; color: var(--danger);">
                    <i class="ph ph-warning-circle" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                    <h3>Failed to load data</h3>
                    <p>Make sure the backend is running and the JSON files exist.</p>
                </div>
            `;
        }
    }

    function showLoading() {
        contentBody.innerHTML = `
            <div class="loading-state">
                <div class="spinner"></div>
                <p>Loading your job data...</p>
            </div>
        `;
    }

    function renderView() {
        if (currentView === 'dashboard') {
            renderDashboard();
        } else if (currentView === 'jobs-found') {
            renderJobsView('Found Jobs', jobsData.found, false);
        } else if (currentView === 'jobs-evaluated') {
            renderJobsView('Evaluated Jobs', jobsData.evaluated, true);
        }
    }

    function renderDashboard() {
        const totalFound = jobsData.found.length;
        const totalEvaluated = jobsData.evaluated.length;
        const recentJobs = jobsData.found.slice(0, 6);

        contentBody.innerHTML = `
            <div class="dashboard-grid">
                <div class="stat-card glass-panel">
                    <div class="stat-header">
                        <span>Total Found</span>
                        <div class="stat-icon"><i class="ph ph-briefcase"></i></div>
                    </div>
                    <div class="stat-value">${totalFound}</div>
                </div>
                <div class="stat-card glass-panel" style="--accent-primary: var(--accent-secondary);">
                    <div class="stat-header">
                        <span>Evaluated</span>
                        <div class="stat-icon" style="color: var(--accent-secondary); background: rgba(236, 72, 153, 0.1);"><i class="ph ph-chart-bar"></i></div>
                    </div>
                    <div class="stat-value">${totalEvaluated}</div>
                </div>
                <div class="stat-card glass-panel" style="--accent-primary: var(--success);">
                    <div class="stat-header">
                        <span>High Matches (>70%)</span>
                        <div class="stat-icon" style="color: var(--success); background: rgba(16, 185, 129, 0.1);"><i class="ph ph-target"></i></div>
                    </div>
                    <div class="stat-value">
                        ${jobsData.evaluated.filter(j => (j.match_score || 0) >= 70).length}
                    </div>
                </div>
            </div>
            
            <div class="section-header">
                <h2 class="section-title">Recently Found Jobs</h2>
                <button class="btn-primary" style="background: var(--bg-surface); color: var(--text-main); font-weight: 500;" onclick="document.querySelector('[data-target=\\'jobs-found\\']').click()">
                    View All
                </button>
            </div>
            <div class="jobs-grid" id="jobs-grid-container"></div>
        `;

        renderJobsGrid(recentJobs, false);
    }

    function renderJobsView(title, jobs, isEvaluated) {
        contentBody.innerHTML = `
            <div class="section-header">
                <h2 class="section-title">${title} <span style="color: var(--text-muted); font-size: 1rem; font-weight: 400;">(${jobs.length})</span></h2>
            </div>
            <div class="jobs-grid" id="jobs-grid-container"></div>
        `;

        renderJobsGrid(jobs, isEvaluated);
    }

    function getFilteredJobs(term) {
        const list = currentView === 'jobs-found' ? jobsData.found :
            (currentView === 'jobs-evaluated' ? jobsData.evaluated : jobsData.found.slice(0, 6));

        if (!term) return list;

        return list.filter(job => {
            const titleMatch = (job.title || '').toLowerCase().includes(term);
            const compMatch = (job.company || '').toLowerCase().includes(term);
            const descMatch = (job.description || '').toLowerCase().includes(term);
            return titleMatch || compMatch || descMatch;
        });
    }

    function renderJobsGrid(jobs, isEvaluated = false) {
        const container = document.getElementById('jobs-grid-container');
        if (!container) return;

        if (jobs.length === 0) {
            container.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--text-muted);">
                    <i class="ph ph-magnifying-glass" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                    <p>No jobs found matching your criteria.</p>
                </div>
            `;
            return;
        }

        let html = '';
        jobs.forEach(job => {
            // Determine score class
            let scoreHtml = '';
            if (isEvaluated && job.match_score !== undefined) {
                const s = job.match_score;
                let c = 'score-high';
                if (s < 50) c = 'score-low';
                else if (s < 75) c = 'score-med';

                scoreHtml = `<div class="match-score ${c}" title="Match Score">${s}</div>`;
            }

            const title = job.title || 'Untitled Position';
            const company = job.company || 'Unknown Company';
            let locationText = job.location || 'Remote/Unknown';
            // Some jobs have locations as a list/dict, trying to stringify simply
            if (typeof locationText === 'object') {
                locationText = Array.isArray(locationText) ? locationText.join(', ') : 'Location Data';
            }
            // Some source logic
            const sourceText = job.source || 'Scraped';
            const postedText = job.timestamp ? new Date(job.timestamp).toLocaleDateString() : 'Unknown date';

            // Clean description roughly
            let desc = job.description || 'No description available.';
            desc = desc.replace(/<[^>]*>?/gm, ''); // strip HTML

            html += `
                <div class="job-card glass-panel">
                    <div class="job-header">
                        <div>
                            <div class="job-company"><i class="ph-fill ph-buildings"></i> ${company}</div>
                            <h3 class="job-title">${title}</h3>
                        </div>
                        ${scoreHtml}
                    </div>
                    
                    <div class="job-meta-grid">
                        <div class="job-meta"><i class="ph ph-map-pin"></i> ${locationText}</div>
                        <div class="job-meta"><i class="ph ph-clock"></i> ${postedText}</div>
                    </div>
                    
                    <p class="job-desc">${desc}</p>
                    
                    <div class="job-footer">
                        <span class="job-source">${sourceText}</span>
                        <a href="${job.job_url || job.url || '#'}" target="_blank" rel="noopener noreferrer" class="btn-primary">
                            <span>View Job</span>
                            <i class="ph ph-arrow-up-right"></i>
                        </a>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    }
});
