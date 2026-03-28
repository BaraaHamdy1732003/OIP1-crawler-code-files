async function search() {
    const query = document.getElementById("query").value;

    const response = await fetch("/search", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: query })
    });

    const data = await response.json();

    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    data.forEach(item => {
        const div = document.createElement("div");
        div.className = "result-item";

        div.innerHTML = `
            <a href="${item.url}" target="_blank">${item.url}</a>
            <div class="score">Relevance score: ${item.score.toFixed(4)}</div>
        `;

        resultsDiv.appendChild(div);
    });
}